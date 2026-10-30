"""
CerebroSpheroid Cloud Synchronization Client.
Enables transparent, continuous checkpoint resumption across Google Colab, Kaggle, and Local Workstations.

Endpoints:
- Default Sync URL: https://bioflo.in/api/cerebro/sync.php
- Storage Policy: Strictly bounded to latest.pt and best.pt (capped at ~900 MB total server space).
- Chunked Upload: Bypasses Cloudflare 100 MB limits by streaming 10 MB chunks.
"""

import os
import sys
import time
import json
import threading
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Tuple

DEFAULT_URL = os.environ.get("BIOFLO_SYNC_URL", "https://bioflo.in/api/cerebro/sync.php")
DEFAULT_TOKEN = os.environ.get("BIOFLO_SYNC_TOKEN", "CerebroSpheroid_Sync_Secure_2026")

_IN_FLIGHT_UPLOADS: Dict[str, int] = {}
_LAST_BEST_UPLOAD_TIME: float = 0.0
_BEST_COOLDOWN_SECONDS: float = 300.0  # 5 minutes minimum between best.pt uploads
_UPLOAD_LOCK = threading.Lock()


def detect_source_environment() -> str:
    """Detects whether current execution is in Google Colab, Kaggle, or Local."""
    if "COLAB_GPU" in os.environ or "COLAB_TPU_ADDR" in os.environ or os.path.exists("/content"):
        return "colab"
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
        return "kaggle"
    return "local"


def check_remote_manifest(url: Optional[str] = None, token: Optional[str] = None) -> Dict[str, Any]:
    """Queries bioflo.in to retrieve checkpoint status, step numbers, and loss metrics."""
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN

    req = urllib.request.Request(
        f"{sync_url}?action=status",
        headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroCloudSync/1.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"⚠️ [cloud_sync] Unable to reach sync hub ({sync_url}): {e}")
        return {"status": "offline", "has_latest": False, "has_best": False, "manifest": {}}


def pull_checkpoint(
    local_dest: str,
    target_type: str = "latest",
    url: Optional[str] = None,
    token: Optional[str] = None,
    show_progress: bool = True
) -> bool:
    """
    Downloads latest.pt or best.pt from bioflo.in with real-time streaming progress
    and automatic HTTP Range resumption across network drops or server timeouts.
    Returns True if successfully downloaded and verified.
    """
    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN

    os.makedirs(os.path.dirname(os.path.abspath(local_dest)), exist_ok=True)
    download_url = f"{sync_url}?action=download&type={target_type}"

    tmp_dest = local_dest + ".part"
    try:
        print(f"🔄 [cloud_sync] Downloading {target_type}.pt from {sync_url}...")
        t0 = time.time()

        # 1. Probe total size using HTTP Range
        total_bytes = 0
        try:
            probe_req = urllib.request.Request(
                download_url,
                headers={"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroCloudSync/1.0", "Range": "bytes=0-0"}
            )
            with urllib.request.urlopen(probe_req, timeout=15) as p_resp:
                cr = p_resp.headers.get("Content-Range", "")
                if "/" in cr:
                    total_bytes = int(cr.split("/")[-1])
        except Exception:
            pass

        downloaded = 0
        if os.path.exists(tmp_dest):
            downloaded = os.path.getsize(tmp_dest)
            if total_bytes > 0 and downloaded >= total_bytes:
                downloaded = total_bytes

        max_retries = 8
        retry = 0
        while retry < max_retries:
            headers = {"Authorization": f"Bearer {sync_token}", "User-Agent": "CerebroCloudSync/1.0"}
            if downloaded > 0 and total_bytes > 0:
                headers["Range"] = f"bytes={downloaded}-{total_bytes - 1}"
                mode = "ab"
            else:
                mode = "wb"
                downloaded = 0

            req = urllib.request.Request(download_url, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=120) as resp, open(tmp_dest, mode) as f:
                    if total_bytes == 0:
                        total_bytes = int(resp.headers.get("Content-Length", 0))

                    while True:
                        chunk = resp.read(2 * 1024 * 1024)  # 2 MB chunks
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if show_progress and total_bytes > 0:
                            pct = (downloaded / total_bytes) * 100
                            speed = (downloaded / (1024 * 1024)) / max(time.time() - t0, 0.1)
                            print(f"\r  Downloading: {downloaded / 1e6:.1f} / {total_bytes / 1e6:.1f} MB "
                                  f"({pct:.1f}%) | {speed:.1f} MB/s", end="", flush=True)

                if total_bytes > 0 and downloaded >= total_bytes:
                    break
                elif total_bytes > 0 and downloaded < total_bytes:
                    print(f"\n⚠️ [cloud_sync] Transfer paused at {downloaded / 1e6:.1f} / {total_bytes / 1e6:.1f} MB. Resuming via HTTP Range...")
                    retry += 1
                    time.sleep(1.0)
                else:
                    break
            except Exception as conn_err:
                retry += 1
                print(f"\n⚠️ [cloud_sync] Stream interrupted ({conn_err}). Resuming in {retry}s...")
                time.sleep(retry)

        if os.path.exists(tmp_dest):
            actual_sz = os.path.getsize(tmp_dest)
            if total_bytes > 0 and actual_sz < total_bytes:
                print(f"\n❌ [cloud_sync] Incomplete download ({actual_sz:,} of {total_bytes:,} bytes). Keeping .part for future resumption.")
                return False

            import zipfile
            if not zipfile.is_zipfile(tmp_dest):
                print(f"\n⚠️ [cloud_sync] Checkpoint failed zip archive integrity check. Discarding corrupt download.")
                os.remove(tmp_dest)
                return False

            os.replace(tmp_dest, local_dest)
            dt = time.time() - t0
            sz_mb = os.path.getsize(local_dest) / 1e6
            print(f"\n✅ [cloud_sync] Restored {target_type}.pt ({sz_mb:.1f} MB) in {dt:.1f}s -> {local_dest}")
            return True
        else:
            print(f"\n⚠️ [cloud_sync] Downloaded file not found.")
            return False
    except Exception as e:
        print(f"\n⚠️ [cloud_sync] Failed to pull {target_type} checkpoint: {e}")
        return False


def push_checkpoint(
    file_path: str,
    step: int,
    loss: float,
    target_type: str = "latest",
    source_node: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
    chunk_size_mb: int = 10
) -> bool:
    """
    Uploads a checkpoint to bioflo.in using chunked streaming.
    Bypasses Cloudflare 100 MB limits and shared hosting PHP upload ceilings.
    Enforces a 5-minute wait between successive 'best.pt' uploads to minimize traffic.
    """
    if not os.path.exists(file_path):
        print(f"⚠️ [cloud_sync] Cannot push: {file_path} not found.")
        return False

    global _IN_FLIGHT_UPLOADS, _LAST_BEST_UPLOAD_TIME
    with _UPLOAD_LOCK:
        now = time.time()
        if target_type == "best":
            elapsed = now - _LAST_BEST_UPLOAD_TIME
            if elapsed < _BEST_COOLDOWN_SECONDS:
                remaining = int(_BEST_COOLDOWN_SECONDS - elapsed)
                print(f"⏳ [cloud_sync] Best model cooldown active: skipping best.pt upload ({remaining}s remaining until next 5-min window).")
                return False

        if _IN_FLIGHT_UPLOADS.get(target_type, 0) > step:
            # A more recent step is already being or has been uploaded
            return False
        _IN_FLIGHT_UPLOADS[target_type] = step

    sync_url = url or DEFAULT_URL
    sync_token = token or DEFAULT_TOKEN
    node_name = source_node or detect_source_environment()

    file_size = os.path.getsize(file_path)
    chunk_size = chunk_size_mb * 1024 * 1024
    total_chunks = max(1, (file_size + chunk_size - 1) // chunk_size)

    print(f"\n🌐 [cloud_sync] Mirroring {target_type}.pt (Step {step:,}, Loss {loss:.4f}) "
          f"to {sync_url} across {total_chunks} chunk(s)...")

    t0 = time.time()
    try:
        with open(file_path, "rb") as f:
            for idx in range(total_chunks):
                chunk_data = f.read(chunk_size)
                if not chunk_data:
                    break

                # Multipart boundary assembly
                boundary = f"----WebKitFormBoundary{os.urandom(8).hex()}"
                body = []

                # Metadata fields
                for k, v in [
                    ("action", "upload_chunk"),
                    ("type", target_type),
                    ("chunk_index", str(idx)),
                    ("total_chunks", str(total_chunks)),
                    ("step", str(step)),
                    ("loss", f"{loss:.6f}"),
                    ("source", node_name),
                ]:
                    body.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8"))

                # Chunk payload
                body.append(
                    f"--{boundary}\r\nContent-Disposition: form-data; name=\"chunk\"; filename=\"chunk_{idx}.bin\"\r\n"
                    f"Content-Type: application/octet-stream\r\n\r\n".encode("utf-8")
                )
                body.append(chunk_data)
                body.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))

                payload = b"".join(body)

                req = urllib.request.Request(
                    sync_url,
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {sync_token}",
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                        "Content-Length": str(len(payload)),
                        "User-Agent": "CerebroCloudSync/1.0"
                    }
                )

                success = False
                for attempt in range(1, 4):
                    try:
                        with urllib.request.urlopen(req, timeout=120) as resp:
                            resp_json = json.loads(resp.read().decode("utf-8"))
                            if resp_json.get("success"):
                                success = True
                                break
                            else:
                                print(f"\n⚠️ [cloud_sync] Chunk {idx + 1}/{total_chunks} rejected (attempt {attempt}): {resp_json}")
                    except Exception as e:
                        print(f"\n⚠️ [cloud_sync] Chunk {idx + 1}/{total_chunks} failed (attempt {attempt}): {e}")
                    time.sleep(attempt * 1.5)

                # Explicit memory release
                del payload, body, chunk_data
                if not success:
                    print(f"\n❌ [cloud_sync] Aborted upload at chunk {idx + 1}/{total_chunks} after 3 attempts.")
                    return False
                pct = ((idx + 1) / total_chunks) * 100
                print(f"\r  Uploaded chunk {idx + 1}/{total_chunks} ({pct:.0f}%)", end="", flush=True)

        print()
        dt = time.time() - t0
        sz_mb = file_size / 1e6
        print(f"✅ [cloud_sync] Successfully uploaded {target_type}.pt ({sz_mb:.1f} MB) in {dt:.1f}s to bioflo.in!")
        if target_type == "best":
            _LAST_BEST_UPLOAD_TIME = time.time()
        return True
    finally:
        with _UPLOAD_LOCK:
            if _IN_FLIGHT_UPLOADS.get(target_type) == step:
                _IN_FLIGHT_UPLOADS.pop(target_type, None)


def push_checkpoint_async(
    file_path: str,
    step: int,
    loss: float,
    target_type: str = "latest",
    source_node: Optional[str] = None,
    url: Optional[str] = None,
    token: Optional[str] = None,
    chunk_size_mb: int = 10
):
    """Asynchronous background upload. 0 ms interruption to training loops."""
    thread = threading.Thread(
        target=push_checkpoint,
        args=(file_path, step, loss, target_type, source_node, url, token, chunk_size_mb),
        daemon=True,
        name=f"CloudSync_{target_type}_{step}"
    )
    thread.start()
    return thread


def auto_sync_checkpoints(
    local_dir: str = "checkpoints",
    url: Optional[str] = None,
    token: Optional[str] = None
) -> Optional[str]:
    """
    Checks remote status against local files.
    If remote latest has higher step than local, automatically downloads it.
    Returns path to the best available checkpoint file for resuming.
    """
    os.makedirs(local_dir, exist_ok=True)
    manifest_info = check_remote_manifest(url, token)
    remote_manifest = manifest_info.get("manifest", {})
    if not isinstance(remote_manifest, dict):
        remote_manifest = {}
    remote_latest = remote_manifest.get("latest", {}) if isinstance(remote_manifest, dict) else {}
    remote_step = remote_latest.get("step", 0) if isinstance(remote_latest, dict) else 0

    # Inspect local checkpoint directory
    local_latest_step = -1
    local_latest_path = None

    import glob
    candidates = glob.glob(os.path.join(local_dir, "*_step_*.pt")) + glob.glob(os.path.join(local_dir, "*latest*.pt"))
    for cand in candidates:
        import re
        m = re.search(r"step_(\d+)", os.path.basename(cand))
        if m:
            s = int(m.group(1))
            if s > local_latest_step:
                local_latest_step = s
                local_latest_path = cand

    print(f"🔍 [cloud_sync] Checkpoint Assessment: Local Step = {local_latest_step:,} | Remote Step = {remote_step:,}")

    target_local_file = os.path.join(local_dir, "cerebro_spheroid_v4_latest.pt")
    if remote_step > local_latest_step and manifest_info.get("has_latest"):
        print(f"📥 [cloud_sync] Remote checkpoint is newer ({remote_step:,} > {local_latest_step:,}). Pulling from bioflo.in...")
        success = pull_checkpoint(target_local_file, target_type="latest", url=url, token=token)
        if success:
            return target_local_file

    return local_latest_path or (target_local_file if os.path.exists(target_local_file) else None)
