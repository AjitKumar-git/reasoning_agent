"""
Multi-threaded fast downloader for BioFlo checkpoints using HTTP Range requests.
"""

import os
import sys
import time
import zipfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_URL = "https://bioflo.in/api/cerebro/sync.php"
DEFAULT_TOKEN = "CerebroSpheroid_Sync_Secure_2026"

def download_chunk(url, token, start, end, filepath, chunk_idx):
    headers = {
        "Authorization": f"Bearer {token}",
        "Range": f"bytes={start}-{end}",
        "User-Agent": "CerebroFastSync/1.0"
    }
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
                if len(data) != (end - start + 1):
                    raise ValueError(f"Chunk {chunk_idx}: expected {end - start + 1} bytes, got {len(data)}")
                with open(filepath, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                return chunk_idx, len(data)
        except Exception as e:
            if attempt == 4:
                raise e
            time.sleep(1.0 + attempt)

def parallel_download(target_type: str, dest_path: str, num_threads: int = 12):
    url = f"{DEFAULT_URL}?action=download&type={target_type}"
    print(f"🔄 Probing size for {target_type}...")
    probe_req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {DEFAULT_TOKEN}", "User-Agent": "CerebroFastSync/1.0", "Range": "bytes=0-0"}
    )
    with urllib.request.urlopen(probe_req, timeout=15) as p_resp:
        cr = p_resp.headers.get("Content-Range", "")
        if "/" not in cr:
            raise RuntimeError(f"Server did not return Content-Range: {cr}")
        total_bytes = int(cr.split("/")[-1])

    print(f"📦 Total size for {target_type}: {total_bytes:,} bytes ({total_bytes / 1e6:.1f} MB)")
    part_path = dest_path + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)

    # Pre-allocate file
    with open(part_path, "wb") as f:
        f.truncate(total_bytes)

    # Split into chunks
    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    ranges = []
    start = 0
    while start < total_bytes:
        end = min(start + chunk_size - 1, total_bytes - 1)
        ranges.append((start, end))
        start = end + 1

    print(f"🚀 Launching {len(ranges)} chunks across {num_threads} parallel threads...")
    t0 = time.time()
    completed_bytes = 0

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(download_chunk, url, DEFAULT_TOKEN, r[0], r[1], part_path, idx): idx
            for idx, r in enumerate(ranges)
        }
        for fut in as_completed(futures):
            idx, length = fut.result()
            completed_bytes += length
            pct = (completed_bytes / total_bytes) * 100
            elapsed = max(time.time() - t0, 0.1)
            speed = (completed_bytes / 1e6) / elapsed
            print(f"\r  Progress: {completed_bytes / 1e6:.1f}/{total_bytes / 1e6:.1f} MB ({pct:.1f}%) | Speed: {speed:.2f} MB/s", end="", flush=True)

    print()
    if not zipfile.is_zipfile(part_path):
        raise ValueError(f"Corrupt checkpoint download for {target_type}")

    os.replace(part_path, dest_path)
    total_time = time.time() - t0
    print(f"✅ Successfully downloaded {dest_path} in {total_time:.1f}s ({total_bytes / 1e6 / total_time:.2f} MB/s)!")

if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["latest", "best", "both"], default="both")
    args = parser.parse_args()

    # Query manifest to get real step numbers
    manifest = {}
    try:
        req = urllib.request.Request(DEFAULT_URL, headers={"Authorization": f"Bearer {DEFAULT_TOKEN}", "User-Agent": "CerebroFastSync/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            manifest = data.get("manifest", {})
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch manifest: {e}")

    latest_step = manifest.get("latest", {}).get("step", "latest")
    best_step = manifest.get("best", {}).get("step", "best")

    if args.type in ("latest", "both"):
        dest = f"checkpoints/cerebro_spheroid_v4_step{latest_step}_latest.pt" if latest_step != "latest" else "checkpoints/cerebro_spheroid_v4_latest.pt"
        parallel_download("latest", dest, num_threads=12)
        if dest != "checkpoints/cerebro_spheroid_v4_latest.pt":
            import shutil
            shutil.copyfile(dest, "checkpoints/cerebro_spheroid_v4_latest.pt")
    if args.type in ("best", "both"):
        dest = f"checkpoints/cerebro_spheroid_v4_step{best_step}_best.pt" if best_step != "best" else "checkpoints/cerebro_spheroid_v4_best.pt"
        parallel_download("best", dest, num_threads=12)
        if dest != "checkpoints/cerebro_spheroid_v4_best.pt":
            import shutil
            shutil.copyfile(dest, "checkpoints/cerebro_spheroid_v4_best.pt")
