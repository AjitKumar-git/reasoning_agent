#!/usr/bin/env python3
"""
Downloads pre-trained CerebroSpheroid checkpoints from Google Drive / Cloud storage.
Usage:
  python3 scripts/download_checkpoints.py
  python3 scripts/download_checkpoints.py --target v3_colab_best
"""

import os
import sys
import argparse
import urllib.request

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CKPT_DIR = os.path.join(ROOT_DIR, "checkpoints")

CHECKPOINTS = {
    "v3_colab_best": {
        "filename": "cerebro_spheroid_v3_colab_best.pt",
        "file_id": "1OKClaRv-g7qJRl8eBHVQCv1d8F4r--l-",
        "size_bytes": 324569441,
        "description": "CerebroSpheroid V3 checkpoint trained for 7,100 steps on 733k records",
    }
}


def download_drive_file(file_id: str, dest_path: str, expected_size: int = 0) -> str:
    """Downloads a public Google Drive file with progress indication."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and expected_size and os.path.getsize(dest_path) == expected_size:
        print(f"[download] {os.path.basename(dest_path)} is already present ({os.path.getsize(dest_path) / (1024**2):.1f} MB).")
        return dest_path

    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    print(f"[download] Fetching {os.path.basename(dest_path)} from Google Drive ({file_id})...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp_path = dest_path + ".tmp"
    with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out:
        total = int(resp.headers.get("Content-Length", expected_size or 0))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total / (1024 * 1024)
                print(f"\r  Progress: {mb_down:.1f}/{mb_tot:.1f} MB ({pct:.1f}%)", end="", flush=True)
    print()
    os.replace(tmp_path, dest_path)
    print(f"[download] Successfully saved {dest_path} ({os.path.getsize(dest_path) / (1024**2):.1f} MB).")
    return dest_path


def main():
    parser = argparse.ArgumentParser(description="Download CerebroSpheroid Checkpoints")
    parser.add_argument("--target", default="all", choices=["all"] + list(CHECKPOINTS.keys()),
                        help="Checkpoint target to download")
    args = parser.parse_args()

    targets = list(CHECKPOINTS.keys()) if args.target == "all" else [args.target]
    for key in targets:
        meta = CHECKPOINTS[key]
        dest = os.path.join(CKPT_DIR, meta["filename"])
        print(f"Target: {key} ({meta['description']})")
        download_drive_file(meta["file_id"], dest, meta["size_bytes"])


if __name__ == "__main__":
    main()
