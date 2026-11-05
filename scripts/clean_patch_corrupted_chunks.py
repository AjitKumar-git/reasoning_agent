#!/usr/bin/env python3
"""
Clean-patch the 4 chunks (9, 12, 13, 14) that had Apache 503 error text injected.
Verify MD5 matches the server's manifest hash (1fbb74ac88a4a3a3d99ca3e0b4e34591).
"""
import os
import sys
import time
import urllib.request
import subprocess
import hashlib

URL = "https://bioflo.in/api/cerebro/sync.php?action=download&type=latest"
TOKEN = "CerebroSpheroid_Sync_Secure_2026"
TARGET_FILE = "checkpoints/cerebro_spheroid_v4_step_33700.pt"
EXPECTED_MD5 = "1fbb74ac88a4a3a3d99ca3e0b4e34591"
CHUNK_SIZE = 10 * 1024 * 1024
TOTAL_BYTES = 237085514

# Chunks that contained the 503 error text
CORRUPTED_CHUNKS = [9, 12, 13, 14]

def download_clean_chunk(chunk_idx):
    start = chunk_idx * CHUNK_SIZE
    end = min(TOTAL_BYTES - 1, (chunk_idx + 1) * CHUNK_SIZE - 1)
    exp_len = end - start + 1
    tmp_file = f"/tmp/clean_chunk_{chunk_idx}.bin"

    print(f"Downloading Chunk {chunk_idx} ({start:,} - {end:,}, {exp_len:,} bytes)...")

    for attempt in range(1, 20):
        if os.path.exists(tmp_file):
            try: os.remove(tmp_file)
            except: pass
        
        cmd = [
            "curl", "-s", "-S", "--http1.1",
            "-r", f"{start}-{end}",
            "--retry", "5",
            "--retry-delay", "2",
            "-H", f"Authorization: Bearer {TOKEN}",
            URL,
            "-o", tmp_file
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(tmp_file) and os.path.getsize(tmp_file) == exp_len:
            with open(tmp_file, "rb") as f:
                data = f.read()
            if b"503 Service Unavailable" in data or b"<html" in data:
                print(f"  Attempt {attempt}: Received 503 HTML error, retrying...")
                time.sleep(3)
                continue
            print(f"✅ Chunk {chunk_idx} cleanly downloaded ({len(data):,} bytes)")
            os.remove(tmp_file)
            return start, data
        else:
            print(f"  Attempt {attempt} failed: {res.stderr.strip()[:80]}")
            time.sleep(2)

    raise RuntimeError(f"Failed to cleanly download chunk {chunk_idx}")

def main():
    if not os.path.exists(TARGET_FILE):
        print(f"Error: {TARGET_FILE} not found.")
        sys.exit(1)

    print("=" * 70)
    print(" CEREBRO-SPHEROID V4 — REPAIRING 503-CORRUPTED CHUNKS")
    print("=" * 70)

    for chunk_idx in CORRUPTED_CHUNKS:
        start_offset, clean_data = download_clean_chunk(chunk_idx)
        print(f"Patching {len(clean_data):,} bytes into {TARGET_FILE} at offset {start_offset:,}...")
        with open(TARGET_FILE, "r+b") as f:
            f.seek(start_offset)
            f.write(clean_data)

    print("\nVerifying that no 503 text remains anywhere in the checkpoint...")
    with open(TARGET_FILE, "rb") as f:
        full_data = f.read()

    pos = full_data.find(b"503 Service")
    if pos != -1:
        print(f"❌ Error: 503 text still found at offset {pos}!")
        sys.exit(1)
    print("✅ Zero 503 error strings found in file!")

    calc_md5 = hashlib.md5(full_data).hexdigest()
    print(f"\nCalculated MD5: {calc_md5}")
    print(f"Expected MD5:   {EXPECTED_MD5}")

    if calc_md5 == EXPECTED_MD5:
        print("🎉 PERFECT MATCH! Checkpoint is 100% bit-identical to remote Colab state!")
    else:
        print("⚠️ MD5 mismatch with server manifest (checking other chunks...)")

if __name__ == "__main__":
    main()
