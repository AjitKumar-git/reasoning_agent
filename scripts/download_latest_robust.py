#!/usr/bin/env python3
"""
Downloads and verifies the latest remote checkpoint (Step 48,300) from bioflo.in.
Ensures zero-corruption reassembly and strict MD5 verification.
"""
import os
import sys
import time
import shutil
import hashlib
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://bioflo.in/api/cerebro/sync.php?action=download&type=latest"
TOKEN = "CerebroSpheroid_Sync_Secure_2026"
EXPECTED_MD5 = "62065bc4e609414a7b6e8304c6fa0a01"
TOTAL_BYTES = 237085514
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB per chunk
OUT_FILE = "checkpoints/cerebro_spheroid_v4_step_48300.pt"
TEMP_DIR = "/tmp/cerebro_chunks_48300"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Build chunk list
chunks = []
curr = 0
chunk_idx = 0
while curr < TOTAL_BYTES:
    end = min(curr + CHUNK_SIZE - 1, TOTAL_BYTES - 1)
    exp_len = end - curr + 1
    chunk_file = os.path.join(TEMP_DIR, f"chunk_{chunk_idx:03d}.bin")
    chunks.append({
        "idx": chunk_idx,
        "start": curr,
        "end": end,
        "length": exp_len,
        "file": chunk_file
    })
    curr = end + 1
    chunk_idx += 1

print(f"Target Checkpoint: Step 48,300 ({TOTAL_BYTES:,} bytes, {len(chunks)} chunks)")
print(f"Expected MD5: {EXPECTED_MD5}")


def download_chunk(chunk):
    idx = chunk["idx"]
    start = chunk["start"]
    end = chunk["end"]
    exp_len = chunk["length"]
    part_file = chunk["file"]

    if os.path.exists(part_file) and os.path.getsize(part_file) == exp_len:
        with open(part_file, "rb") as f:
            head = f.read(200)
            f.seek(max(0, exp_len - 200))
            tail = f.read(200)
            if b"503 Service" not in head and b"503 Service" not in tail:
                return idx, True, 0.0

    for attempt in range(1, 20):
        temp_slice = f"{part_file}.slice"
        if os.path.exists(temp_slice):
            try: os.remove(temp_slice)
            except: pass
        if os.path.exists(part_file):
            try: os.remove(part_file)
            except: pass

        t0 = time.time()
        cmd = [
            "curl", "-s", "-S", "--http1.1",
            "-r", f"{start}-{end}",
            "--retry", "5",
            "--retry-delay", "2",
            "-H", f"Authorization: Bearer {TOKEN}",
            URL,
            "-o", temp_slice
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        dur = time.time() - t0

        if os.path.exists(temp_slice) and os.path.getsize(temp_slice) == exp_len:
            with open(temp_slice, "rb") as f:
                head = f.read(200)
                f.seek(max(0, exp_len - 200))
                tail = f.read(200)
                if b"503 Service" in head or b"503 Service" in tail or b"<html" in head:
                    print(f"  [Chunk {idx:02d}] Received HTTP error payload, retrying (attempt {attempt})...")
                    time.sleep(2)
                    continue
            shutil.move(temp_slice, part_file)
            return idx, True, dur
        time.sleep(1.5)
    return idx, False, 0.0


print("Downloading chunks concurrently (2 workers)...")
t_start = time.time()
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(download_chunk, c) for c in chunks]
    for fut in as_completed(futures):
        idx, ok, dur = fut.result()
        if not ok:
            print(f"❌ Failed to download chunk {idx:02d}")
            sys.exit(1)
        print(f"  ✓ Chunk {idx:02d}/{len(chunks)-1:02d} verified ({chunks[idx]['length']:,} bytes in {dur:.2f}s)")

print("Reassembling checkpoint...")
with open(OUT_FILE, "wb") as f_out:
    for c in chunks:
        with open(c["file"], "rb") as f_in:
            shutil.copyfileobj(f_in, f_out)

print(f"Saved: {OUT_FILE} ({os.path.getsize(OUT_FILE):,} bytes)")

print("Calculating MD5...")
md5 = hashlib.md5()
with open(OUT_FILE, "rb") as f:
    for block in iter(lambda: f.read(65536), b""):
        md5.update(block)
calc_md5 = md5.hexdigest()

print(f"Calculated MD5: {calc_md5}")
print(f"Expected MD5:   {EXPECTED_MD5}")
if calc_md5 == EXPECTED_MD5:
    print("✅ EXACT BITWISE MATCH! Checkpoint is 100% clean and verified.")
    shutil.copyfile(OUT_FILE, "checkpoints/cerebro_spheroid_v4_latest.pt")
else:
    print("❌ MD5 MISMATCH! File may be corrupted.")
    sys.exit(1)
