#!/usr/bin/env python3
"""
Robust, verified multi-chunk downloader for CerebroSpheroid checkpoints from bioflo.in.
Uses HTTP Range requests with 10MB chunks and 2 concurrent workers to respect
Apache resource constraints and avoid 503 throttling.
"""
import os
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://bioflo.in/api/cerebro/sync.php?action=download&type=latest"
TOKEN = "CerebroSpheroid_Sync_Secure_2026"
TOTAL_BYTES = 237085514
CHUNK_SIZE = 10 * 1024 * 1024  # 10 MB per chunk
OUT_FILE = "checkpoints/cerebro_spheroid_v4_step_33700.pt"
TEMP_DIR = "/tmp/cerebro_chunks"

os.makedirs("checkpoints", exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Build chunk ranges
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

print(f"Total size: {TOTAL_BYTES:,} bytes ({TOTAL_BYTES/1e6:.1f} MB)")
print(f"Number of chunks: {len(chunks)} (~10 MB each)")

def download_single_chunk(chunk):
    idx = chunk["idx"]
    start = chunk["start"]
    end = chunk["end"]
    exp_len = chunk["length"]
    part_file = chunk["file"]

    # Check if already downloaded and valid
    if os.path.exists(part_file) and os.path.getsize(part_file) == exp_len:
        # Check no 503 error text
        with open(part_file, "rb") as f:
            f.seek(max(0, exp_len - 100))
            if b"503 Service Unavailable" not in f.read():
                return idx, True, 0.0

    # Retry loop with byte-level chunk resumption
    for attempt in range(1, 25):
        cur_sz = os.path.getsize(part_file) if os.path.exists(part_file) else 0
        if cur_sz >= exp_len:
            with open(part_file, "rb") as f:
                f.seek(max(0, exp_len - 100))
                if b"503 Service Unavailable" not in f.read():
                    return idx, True, 0.0
            # If invalid text at end, remove and restart
            try: os.remove(part_file)
            except: pass
            cur_sz = 0

        req_start = start + cur_sz
        req_end = end
        
        t0 = time.time()
        # Download remainder to temporary part
        temp_slice = f"{part_file}.slice"
        if os.path.exists(temp_slice):
            try: os.remove(temp_slice)
            except: pass

        cmd = [
            "curl", "-s", "-S", "--http1.1",
            "-r", f"{req_start}-{req_end}",
            "--retry", "5",
            "--retry-delay", "2",
            "-H", f"Authorization: Bearer {TOKEN}",
            URL,
            "-o", temp_slice
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        dur = time.time() - t0

        if os.path.exists(temp_slice) and os.path.getsize(temp_slice) > 0:
            slice_sz = os.path.getsize(temp_slice)
            with open(part_file, "ab") as f_out, open(temp_slice, "rb") as f_in:
                f_out.write(f_in.read())
            try: os.remove(temp_slice)
            except: pass

            actual_sz = os.path.getsize(part_file)
            if actual_sz == exp_len:
                # Validate not a 503 text
                with open(part_file, "rb") as f:
                    f.seek(max(0, exp_len - 100))
                    if b"503 Service Unavailable" not in f.read():
                        return idx, True, dur
                print(f"  [Chunk {idx:02d}] Completed with 503 error, resetting...")
                try: os.remove(part_file)
                except: pass
            else:
                print(f"  [Chunk {idx:02d}] Resumed progress: {actual_sz:,}/{exp_len:,} bytes ({actual_sz/exp_len*100:.1f}%)")
        else:
            print(f"  [Chunk {idx:02d}] Attempt {attempt} failed: {res.stderr.strip()[:80]}")
        
        time.sleep(2)

    return idx, False, 0.0

def main():
    t_start = time.time()
    completed_bytes = 0
    
    # Check already valid chunks
    valid_count = 0
    for c in chunks:
        if os.path.exists(c["file"]) and os.path.getsize(c["file"]) == c["length"]:
            valid_count += 1
            completed_bytes += c["length"]
    
    print(f"Resuming with {valid_count}/{len(chunks)} chunks already verified ({completed_bytes/1e6:.1f} MB)")

    # 2 concurrent workers to avoid overloading Apache PHP pool
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(download_single_chunk, c): c for c in chunks}
        for future in as_completed(futures):
            c = futures[future]
            idx, success, dur = future.result()
            if not success:
                print(f"❌ Permanent failure on chunk {idx}!")
                sys.exit(1)
            completed_bytes += c["length"]
            pct = (completed_bytes / TOTAL_BYTES) * 100
            print(f"  ✅ [{completed_bytes/1e6:5.1f}/{TOTAL_BYTES/1e6:5.1f} MB, {pct:5.1f}%] Chunk {idx:02d}/{len(chunks)-1:02d} verified in {dur:.1f}s")

    print(f"\nAll {len(chunks)} chunks verified! Assembling into {OUT_FILE}...")
    with open(OUT_FILE, "wb") as out_f:
        for c in chunks:
            with open(c["file"], "rb") as pf:
                out_f.write(pf.read())
            os.remove(c["file"])
    try:
        os.rmdir(TEMP_DIR)
    except:
        pass

    final_size = os.path.getsize(OUT_FILE)
    print(f"✅ Assembly complete! Final file size: {final_size:,} bytes in {time.time()-t_start:.1f}s")
    assert final_size == TOTAL_BYTES, f"Size mismatch: {final_size} != {TOTAL_BYTES}"

if __name__ == "__main__":
    main()
