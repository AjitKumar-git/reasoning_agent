import os
import time
import subprocess
import sys

target = "checkpoints/cerebro_spheroid_v4_step_84800_latest.pt"
print(f"Waiting for {target} to complete downloading and verifying...")

t0 = time.time()
while not os.path.exists(target):
    part_file = target + ".part"
    if os.path.exists(part_file):
        sz = os.path.getsize(part_file) / (1024 * 1024)
        print(f"\rDownload in progress: {sz:.1f} / 226.1 MB ({sz/226.1*100:.1f}%)", end="", flush=True)
    time.sleep(5)
    if time.time() - t0 > 900:  # 15 min safety timeout
        print("\nTimeout waiting for download.")
        sys.exit(1)

print(f"\n✅ File {target} is ready! Size: {os.path.getsize(target)/(1024*1024):.1f} MB")
print("Launching comprehensive evaluation...")
subprocess.run(["python3", "scripts/test_latest_checkpoint.py", target], check=True)
