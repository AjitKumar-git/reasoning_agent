"""
Monitor and fetch logs/outputs from the active Kaggle run for CerebroSpheroid V4.
"""

import os
import subprocess
import sys


def main():
    kernel_slug = "ajitsheoran/reasoning-agent"
    print("=" * 60)
    print(f" Checking Kaggle Run: {kernel_slug}")
    print("=" * 60)

    # 1. Check status
    res = subprocess.run(["kaggle", "kernels", "status", kernel_slug], capture_output=True, text=True)
    status_line = res.stdout.strip()
    print(status_line)

    # 2. Try fetching output if available
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".kaggle_output"))
    os.makedirs(out_dir, exist_ok=True)
    res_out = subprocess.run(["kaggle", "kernels", "output", kernel_slug, "-p", out_dir], capture_output=True, text=True)
    
    files = os.listdir(out_dir)
    if files:
        print(f"\nFiles in {out_dir}:")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(out_dir, f))
            print(f"  • {f} ({size:,} bytes)")
    else:
        print("\n(No output files written to disk yet - training is executing in Kaggle container)")


if __name__ == "__main__":
    main()
