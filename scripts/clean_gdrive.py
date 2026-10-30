"""
clean_gdrive.py - Clean Google Drive /content/drive/MyDrive/cerebro_checkpoints
Keeps ONLY the best and latest Option B (v4 small) checkpoints and deletes all others.
Can be executed directly in Colab after drive.mount('/content/drive').
"""

import os
import glob
import shutil

DRIVE_PATHS = [
    "/content/drive/MyDrive/cerebro_checkpoints",
    "/content/drive/MyDrive/checkpoints",
    os.path.expanduser("~/GoogleDrive/cerebro_checkpoints"),
    os.path.expanduser("~/drive/MyDrive/cerebro_checkpoints"),
]


def clean_drive():
    target_dir = None
    for p in DRIVE_PATHS:
        if os.path.exists(p):
            target_dir = p
            break

    if not target_dir:
        print("Google Drive directory not found locally.")
        print("To clean Google Drive, run this script inside Google Colab:")
        print("   from google.colab import drive; drive.mount('/content/drive')")
        print("   !python3 scripts/clean_gdrive.py")
        return

    print(f"Cleaning Google Drive directory: {target_dir}")
    all_files = glob.glob(os.path.join(target_dir, "*"))

    keep_names = {
        "cerebro_spheroid_v4_small_kaggle_best.pt",
        "cerebro_spheroid_v4_small_kaggle_best.meta.json",
        "cerebro_spheroid_v4_small_kaggle_latest.pt",
        "cerebro_spheroid_v4_small_kaggle_latest.meta.json",
        "cerebro_spheroid_v4_small_kaggle_state.pt",
        "cerebro_spheroid_v4_small_kaggle_state.json",
        "cerebro_spheroid_v4_jax_latest.pt",
        "cerebro_spheroid_v4_jax_best.pt",
        "latest.pt",
        "best.pt",
        "manifest.json",
    }

    deleted = 0
    kept = 0
    freed_bytes = 0

    for f in all_files:
        name = os.path.basename(f)
        size = os.path.getsize(f) if os.path.isfile(f) else 0

        if name in keep_names or "small_kaggle" in name:
            print(f"  [KEEP]   {name} ({size / (1024**2):.1f} MB)")
            kept += 1
        else:
            print(f"  [DELETE] {name} ({size / (1024**2):.1f} MB)")
            try:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                else:
                    os.remove(f)
                deleted += 1
                freed_bytes += size
            except Exception as e:
                print(f"    Failed to delete {name}: {e}")

    print(f"\nGoogle Drive Cleanup Complete!")
    print(f"  • Retained: {kept} critical files (best & latest only)")
    print(f"  • Removed : {deleted} obsolete/failed checkpoints")
    print(f"  • Freed   : {freed_bytes / (1024**2):.1f} MB storage")

    # Ensure best & latest Option B are mirrored in Google Drive
    target_latest = os.path.join(target_dir, "cerebro_spheroid_v4_small_kaggle_best.pt")
    if not os.path.exists(target_latest):
        local_src = os.path.join("checkpoints", "cerebro_spheroid_v4_small_kaggle_best.pt")
        if os.path.exists(local_src):
            print(f"\n[sync] Copying local Option B best into Drive: {target_latest}...")
            shutil.copy2(local_src, target_latest)
        else:
            print("\n[sync] Pulling Option B champion from bioflo.in into Drive...")
            try:
                from src.cloud_sync import pull_checkpoint
                pull_checkpoint(target_latest, target_type="latest")
            except Exception as e:
                print(f"  [warn] Could not pull from bioflo.in: {e}")


if __name__ == "__main__":
    clean_drive()
