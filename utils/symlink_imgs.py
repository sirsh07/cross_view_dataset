import os
from pathlib import Path

colmap_root = Path("/home/smitra/cv_dataset/dataset_50sites/colmap/results")
mast3r_root = Path("/home/smitra/cv_dataset/dataset_50sites/mast3r_sfm/mast3r/results")

for colmap_output in colmap_root.rglob("output/undistorted_images_folder"):
    # Derive the relative path from colmap_root
    try:
        rel_path = colmap_output.relative_to(colmap_root)
    except ValueError:
        continue

    # Define the corresponding reconstruction folder in mast3r
    mast3r_reconstruction = mast3r_root / rel_path.parent / "reconstruction"
    mast3r_symlink = mast3r_reconstruction / "images"

    # Skip if target doesn't exist
    if not colmap_output.exists():
        continue

    # Ensure reconstruction folder exists
    mast3r_reconstruction.mkdir(parents=True, exist_ok=True)

    # Create symlink if not already present
    if not mast3r_symlink.exists():
        os.symlink(colmap_output, mast3r_symlink)
        print(f"Linked: {mast3r_symlink} -> {colmap_output}")
    else:
        print(f"Skipped (exists): {mast3r_symlink}")
