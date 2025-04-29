import os
from tqdm import tqdm

base_data = "/home/sirsh/cv_dataset/dataset_50sites/data"
base_results = "/home/sirsh/cv_dataset/dataset_50sites/mast3r/results"

# Walk through all 'reconstruction' folders
for root, dirs, files in os.walk(base_results):
    if os.path.basename(root) == "reconstruction":
        # Build the relative path to 'data' folder
        relative_path = os.path.relpath(root, base_results)
        # This gives 'aerial/train/IDXXXX/pXXX/output/reconstruction'
        relative_parts = relative_path.split(os.sep)

        if len(relative_parts) < 5:
            print(f"Skipping {root}, path structure unexpected.")
            continue

        # We want 'aerial/train/IDXXXX/pXXX'
        data_relative_path = os.path.join(*relative_parts[:-2])  # Drop 'output/reconstruction'

        src_images = os.path.join(base_data, data_relative_path, "images")
        dest_link = os.path.join(root, "images")

        if os.path.exists(src_images):
            # Remove old symlink if it exists
            if os.path.islink(dest_link) or os.path.exists(dest_link):
                os.remove(dest_link)
            # Create new symlink
            os.symlink(src_images, dest_link)
            print(f"Linked {src_images} -> {dest_link}")
        else:
            print(f"Warning: Source {src_images} does not exist, skipping.")
