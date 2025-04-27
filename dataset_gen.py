import os
import csv
import random
from tqdm import tqdm
import logging
import shutil  # for copying files

# def setup_logging(log_file):
#     logging.basicConfig(
#         filename=log_file,
#         level=logging.INFO,
#         format='%(asctime)s - %(message)s',
#         filemode='w'
#     )
#     logging.info("Logging initialized.")

def sample_and_combine_folders(base_dir, 
                            #    target_dir, 
                            #    split_files, 
                            #    log_file, 
                            #    csv_output, 
                            #    sample_ratio=1/3
                               ):
    """
    Sample 1/3 from middle, left, right folders and combine into target_dir.
    Save log and CSV.
    """
    # if not os.path.exists(target_dir):
    #     os.makedirs(target_dir)

    # setup_logging(log_file)
    
    middle_dir = base_dir
    left_dir = base_dir.replace("middle", "left")
    right_dir = base_dir.replace("middle", "right")
    
    import pdb; pdb.set_trace()
    
    
    
    
    

    # for folder in tqdm(folders, desc="Sampling Folders"):
    #     folder_path = os.path.join(base_dir, folder)
    #     if not os.path.exists(folder_path):
    #         logging.warning(f"Folder not found: {folder_path}")
    #         continue
        
    #     files = [f for f in os.listdir(folder_path) if f.endswith(".jpeg")]
    #     sample_size = int(len(files) * sample_ratio)
    #     sampled_files = random.sample(files, sample_size)

    #     target_subdir = os.path.join(target_dir, folder)
    #     if not os.path.exists(target_subdir):
    #         os.makedirs(target_subdir)

    #     for file_name in tqdm(sampled_files, desc=f"Processing {folder}", leave=False):
    #         src_file = os.path.join(folder_path, file_name)
    #         dest_file = os.path.join(target_subdir, file_name)

    #         try:
    #             if not os.path.exists(dest_file):
    #                 shutil.copy(src_file, dest_file)  # or use os.symlink(src_file, dest_file)
    #             logging.info(f"Copied {src_file} -> {dest_file}")
    #             csv_data.append([folder, file_name, dest_file])
    #         except Exception as e:
    #             logging.error(f"Failed to copy {src_file} -> {dest_file}: {e}")

    # Save CSV with sampled info
    # with open(csv_output, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["OriginalFolder", "FileName", "NewLocation"])
    #     writer.writerows(csv_data)

    # print(f"Sampling and combining completed. Log: {log_file}, CSV: {csv_output}")

# Example usage:
if __name__ == "__main__":
    sample_and_combine_folders(
        base_dir="/home/zhyw86/WorkSpace/google-earth/data/aerial/middle/ID0001/"
        # base_dir="/home/zhyw86/WorkSpace/google-earth/split_folders",
        # target_dir="/home/zhyw86/WorkSpace/google-earth/combined_sampled_output",
        # split_file="sampled_files.txt",
        # log_file="sample_combine.log",
        # csv_output="sampled_files.csv",
        # sample_ratio=1/3  # Default 1/3 sampling
    )
