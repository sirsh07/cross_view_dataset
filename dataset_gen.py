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
                               split_folder,
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
    
    
    # get right and left data folders
    folder_name =  os.path.basename(base_dir.rstrip('/'))
    parent_dir = os.path.dirname(base_dir.rstrip('/'))
    
    left_dir = os.path.join(parent_dir.replace("middle", "left"), f'{folder_name}_left')
    right_dir = os.path.join(parent_dir.replace("middle", "right"), f'{folder_name}_right')
    
    
    # get right and left split files
    # split_folder="/home/zhyw86/WorkSpace/google-earth/sampling/aerial/middle/random/ID0001/",
    folder_name = os.path.basename(split_folder.rstrip('/'))
    parent_dir = os.path.dirname(split_folder.rstrip('/'))
    
    parent_dir = parent_dir.replace("random","")
    
    split_file = os.path.join(split_folder, f'{folder_name}_train.txt')
    left_split_file = os.path.join(parent_dir.replace("middle", "left"), f"{folder_name}" ,f'{folder_name}_left_train.txt')
    right_split_file = os.path.join(parent_dir.replace("middle", "right"), f"{folder_name}" ,f'{folder_name}_right_train.txt')
    
    # Read split files
    with open(split_file, "r") as f:
        middle_files = [line.strip() for line in f.readlines()]
        # sampled_middle_files = [middle_files[i] for i in range(0, len(middle_files), 3)]
        sampled_middle_files = random.sample(middle_files, int(len(middle_files) * 1/3))
        sampled_middle_file_path = [os.path.join(base_dir,"footage", file) for file in sampled_middle_files]
        
    with open(left_split_file, "r") as f:
        left_files = [line.strip() for line in f.readlines()]
        # sampled_left_files = [left_files[i] for i in range(1, len(left_files), 3)]
        sampled_middle_files = random.sample(left_files, int(len(left_files) * 1/3))
        sampled_left_file_path = [os.path.join(left_dir,"footage", file) for file in sampled_left_files]
        
    with open(right_split_file, "r") as f:
        right_files = [line.strip() for line in f.readlines()]
        # sampled_right_files = [right_files[i] for i in range(2, len(right_files), 3)]
        sampled_right_file_path = random.sample(right_files, int(len(right_files) * 1/3))
        sampled_right_file_path = [os.path.join(right_dir,"footage", file) for file in sampled_right_files]
        
    # Combine all files
    all_sampled_file_path = sampled_middle_file_path + sampled_left_file_path + sampled_right_file_path
    import pdb; pdb.set_trace()
    # permute all sampled files
    random.shuffle(all_sampled_file_path)
    # split_100p = random.permute all_sampled_file_path
    # split_50p = [split_100p[i] for i in range(0, len(split_100p), 2)]
    # split_25p = [split_100p[i] for i in range(0, len(split_100p), 4)]
    # split_12p = [split_100p[i] for i in range(0, len(split_100p), 8)]
    
    
    
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
        base_dir="/home/zhyw86/WorkSpace/google-earth/data/aerial/middle/ID0001/",
        split_folder="/home/zhyw86/WorkSpace/google-earth/sampling/aerial/middle/random/ID0001/",
        # ID0001_train.txt",
        # base_dir="/home/zhyw86/WorkSpace/google-earth/split_folders",
        # target_dir="/home/zhyw86/WorkSpace/google-earth/combined_sampled_output",
        # split_file="sampled_files.txt",
        # log_file="sample_combine.log",
        # csv_output="sampled_files.csv",
        # sample_ratio=1/3  # Default 1/3 sampling
    )
