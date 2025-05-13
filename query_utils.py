import os
import os
import csv
import random
from tqdm import tqdm
import logging
import shutil  # for copying files
import pandas as pd




def sample_and_combine_folders(base_dir, 
                               split_folder,
                               target_dir,
                               site_id
                               ):
    """
    Sample 1/3 from middle, left, right folders and combine into target_dir.
    Save log and CSV.
    """
    
    
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
        sampled_left_files = random.sample(left_files, int(len(left_files) * 1/3))
        sampled_left_file_path = [os.path.join(left_dir,"footage", file) for file in sampled_left_files]
        
    with open(right_split_file, "r") as f:
        right_files = [line.strip() for line in f.readlines()]
        # sampled_right_files = [right_files[i] for i in range(2, len(right_files), 3)]
        sampled_right_files = random.sample(right_files, int(len(right_files) * 1/3))
        sampled_right_file_path = [os.path.join(right_dir,"footage", file) for file in sampled_right_files]
        
    # Combine all files
    all_sampled_file_path = sampled_middle_file_path + sampled_left_file_path + sampled_right_file_path
    random.shuffle(all_sampled_file_path)
    split_50p = [all_sampled_file_path[i] for i in range(0, len(all_sampled_file_path), 2)]
    split_25p = [all_sampled_file_path[i] for i in range(0, len(all_sampled_file_path), 4)]
    split_12p = [all_sampled_file_path[i] for i in range(0, len(all_sampled_file_path), 8)]
    
    
    os.makedirs(target_dir, exist_ok=True)
    
    # create 100p folder
    target_100p = os.path.join(target_dir, "p100","images")
    target_50p = os.path.join(target_dir, "p50","images")
    target_25p = os.path.join(target_dir, "p25","images")
    target_12p = os.path.join(target_dir, "p12","images")
    os.makedirs(target_100p, exist_ok=True)
    os.makedirs(target_50p, exist_ok=True)
    os.makedirs(target_25p, exist_ok=True)
    os.makedirs(target_12p, exist_ok=True)
    
    original_file_paths = []
    new_file_paths = []
    file_names = []
    splits = []
    
    for file_path in tqdm(all_sampled_file_path, desc="Symlinking files to 100p folder"):
        file_name = os.path.basename(file_path)
        target_file_path = os.path.join(target_100p, file_name)
        if not os.path.exists(target_file_path):
            os.symlink(file_path, target_file_path)
        original_file_paths.append(file_path)
        new_file_paths.append(target_file_path)
        file_names.append(file_name)
        splits.append("100")
    
    for file_path in tqdm(split_50p, desc="Symlinking files to 50p folder"):
        file_name = os.path.basename(file_path)
        target_file_path = os.path.join(target_50p, file_name)
        if not os.path.exists(target_file_path):
            os.symlink(file_path, target_file_path)
        original_file_paths.append(file_path)
        new_file_paths.append(target_file_path)
        file_names.append(file_name)
        splits.append("50")
    
    for file_path in tqdm(split_25p, desc="Symlinking files to 25p folder"):
        file_name = os.path.basename(file_path)
        target_file_path = os.path.join(target_25p, file_name)
        if not os.path.exists(target_file_path):
            os.symlink(file_path, target_file_path)
        original_file_paths.append(file_path)
        new_file_paths.append(target_file_path)
        file_names.append(file_name)
        splits.append("25")
    
    for file_path in tqdm(split_12p, desc="Symlinking files to 12p folder"):
        file_name = os.path.basename(file_path)
        target_file_path = os.path.join(target_12p, file_name)
        if not os.path.exists(target_file_path):
            os.symlink(file_path, target_file_path) 
        original_file_paths.append(file_path)
        new_file_paths.append(target_file_path)
        file_names.append(file_name)
        splits.append("12")
        
    csv_data = pd.DataFrame({
        "OriginalFilePath": original_file_paths,
        "FileName": file_names,
        "NewFilePath": new_file_paths,
        "Split": splits
    })
    
    csv_data.to_csv(os.path.join(target_dir, f"sampled_files_{site_id}.csv"), index=False)    



def get_test_splits(root_dir: str = "/home/zhyw86/WorkSpace/google-earth/sampling"):
    
    street_split = "/home/zhyw86/WorkSpace/google-earth/sampling/street/random/ID0001_street/ID0001_street_test.txt"
    right_split = "/home/zhyw86/WorkSpace/google-earth/sampling/aerial/right/ID0001/ID0001_right_test.txt"
    middle_split = "/home/zhyw86/WorkSpace/google-earth/sampling/aerial/middle/random/ID0001/ID0001_test.txt"
    left_split = "/home/zhyw86/WorkSpace/google-earth/sampling/aerial/left/ID0001/ID0001_left_test.txt"
    
    street_folder = "/home/zhyw86/WorkSpace/google-earth/data/street/ID0001_street/footage"

    
    for ids in os.listdir("/home/sirsh/cv_dataset/dataset_50sites/colmap/metadata/aerial_street/train"):
        
        if not os.path.exists(street_split.replace("ID0001", ids)):
            print(f"street_split: {street_split.replace('ID0001', ids)} not exists")
        else:
            street_split_file = street_split.replace("ID0001", ids)
            with open(street_split_file, "r") as f:
                street_files = [line.strip() for line in f.readlines()]
                test_street_files = random.sample(street_files, 15)
                test_street_files_path = [os.path.join(street_folder.replace("ID0001", ids), file) for file in test_street_files]
                import pdb; pdb.set_trace()
        # if not os.path.exists(street_split.replace("ID0001", ids)):    
            
        if not os.path.exists(right_split.replace("ID0001", ids)):
            print(f"right_split: {right_split.replace('ID0001', ids)} not exists")
        else:
            right_split_file = right_split.replace("ID0001", ids)
            
        if not os.path.exists(middle_split.replace("ID0001", ids)):
            print(f"middle_split: {middle_split.replace('ID0001', ids)} not exists")
        else:
            middle_split_file = middle_split.replace("ID0001", ids)
        
        if not os.path.exists(left_split.replace("ID0001", ids)):
            print(f"left_split: {left_split.replace('ID0001', ids)} not exists")
        else:
            left_split_file = left_split.replace("ID0001", ids)
        
        # with open(split_file, "r") as f:
        # middle_files = [line.strip() for line in f.readlines()]
        # # sampled_middle_files = [middle_files[i] for i in range(0, len(middle_files), 3)]
        # sampled_middle_files = random.sample(middle_files, int(len(middle_files) * 1/3))
        # sampled_middle_file_path = [os.path.join(base_dir,"footage", file) for file in sampled_middle_files]
        

if __name__ == "__main__":
    get_test_splits()

