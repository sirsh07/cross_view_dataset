import os
import json
from typing import List, Tuple, Dict, Any
import pandas as pd
from tqdm import tqdm
import numpy as np
from ges_utils import json_to_empty_colmap_model


def rot_ecef2enu(lat, lon):
    lamb = np.deg2rad(lon)
    phi = np.deg2rad(lat)
    sL = np.sin(lamb)
    sP = np.sin(phi)
    cL = np.cos(lamb)
    cP = np.cos(phi)
    rot = np.array([
        [     -sL,       cL,  0],
        [-sP * cL, -sP * sL, cP],
        [ cP * cL,  cP * sL, sP],
    ])
    return rot

# ----------------------------------------------------------------------
# 1.  Gather all *.json paths under the google-earth directory
# ----------------------------------------------------------------------
def collect_site_json_paths(root_dir: str = "/home/zhyw86/WorkSpace/google-earth/data/") -> List[str]:
    """
    Recursively walk `root_dir` and return every absolute path that ends in '.json'.
    Adjust the `if` block inside the loop if you need stricter pattern filtering.
    """
    json_paths: List[str] = []

    for cur_dir, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".json"):               # <-- tweak this if needed
                json_paths.append(os.path.join(cur_dir, fname))

    return json_paths


# ----------------------------------------------------------------------
# 2.  Read / parse the collected files
# ----------------------------------------------------------------------
def load_json_files(paths: List[str]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Given a list of file paths, open and parse each JSON.
    Returns a list of (path, parsed_dict) tuples.
    """
    records: List[Tuple[str, Dict[str, Any]]] = []

    for fpath in paths:
        try:
            with open(fpath, "r") as f:
                records.append((fpath, json.load(f)))
        except (OSError, json.JSONDecodeError) as err:
            print(f"[WARN] skipped {fpath}: {err}")

    return records

def load_dataset_files(root_dir: str = "/home/sirsh/cv_dataset/dataset_50sites/data") -> List[Tuple[str, Dict[str, Any]]]:
    """
    Recursively walk `root_dir` and return every absolute path that ends in '.csv'.
    Adjust the `if` block inside the loop if you need stricter pattern filtering.
    """
    csv_paths: List[str] = []

    for cur_dir, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".csv"):               # <-- tweak this if needed
                csv_paths.append(os.path.join(cur_dir, fname))

    return csv_paths
    

def get_data(csv_file: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Given a CSV file and a list of JSON files, load the CSV into a DataFrame
    and return it along with the parsed JSON data.
    """
    
    df = pd.read_csv(csv_file)
    
    def get_metadata_file_path(original_path: str) -> str:
        # Extract the directory and file prefix
        dir_path = os.path.dirname(original_path)  # Get the directory path
        parent_dir = os.path.dirname(dir_path)  # Get the parent directory path
        file_prefix = os.path.basename(parent_dir)
        # Construct the metadata file path
        return os.path.join(parent_dir, f"{file_prefix}.json")
    
    # Apply the function to the 'OriginalFilePath' column
    df["MetaDataPath"] = df["OriginalFilePath"].apply(get_metadata_file_path)
    
    sanity_check = df["MetaDataPath"].apply(os.path.exists)
    
    sanity_check = df["MetaDataPath"].apply(os.path.exists)
    if not sanity_check.all():
        missing_files = df.loc[~sanity_check, "MetaDataPath"]
        raise AssertionError(f"The following metadata files are missing:\n{missing_files.tolist()}")
    
    print("All metadata files exist.")
    
    unique_metadata = df["MetaDataPath"].unique()
    
    save_dir = os.path.dirname(csv_file)
    
    for metadata_path in unique_metadata:
        
        f_name = os.path.basename(metadata_path).split(".")[0]
        f_dir = os.path.join(save_dir, "ge_metadata",f_name)  
        os.makedirs(f_dir, exist_ok=True)
        json_to_empty_colmap_model(unique_metadata[0], f_dir, max_num_images=400)
    
    

# ----------------------------------------------------------------------
# example usage
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # step 1 – collect paths (one-time, cheap)
    # all_json_paths = collect_site_json_paths()
    # print(f"Found {len(all_json_paths)} JSON files.")
    
    if os.path.exists("./cache_files/all_csv_paths.txt"):
        with open("./cache_files/all_csv_paths.txt", "r") as f:
            all_csv_paths = f.readlines()
            all_csv_paths = [path.strip() for path in all_csv_paths]
    
    else:
        all_csv_paths = load_dataset_files()
        os.makedirs("./cache_files", exist_ok=True)
        with open("./cache_files/all_csv_paths.txt", "w") as f:
            for path in all_csv_paths:
                f.write(path + "\n")
    
    print(f"Found {len(all_csv_paths)} CSV files.")
    
    for csv_path in tqdm(all_csv_paths):
        
        get_data(csv_path)    
    


    # step 2 – load them (can be re-run, or chunked/batched if huge)
    # metadata = load_json_files(all_json_paths)
    # print(f"Successfully loaded {len(metadata)} files.")
    # if metadata:
    #     p, sample = metadata[0]
    #     print("First file:", p)
    #     print("Top-level keys:", list(sample.keys()))
