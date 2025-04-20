import argparse
import os

def run_colmap(dataset_path):
    """
    Run COLMAP on the given dataset path.
    """
    print(f"Running COLMAP on dataset at: {dataset_path}")
    # Here you would add the actual COLMAP command
    # For example:
    # subprocess.run(["colmap", "feature_extractor", "--database_path", f"{dataset_path}/database.db", ...])
    
    
def run_mast3r(dataset_path):
    
    """
    Run MAST3R on the given dataset path.
    """
    print(f"Running MAST3R on dataset at: {dataset_path}")
    # Here you would add the actual MAST3R command
    # For example:
    # subprocess.run(["mast3r", "--input", dataset_path, ...])
    
def process_dataset(aerial_dataset_path, ground_dataset_path, results_dir):
    """
    Process the dataset using COLMAP and MAST3R.
    """
    
    # create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # create log file
    log_file = os.path.join(results_dir, "process_log.txt")
    with open(log_file, "w") as f:
        f.write("Processing dataset...\n")
        f.write(f"Aerial dataset path: {aerial_dataset_path}\n")
        f.write(f"Ground dataset path: {ground_dataset_path}\n")
        
    # list files in the dataset directories and write to log in same line
    f.write("Files in aerial dataset:\n")
    for root, dirs, files in os.walk(aerial_dataset_path):
        for file in files:
            f.write(f"{file}, ")
    f.write("\n")
    f.write("Files in ground dataset:\n")
    for root, dirs, files in os.walk(ground_dataset_path):
        for file in files:
            f.write(f"{file}, ")
    f.write("\n")
    
    # create a new directory for combined aerial and ground and symlink to the original
    combined_dataset_path = os.path.join(results_dir, "combined_dataset")
    os.makedirs(combined_dataset_path, exist_ok=True)
    os.symlink(aerial_dataset_path, os.path.join(combined_dataset_path, "aerial"))
    os.symlink(ground_dataset_path, os.path.join(combined_dataset_path, "ground"))
    
    
    # create a loop with 3 directories, ground dataset, aerial dataset, and combined dataset
    dataset_paths = [ground_dataset_path, aerial_dataset_path, combined_dataset_path]
    
    # take a random sample of 12%, 25%, 50%, 100%  of the files in each directory
    sample_sizes = [0.12, 0.25, 0.5, 1.0]
    
    # create new directories for each combination of dataset and sample size and symlink to the original
    for dataset_path in dataset_paths:
        for sample_size in sample_sizes:
            sample_dir = os.path.join(results_dir, f"sample_{int(sample_size*100)}")
            os.makedirs(sample_dir, exist_ok=True)
            sample_path = os.path.join(sample_dir, os.path.basename(dataset_path))
            os.symlink(dataset_path, sample_path)
            


def main():
    
    parser = argparse.ArgumentParser(description="Process dataset paths.")
    
    # Add arguments for ground and aerial dataset paths
    parser.add_argument("--ground_dataset_path", type=str, help="Path to the ground dataset")
    parser.add_argument("--aerial_dataset_path", type=str, help="Path to the aerial dataset")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")
    
    args = parser.parse_args()

    ground_dataset_path = args.ground_dataset_path
    aerial_dataset_path = args.aerial_dataset_path

    print(f"Ground dataset path provided: {ground_dataset_path}")
    print(f"Aerial dataset path provided: {aerial_dataset_path}")
    
    



if __name__ == "__main__":
    main()