#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1 -C 'volta|turing|ampere'
#SBATCH --cpus-per-gpu=10
#SBATCH -C 'gmem32|gmem80|gmem48|gmem24'
#SBATCH --job-name=dataset_50sites_cmap_%A_%a  # Job name includes job array ID and task ID
#SBATCH --output=slurm_outputs/dataset_50sites_cmap/dataset_50sites_cmap_%A_%a.out  # Output file for each job in the array
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos day
##SBATCH --array=0-$(($(find /home/sirsh/cv_dataset/dataset_50sites/data/ -type d -name "images" | wc -l) - 1))%2  # Dynamically set array size
#SBATCH --array=0-120%2  # Dynamically set array size

# Define base directories
base_input_dir="/home/sirsh/cv_dataset/dataset_50sites/data/"
base_metadata_dir="/home/sirsh/cv_dataset/dataset_50sites/colmap/metadata/"
base_results_dir="/home/sirsh/cv_dataset/dataset_50sites/colmap/results/"

# Generate a list of all input image folders
input_folders=($(find "$base_input_dir" -type d -name "images"))

# Select the corresponding input folder based on SLURM_ARRAY_TASK_ID
input_folder=${input_folders[$SLURM_ARRAY_TASK_ID]}

# Generate the corresponding metadata folder
relative_path=${input_folder#"$base_input_dir"}  # Get the relative path of the input folder
metadata_folder="$base_metadata_dir/$relative_path"
results_folder="$base_results_dir/${relative_path%/images}"

# Define workspace and output folders
workspace_folder="${results_folder}/workspace"
output_folder="${results_folder}/output"

# Activate the environment and run the job
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
module load use.own
module load colmap
source ~/smenv_colmap/bin/activate
nvidia-smi
squeue --me

# Display the parameters for debugging
echo "Job array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Input folder: $input_folder"
echo "Metadata folder: $metadata_folder"
echo "Workspace folder: $workspace_folder"
echo "Output folder: $output_folder"

cd /home/sirsh/wriva_2024/accenture-colmap_plus

# Run the Python script with the corresponding parameters
python3 run_wriva_reconstruction.py \
    --input_images_folder "$input_folder" \
    --workspace_folder "$workspace_folder" \
    --output_folder "$output_folder" \
    --enable_doppelganger_threshold 200 \
    --max_images 500 \
    --num_threads 6 \
    --input_metadata_folder "$metadata_folder" \
    --ground_truth_mode