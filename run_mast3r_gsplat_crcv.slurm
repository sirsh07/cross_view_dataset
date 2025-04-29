#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1 -C 'volta|turing|ampere'
#SBATCH --cpus-per-gpu=10
#SBATCH -C 'gmem80|gmem48'
#SBATCH --job-name=dataset_50sites_gsplat_mast3r_%A_%a  # Job name includes job array ID and task ID
#SBATCH --output=slurm_outputs/dataset_50sites_gsplat_mast3r_v2/dataset_50sites_gsplat_mast3r_%A_%a.out  # Output file for each job in the array
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos day
##SBATCH --array=0-$(($(find /home/sirsh/cv_dataset/dataset_50sites/data/ -type d -name "images" | wc -l) - 1))%2  # Dynamically set array size
#SBATCH --array=0-120  # Dynamically set array size
#SBATCH --exclude=c4-4,c4-6




# Define base directories
base_input_dir="/home/sirsh/cv_dataset/dataset_50sites/mast3r/results"
base_results_dir="/home/sirsh/cv_dataset/dataset_50sites/gsplat_mast3r/"


# Generate a list of all input image folders
input_folders=($(find "$base_input_dir" -type d -name "reconstruction"))

# Select the corresponding input folder based on SLURM_ARRAY_TASK_ID
input_folder=${input_folders[$SLURM_ARRAY_TASK_ID]}

# Generate the corresponding metadata folder
relative_path=${input_folder#"$base_input_dir"}  # Get the relative path of the input folder
results_folder="$base_results_dir/${relative_path%/reconstruction}"


# Define workspace and output folders
output_folder="${results_folder}/3dgs_recon"

mkdir -p "$output_folder"

# Activate the environment and run the job
# module load cuda/cuda-12.1.0
# module load python/python-3.11.4-gcc-12.2.0
module load cuda/12.6
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
source ~/smenv_master/bin/activate
nvidia-smi
squeue --me

# Display the parameters for debugging
echo "Job array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Input folder: $input_folder"
echo "Output folder: $output_folder"

cd /home/sirsh/cv_dataset/cross_view_dataset/gsplat/examples


CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --data_dir "$input_folder" --result_dir "$output_folder" --no_normalize_world_space
