#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8G
#SBATCH --constraint='gpu80'
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --error=pslurm_outputs/dataset_50sites_gsplat/dataset_50sites_gsplat_%A_%a_%J.err
#SBATCH --output=slurm_outputs/dataset_50sites_gsplat/dataset_50sites_gsplat_%A_%a_%J.out
#SBATCH --job-name=dataset_50sites_gsplat_%A_%a
#SBATCH --array=0-120  # Dynamically set array size
##SBATCH --exclude=evc43
## SBATCH --array=0-11  # Create an array with 2 jobs (task IDs 0 to 1)


# Define base directories
base_input_dir="/home/smitra/cv_dataset/dataset_50sites/colmap/results/"
base_results_dir="/home/smitra/cv_dataset/dataset_50sites/gsplat/"


# Generate a list of all input image folders
input_folders=($(find "$base_input_dir" -type d -name "output"))

# Select the corresponding input folder based on SLURM_ARRAY_TASK_ID
input_folder=${input_folders[$SLURM_ARRAY_TASK_ID]}

# Generate the corresponding metadata folder
relative_path=${input_folder#"$base_input_dir"}  # Get the relative path of the input folder
results_folder="$base_results_dir/${relative_path%/output}"


# Define workspace and output folders
output_folder="${results_folder}/3dgs_recon"

mkdir -p "$output_folder"

# Activate the environment and run the job
module load cuda/cuda-12.1.0
module load python/python-3.11.4-gcc-12.2.0
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
source ~/smenv_wriva/bin/activate
nvidia-smi
squeue --me

# Display the parameters for debugging
echo "Job array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Input folder: $input_folder"
echo "Output folder: $output_folder"

cd /home/smitra/cv_dataset/cross_view_dataset/gsplat/examples


CUDA_VISIBLE_DEVICES=0 python simple_trainer.py mcmc --data_dir "$input_folder" --result_dir "$output_folder" --no_normalize_world_space
