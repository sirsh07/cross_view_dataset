#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1 -C 'volta|turing|ampere'
#SBATCH --cpus-per-gpu=10
#SBATCH -C 'gmem32|gmem80|gmem48|gmem24'
##SBATCH -C gmem48
#SBATCH --job-name=ultrra25_cmap_%A_%a  # Job name includes job array ID and task ID
#SBATCH --output=slurm_outputs/ultrra25_cmap/ultrra25_cmap_%A_%a.out  # Output file for each job in the array
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos day
#SBATCH --array=0-7  # Create an array with 4 jobs (task IDs 0 to 3)

# Define input image folders
declare -a input_folders=(
    "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_data/ultrra25-dev-inputs-v01/inputs/view_synthesis/t01_v09_s00_r01_ImageDensity_WACV_dev_A01/train"
    "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_data/ultrra25-dev-inputs-v01/inputs/view_synthesis/t02_v06_s00_r01_CameraModels_WACV_dev_A01/train"
    "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_data/ultrra25-dev-inputs-v01/inputs/view_synthesis/t03_v06_s00_r01_ReconstructedArea_WACV_dev_A01/train"
    "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_data/ultrra25-dev-inputs-v01/inputs/view_synthesis/t04_v11_s00_r01_VaryingAltitudes_WACV_dev_A01/train"
    "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_data/ultrra25-test-inputs-v01/inputs/view_synthesis/t01_v10_s00_r01_ImageDensity_WACV_test_A09/train"
    "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_data/ultrra25-test-inputs-v01/inputs/view_synthesis/t02_v07_s00_r01_CameraModels_WACV_test_A09/train"
    "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_data/ultrra25-test-inputs-v01/inputs/view_synthesis/t03_v07_s00_r01_ReconstructedArea_WACV_test_A09/train"
    "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_data/ultrra25-test-inputs-v01/inputs/view_synthesis/t04_v12_s00_r01_VaryingAltitudes_WACV_test_A09/train"
)

# Select the corresponding input folder based on SLURM_ARRAY_TASK_ID
input_folder=${input_folders[$SLURM_ARRAY_TASK_ID]}

# Define workspace and output folders
output_folder="${input_folder/ultraa_data/ultraa_colmap}/output"
workspace_folder="${input_folder/ultraa_data/ultraa_colmap}/workspace"

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
echo "Workspace folder: $workspace_folder"
echo "Output folder: $output_folder"

# Run the Python script with the corresponding parameters
python3 run_wriva_reconstruction.py \
    --input_images_folder "$input_folder" \
    --workspace_folder "$workspace_folder" \
    --output_folder "$output_folder" \
    --enable_doppelganger_threshold 200 \
    --max_images 500 \
    --num_threads 6 \
    --input_metadata_folder "$input_folder" \
    --ground_truth_mode


# python3 run_wriva_reconstruction.py --input_images_folder "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_data/ultrra25-dev-inputs-v01/inputs/view_synthesis/t01_v09_s00_r01_ImageDensity_WACV_dev_A01/train" --workspace_folder "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_colmap/ultrra25-dev-inputs-v01/inputs/view_synthesis/t01_v09_s00_r01_ImageDensity_WACV_dev_A01/train/workspace" --output_folder "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_colmap/ultrra25-dev-inputs-v01/inputs/view_synthesis/t01_v09_s00_r01_ImageDensity_WACV_dev_A01/train/output"  --enable_doppelganger_threshold 200 --max_images 500 --num_threads 6 --input_metadata_folder "/home/sirsh/cv_gen_eccv25/ultrra25/ultraa_data/ultrra25-dev-inputs-v01/inputs/view_synthesis/t01_v09_s00_r01_ImageDensity_WACV_dev_A01/train" --ground_truth_mode

