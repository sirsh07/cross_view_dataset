#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus=1 -C 'volta|turing|ampere'
#SBATCH --cpus-per-gpu=10
#SBATCH -C 'gmem32|gmem80|gmem48|gmem24'
#SBATCH --job-name=dataset_30sites_mast3r_%A_%a  # Job name includes job array ID and task ID
#SBATCH --output=slurm_outputs/dataset_30sites_mast3r_v2/dataset_30sites_cmap_%A_%a.out  # Output file for each job in the array
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos short
##SBATCH --array=0-$(($(find /home/sirsh/cv_dataset/dataset_30sites/data/ -type d -name "images" | wc -l) - 1))%2  # Dynamically set array size
#SBATCH --array=0-120%15  # Dynamically set array size

# Define base directories
base_input_dir="/home/sirsh/cv_dataset/dataset_30sites/data/"
# base_metadata_dir="/home/sirsh/cv_dataset/dataset_30sites/colmap/metadata/"
base_results_dir="/home/sirsh/cv_dataset/dataset_30sites/mast3r_sfm_v2/results/"

# # Generate a list of all input image folders
# input_folders=($(find "$base_input_dir" -type d -name "images"))

# List of allowed folder names
allowed_folders=("ID0001" "ID0004" "ID0007" "ID0012" "ID0015" "ID0003" "ID0005" "ID0008" "ID0013" "ID0016")

# Generate a list of input folders that match the allowed folder names and contain "images"
input_folders=()
for folder in "${allowed_folders[@]}"; do
    # Search for the folder recursively under the base_input_dir and ensure it contains "images"
    found_folders=$(find "$base_input_dir" -type d -name "$folder" -exec find {} -type d -name "images" \;)
    for found_folder in $found_folders; do
        input_folders+=("$found_folder")
    done
done


# Select the corresponding input folder based on SLURM_ARRAY_TASK_ID
input_folder=${input_folders[$SLURM_ARRAY_TASK_ID]}

# Generate the corresponding metadata folder
relative_path=${input_folder#"$base_input_dir"}  # Get the relative path of the input folder
# metadata_folder="$base_metadata_dir/$relative_path"
results_folder="$base_results_dir/${relative_path%/images}"


# Define workspace and output folders
# workspace_folder="${results_folder}/workspace"
pairs_file="${results_folder}/pairs.txt"
output_folder="${results_folder}/output"

# mkdir -p "$workspace_folder"
mkdir -p "$output_folder"

# Activate the environment and run the job
echo "Running on node: $(hostname)"
echo "Running on node: $SLURMD_NODENAME"
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
# module load use.own
# module load colmap
# source ~/smenv_colmap/bin/activate
module load cuda/12.1
source ~/smenv_master/bin/activate
nvidia-smi
squeue --me

# Display the parameters for debugging
echo "Job array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Input folder: $input_folder"
# echo "Metadata folder: $metadata_folder"
# echo "Workspace folder: $workspace_folder"
echo "Pairs file: $pairs_file"
echo "Output folder: $output_folder"

cd /home/sirsh/cv_dataset/cross_view_dataset/mast3r_sfm/mast3r


python make_pairs.py \
    --dir "$input_folder" \
    --output "$pairs_file" \
    --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --scene_graph swin 
    # --scene_graph complete \

python kapture_mast3r_mapping.py \
    --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
    --dir_same_camera "$input_folder" \
    --output "$output_folder" \
    --pairsfile_path "$pairs_file"



# python make_pairs.py --dir "$input_folder" --output "$pairs_file" --scene_graph complete --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

# python kapture_mast3r_mapping.py --weights /home/sirsh/aerial_gen/aerial_scene_gen/master_sfm/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth --dir_same_camera /home/sirsh/cv_dataset/dataset_30sites/data/aerial/train/ID0003/p12/images --output /home/sirsh/cv_dataset/mastrf_temp/instant_splat/ID0003/output/ --pairsfile_path /home/sirsh/cv_dataset/mastrf_temp/instant_splat/ID0003/pairs.txt


# Run the Python script with the corresponding parameters
# python3 run_wriva_reconstruction.py \
#     --input_images_folder "$input_folder" \
#     --workspace_folder "$workspace_folder" \
#     --output_folder "$output_folder" \
#     --enable_doppelganger_threshold 200 \
#     --max_images 500 \
#     --num_threads 6 \
#     --input_metadata_folder "$metadata_folder" \
#     --ground_truth_mode