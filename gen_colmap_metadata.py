import os
import json
import argparse
from tqdm import tqdm

def crawl_directories(base_input_dir, base_output_dir):
    """
    Crawl all directories under the base input directory and generate corresponding output directories.

    Args:
        base_input_dir (str): Base directory to crawl for input folders.
        base_output_dir (str): Base directory to create corresponding output folders.

    Returns:
        list: A list of tuples containing input and output folder paths.
    """
    input_output_pairs = []

    for root, dirs, files in os.walk(base_input_dir):
        if "images" in root:  # Look for directories containing "images"
            input_folder = root
            relative_path = os.path.relpath(root, base_input_dir)  # Get relative path from base_input_dir
            output_folder = os.path.join(base_output_dir, relative_path)  # Create corresponding output folder
            if not os.path.exists(output_folder):
                input_output_pairs.append((input_folder, output_folder))

    return input_output_pairs

def generate_metadata_for_all(base_input_dir, base_output_dir):
    """
    Generate metadata JSON files for all image folders under the base input directory.

    Args:
        base_input_dir (str): Base directory to crawl for input folders.
        base_output_dir (str): Base directory to create corresponding output folders.
    """
    # Crawl directories and get input-output folder pairs
    input_output_pairs = crawl_directories(base_input_dir, base_output_dir)

    # for input_folder, output_folder in input_output_pairs:
    #     print(f"Processing: {input_folder} -> {output_folder}")
    #     generate_metadata(input_folder, output_folder)
    for input_folder, output_folder in tqdm(input_output_pairs, desc="Processing Folders"):
        generate_metadata(input_folder, output_folder)

def generate_metadata(input_folder, output_folder):
    """
    Generate metadata JSON files for images in the input folder.

    Args:
        input_folder (str): Path to the folder containing images.
        output_folder (str): Path to the folder to save metadata files.
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Template metadata structure
    metadata_template = {
        "version": "4.2.1",
        "fname": None,
        "site": None,
        "source": None,
        "collection": None,
        "timestamp": None,
        "extrinsics": {
            "lat": None,
            "lon": None,
            "alt": None,
            "omega": None,
            "phi": None,
            "kappa": None
        },
        "projection": None,
        "calibration": None,
        "geolocation": None,
        "env_conditions": None,
        "type": None,
        "modes": None,
        "exterior": None,
        "interior": None,
        "transient_occlusions": None,
        "artifacts": None,
        "masks": None
    }

    # Get all image files from the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Generate metadata files
    for image_name in sorted(image_files):
        metadata_filename = f"{os.path.splitext(image_name)[0]}.json"
        metadata_path = os.path.join(output_folder, metadata_filename)
        
        # Update metadata template with filename and type
        metadata = metadata_template.copy()
        metadata["fname"] = image_name
        metadata["type"] = "airborne" if "street" not in image_name else "ground"
        
        # Write JSON file
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    # print(f"Metadata files saved in '{output_folder}' folder.")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate metadata for image files in a directory structure.")
    parser.add_argument("--base_input_dir", type=str, required=True, help="Base directory containing input folders")
    parser.add_argument("--base_output_dir", type=str, required=True, help="Base directory to save metadata files")
    args = parser.parse_args()

    # Generate metadata for all directories
    generate_metadata_for_all(args.base_input_dir, args.base_output_dir)