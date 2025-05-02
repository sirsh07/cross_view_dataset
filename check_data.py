import os
from PIL import Image

def find_unopenable_images(base_dir):
    """
    Traverse the directory structure and check if all images can be opened.

    Args:
        base_dir (str): Base directory to search for images.

    Returns:
        list: A list of paths to images that cannot be opened.
    """
    unopenable_images = []

    # Walk through all directories and files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpeg', '.jpg', '.png')):  # Check for image files
                image_path = os.path.join(root, file)
                try:
                    # Attempt to open the image
                    with Image.open(image_path) as img:
                        img.verify()  # Verify that the file is a valid image
                except Exception as e:
                    # If an error occurs, add the file to the list
                    unopenable_images.append(image_path)
                    print(f"Cannot open image: {image_path} - Error: {e}")

    return unopenable_images


def check_colmap(base_dir):
    empty_sparse_dirs = []
    missing_sparse_dirs = []

    for root, dirs, files in os.walk(base_dir):
        if os.path.basename(root) == "output":
            sparse_path = os.path.join(root, "sparse")
            if os.path.exists(sparse_path):
                if not any(os.scandir(sparse_path)):  # Empty if no files or dirs
                    empty_sparse_dirs.append(sparse_path)
            else:
                missing_sparse_dirs.append(root)

    print("Folders where 'sparse' exists but is empty:")
    for d in empty_sparse_dirs:
        print(d)

    print("\nFolders where 'sparse' is missing inside 'output':")
    for d in missing_sparse_dirs:
        print(d)

    return empty_sparse_dirs, missing_sparse_dirs

if __name__ == "__main__":
    base_dir = "/home/sirsh/cv_dataset/dataset_50sites/data/"
    unopenable_images = find_unopenable_images(base_dir)
    
    
    if unopenable_images:
        print("\nThe following images cannot be opened:")
        # for image in unopenable_images:
        #     print(image)
    else:
        print("\nAll images can be opened successfully.")
        
    
    base_dir = "/home/sirsh/cv_dataset/dataset_30sites/colmap/results/"
    empty_dirs, missing_dirs = check_colmap(base_dir)
    if empty_dirs:
        print("\nThe following sparse directories are empty:")
        for d in empty_dirs:
            print(d)
    else:
        print("\nNo empty sparse directories found.")
    if missing_dirs:
        print("\nThe following sparse directories are missing:")
        for d in missing_dirs:
            print(d)
    else:
        print("\nNo missing sparse directories found.")    
        