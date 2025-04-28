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

if __name__ == "__main__":
    base_dir = "/home/sirsh/cv_dataset/dataset_50sites/data/"
    unopenable_images = find_unopenable_images(base_dir)

    if unopenable_images:
        print("\nThe following images cannot be opened:")
        # for image in unopenable_images:
        #     print(image)
    else:
        print("\nAll images can be opened successfully.")