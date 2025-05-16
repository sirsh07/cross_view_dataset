import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, simpledialog

def adjust_pan_angle(camera_data, pan_offset):
    """
    Adjusts the pan angle (rotation.z) of each camera frame by pan_offset degrees.

    Args:
        camera_data (dict): The JSON data containing the cameraFrames.
        pan_offset (float): The degree offset to be added to each frame's current pan angle.
    
    Returns:
        dict: The updated camera data.
    """
    if "cameraFrames" not in camera_data:
        raise KeyError("The JSON data does not contain 'cameraFrames'.")

    for frame in camera_data["cameraFrames"]:
        if "rotation" in frame and "z" in frame["rotation"]:
            original_angle = frame["rotation"]["z"]
            frame["rotation"]["z"] = original_angle + pan_offset
            # Optional: Normalize the angle to the range 0-360:
            # frame["rotation"]["z"] %= 360
    return camera_data

def main():
    # Create a root window and hide it
    root = tk.Tk()
    root.withdraw()
    root.update()

    # Open a file dialog to pick the input JSON file
    input_file = filedialog.askopenfilename(
        title="Select Input JSON File",
        filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
    )
    if not input_file:
        print("No file selected. Exiting.")
        sys.exit(1)

    # Ask user for the pan offset via an input dialog (as a float)
    pan_offset = simpledialog.askfloat("Pan Offset", "Enter the pan offset in degrees:")
    if pan_offset is None:
        print("No pan offset provided. Exiting.")
        sys.exit(1)

    # Read the input JSON file
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading the file: {e}")
        sys.exit(1)

    # Adjust the pan angle
    try:
        adjusted_data = adjust_pan_angle(data, pan_offset)
    except Exception as e:
        print(f"Error adjusting pan angle: {e}")
        sys.exit(1)

    # Construct the output filename by appending '_adjusted' before the file extension
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_adjusted{ext}"

    # Write the adjusted data to the output JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(adjusted_data, f, indent=4)
    except Exception as e:
        print(f"Error writing the output file: {e}")
        sys.exit(1)

    print(f"Updated pan angle by {pan_offset} degrees and saved to {output_file}.")
    root.destroy()

if __name__ == "__main__":
    main()
