import cv2
import numpy as np
import os

import file_util

def detect_exposure(image_path, overexposure_thresh=0.95, underexposure_thresh=0.05, pixel_ratio_thresh=0.05):
    """
    Detect if an image is overexposed or underexposed.
    
    Args:
        image_path (str): Path to the image file.
        overexposure_thresh (float): Pixel brightness threshold for overexposure (0.0-1.0).
        underexposure_thresh (float): Pixel brightness threshold for underexposure (0.0-1.0).
        pixel_ratio_thresh (float): Fraction of total pixels allowed to trigger a warning.

    Returns:
        str: 'overexposed', 'underexposed', or 'normal'
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Normalize pixel values to [0, 1]
    image = image / 255.0

    # Find very bright and very dark pixels
    overexposed_pixels = np.sum(image > overexposure_thresh)
    underexposed_pixels = np.sum(image < underexposure_thresh)
    total_pixels = image.size

    overexposed_ratio = overexposed_pixels / total_pixels
    underexposed_ratio = underexposed_pixels / total_pixels

    if overexposed_ratio > pixel_ratio_thresh:
        return "overexposed"
    elif underexposed_ratio > pixel_ratio_thresh:
        return "underexposed"
    else:
        return "normal"

import os
import shutil

def scan_and_copy(folder_path, temp_folder="temp_exposure"):
    """
    Scan all images in a folder, detect exposure status,
    and copy them into temp folders by status.

    Args:
        folder_path (str): Folder containing images.
        temp_folder (str): Temporary parent folder to store categorized images.
    """
    # Create the temp exposure folders
    categories = ['overexposed', 'underexposed', 'normal']
    for category in categories:
        os.makedirs(os.path.join(temp_folder, category), exist_ok=True)

    print(f"Scanning folder: {folder_path}")
    print(os.listdir(folder_path))

    files = file_util.get_all_image_paths_recursive(folder_path)


    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                status = detect_exposure(file)
                print(f"{file}: {status}")

                # Define destination path
                dest_path = os.path.join(temp_folder, status, file.split(os.sep)[-1])

                # Copy the file
                shutil.copy2(file, dest_path)

            except Exception as e:
                print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    # folder = input("Enter the path to the image folder: ")
    folder = "/home/zach/Desktop/PhotoData"
    scan_and_copy(folder)
