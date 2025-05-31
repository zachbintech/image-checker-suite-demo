import cv2
import os

from similar import group_similar_images_in_memory

def load_images_from_directory(directory_path):
    """
    Load images from a directory.

    Args:
        directory_path (str): Path to the directory containing images.

    Returns:
        list: List of tuples (image_name, image_array).
    """
    images = []
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append((filename, image))
    return images

# Example usage

directory_path = "./similar_images"  # Replace with the path to your image directory
images = load_images_from_directory(directory_path)

# Test the group_similar_images_in_memory function
groups = group_similar_images_in_memory(images, threshold_matches=68)
print("Grouped Images:", groups)