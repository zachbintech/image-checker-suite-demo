import cv2
import os


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

def get_all_image_paths_recursive(root_dir):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(dirpath, f))
    return image_paths