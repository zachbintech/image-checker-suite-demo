import cv2
import numpy as np
import os
import shutil
from PIL import Image
import imagehash
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dino_embeddings import compute_embedding  # Import the function from dino_embedding.py

sift = cv2.SIFT_create()


def find_duplicate_images_in_directory(directory_path):
    """
    Computes embeddings for all images in a directory and finds duplicates based on cosine similarity.

    Args:
        directory_path (str): Path to the directory containing image files.

    Returns:
        dict: A dictionary of similarity scores between image pairs.
    """
    # Get all image file paths in the directory
    image_paths = [
        os.path.abspath(os.path.join(directory_path, file))
        for file in os.listdir(directory_path)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
    ]

    # Store embeddings for multiple images
    embeddings = {path: compute_embedding(path) for path in image_paths}

    # Compare embeddings to find duplicates
    similarities = {}
    for i, (path1, embedding1) in enumerate(embeddings.items()):
        for j, (path2, embedding2) in enumerate(embeddings.items()):
            if i < j:  # Avoid duplicate comparisons
                similarity = cosine_similarity(embedding1, embedding2)
                similarities[frozenset([path1, path2])] = similarity[0][0]
                print(f"Similarity between {path1} and {path2}: {similarity[0][0]}")

    return similarities


def get_descriptors(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

def match_descriptors(desc1, desc2):
    if desc1 is None or desc2 is None:
        return 0
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    # Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return len(good_matches)


def get_hash_difference(image1, image2):
    """
    Calculate the perceptual hash difference between two images.

    Args:
        image1 (numpy.ndarray): First image as a NumPy array.
        image2 (numpy.ndarray): Second image as a NumPy array.

    Returns:
        int: The Hamming distance between the perceptual hashes of the two images.
    """
    # Convert images to PIL format
    pil_image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    pil_image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

    # Compute perceptual hashes
    hash1 = imagehash.phash(pil_image1)
    hash2 = imagehash.phash(pil_image2)

    # Return the Hamming distance between the hashes
    return hash1 - hash2




def compute_and_store_hashes(image_dir):
    """
    Compute and store perceptual hashes for all images in a directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        dict: A dictionary where keys are image file names and values are their perceptual hashes.
    """
    hashes = {}
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = cv2.imread(file_path)
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            hashes[filename] = imagehash.phash(pil_image)
    return hashes


def group_images_by_hash(hashes, threshold=5):
    """
    Group images based on perceptual hash differences.

    Args:
        hashes (dict): A dictionary where keys are image file names and values are their perceptual hashes.
        threshold (int): Maximum Hamming distance to consider images as similar.

    Returns:
        list: Groups of similar images, where each group is a list of image file names.
    """
    groups = []
    visited = set()

    for image1, hash1 in hashes.items():
        if image1 in visited:
            continue
        group = [image1]
        visited.add(image1)
        for image2, hash2 in hashes.items():
            if image2 in visited:
                continue
            if abs(hash1 - hash2) <= threshold:
                group.append(image2)
                visited.add(image2)
        groups.append(group)

    return groups

import os

def process_and_group_images(directory, threshold=5):
    """
    Compute hashes for images in a directory and group them by similarity.

    Args:
        directory (str): Path to the directory containing images.
        threshold (int): Threshold for grouping similar images.

    Returns:
        list: A list of groups, where each group is a list of image filenames.
    """
    if not os.path.isdir(directory):
        raise ValueError(f"The provided path '{directory}' is not a valid directory.")

    # Compute hashes and group images
    hashes = compute_and_store_hashes(directory)
    groups = group_images_by_hash(hashes, threshold=threshold)
    return groups

