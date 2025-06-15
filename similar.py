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

import os
import cv2
import torch
import numpy as np
from PIL import Image
import imagehash
from sklearn.metrics.pairwise import cosine_similarity
# Optional PCA support
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import normalize

from dino_embeddings import compute_embedding  # Import your DINOv2 embedding function


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




def compute_and_store_hashes(image_paths):
    """
    Compute perceptual hashes (phash) for each image.

    Args:
        image_paths (list): List of image file paths.

    Returns:
        dict: A dictionary mapping image paths to their phash.
    """
    hashes = {}
    for path in image_paths:
        pil_image = Image.open(path)
        hashes[path] = imagehash.phash(pil_image)
    return hashes


def group_images_by_hash(hashes, threshold=5):
    """
    Group images based on perceptual hash differences.

    Args:
        hashes (dict): Mapping from image paths to phashes.
        threshold (int): Maximum Hamming distance to group images.

    Returns:
        list: Groups of similar images based on hashing.
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


def compute_image_embeddings(image_paths):
    """
    Compute DINOv2 embeddings for a list of image paths.

    Args:
        image_paths (list): List of image file paths.

    Returns:
        dict: Mapping from image paths to their embedding vectors.
    """
    return {path: compute_embedding(path) for path in image_paths}


def compute_cosine_similarities(embeddings, similarity_threshold=0.9):
    """
    Compute pairwise cosine similarity between embeddings.

    Args:
        embeddings (dict): Mapping from image paths to embedding vectors.
        similarity_threshold (float): Threshold above which images are considered similar.

    Returns:
        dict: Similar image pairs and their similarity score.
    """
    similarities = {}
    image_list = list(embeddings.keys())

    for i in range(len(image_list)):
        for j in range(i + 1, len(image_list)):
            path1, path2 = image_list[i], image_list[j]
            
            # Flatten embeddings to ensure they are 1D
            embedding1 = embeddings[path1].flatten()
            embedding2 = embeddings[path2].flatten()
            
            sim = cosine_similarity([embedding1], [embedding2])[0][0]
            if sim >= similarity_threshold:
                similarities[frozenset([path1, path2])] = sim
                print(f"Similarity between {path1} and {path2}: {sim:.4f}")
    return similarities


def find_duplicate_images_in_directory(directory_path, hash_threshold=5, similarity_threshold=0.9):
    """
    Full pipeline: Detect near-duplicate images using hash filtering and DINO embedding similarity.

    Args:
        directory_path (str): Path to the image directory.
        hash_threshold (int): Max phash Hamming distance for grouping.
        similarity_threshold (float): Cosine similarity threshold for duplicates.

    Returns:
        dict: Similar image pairs with similarity scores.
    """
    # Step 1: Gather image paths
    image_paths = [
        os.path.abspath(os.path.join(directory_path, file))
        for file in os.listdir(directory_path)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ]

    if not image_paths:
        print("No valid image files found in directory.")
        return {}

    # Step 2: Compute perceptual hashes
    hashes = compute_and_store_hashes(image_paths)

    # Step 3: Group images that are visually similar based on hash
    grouped_images = group_images_by_hash(hashes, threshold=hash_threshold)

    # Step 4: Flatten grouped images and remove duplicates
    unique_images = sorted({img for group in grouped_images for img in group})

    # Step 5: Compute DINOv2 embeddings
    embeddings = compute_image_embeddings(unique_images)

    # (Optional) PCA normalization - uncomment to use
    # vectors = np.vstack(list(embeddings.values()))
    # pca = PCA(n_components=100).fit_transform(vectors)
    # normalized = normalize(pca)
    # embeddings = {k: normalized[i] for i, k in enumerate(embeddings)}

    # Step 6: Compute cosine similarities
    similarities = compute_cosine_similarities(embeddings, similarity_threshold)

    return similarities


