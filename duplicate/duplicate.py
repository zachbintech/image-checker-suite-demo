import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dino_embeddings import compute_embedding  # Import the function from dino_embedding.py

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
