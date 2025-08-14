"""
API functions for image similarity detection in Streamlit uploads using DINOv2 embeddings.
Adapted from evaluate_groupings.py and similar_core.py functionality.
"""
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from sklearn.metrics.pairwise import cosine_similarity
from similar.dino_embeddings import compute_embedding
from typing import List, Tuple, Dict


def compute_embeddings_for_uploaded_images(images: List[Tuple[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Compute DINOv2 embeddings for uploaded images by saving them temporarily.
    
    Args:
        images: List of (image_name, image_array) tuples
        
    Returns:
        Dictionary mapping image names to their embedding vectors
    """
    embeddings = {}
    temp_files = []
    
    try:
        # Create temporary files for each image
        for image_name, image_array in images:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.jpg', prefix='streamlit_')
            temp_files.append((temp_fd, temp_path))
            
            # Convert BGR to RGB and save
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(temp_path, 'JPEG')
            
            # Compute embedding
            embedding = compute_embedding(temp_path)
            embeddings[image_name] = embedding
            
    finally:
        # Clean up temporary files
        for temp_fd, temp_path in temp_files:
            try:
                os.close(temp_fd)
                os.unlink(temp_path)
            except:
                pass
    
    return embeddings


def find_similar_pairs(embeddings: Dict[str, np.ndarray], similarity_threshold: float = 0.8) -> List[Tuple[str, str, float]]:
    """
    Find pairs of similar images based on cosine similarity of embeddings.
    
    Args:
        embeddings: Dictionary mapping image names to embedding vectors
        similarity_threshold: Minimum cosine similarity to consider images similar
        
    Returns:
        List of (image1_name, image2_name, similarity_score) tuples
    """
    similar_pairs = []
    image_names = list(embeddings.keys())
    
    for i in range(len(image_names)):
        for j in range(i + 1, len(image_names)):
            name1, name2 = image_names[i], image_names[j]
            
            # Flatten embeddings to ensure they are 1D
            embedding1 = embeddings[name1].flatten()
            embedding2 = embeddings[name2].flatten()
            
            # Compute cosine similarity
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            if similarity >= similarity_threshold:
                similar_pairs.append((name1, name2, similarity))
    
    return similar_pairs


def group_similar_images_by_embeddings(images: List[Tuple[str, np.ndarray]], similarity_threshold: float = 0.8) -> List[List[str]]:
    """
    Group similar images using DINOv2 embeddings and cosine similarity.
    
    Args:
        images: List of (image_name, image_array) tuples
        similarity_threshold: Minimum cosine similarity to group images together
        
    Returns:
        List of groups, where each group is a list of image names
    """
    if not images:
        return []
    
    # Compute embeddings for all images
    embeddings = compute_embeddings_for_uploaded_images(images)
    
    # Find similar pairs
    similar_pairs = find_similar_pairs(embeddings, similarity_threshold)
    
    # Build groups using union-find approach
    groups = {}  # image_name -> group_id
    group_members = {}  # group_id -> set of image_names
    next_group_id = 0
    
    # Initialize each image as its own group
    for image_name, _ in images:
        groups[image_name] = next_group_id
        group_members[next_group_id] = {image_name}
        next_group_id += 1
    
    # Merge groups based on similarity pairs
    for name1, name2, similarity in similar_pairs:
        group1 = groups[name1]
        group2 = groups[name2]
        
        if group1 != group2:
            # Merge smaller group into larger group
            if len(group_members[group1]) < len(group_members[group2]):
                group1, group2 = group2, group1
            
            # Update group assignments
            for member in group_members[group2]:
                groups[member] = group1
            
            # Merge group members
            group_members[group1].update(group_members[group2])
            del group_members[group2]
    
    # Convert to list of lists
    result_groups = [list(members) for members in group_members.values()]
    
    return result_groups


def summarize_similarity_results(groups: List[List[str]], similar_pairs: List[Tuple[str, str, float]]) -> Dict:
    """
    Create a summary of similarity detection results for Streamlit display.
    
    Args:
        groups: List of image groups
        similar_pairs: List of similar image pairs with scores
        
    Returns:
        Dictionary with similarity statistics
    """
    group_sizes = [len(group) for group in groups]
    singleton_count = sum(1 for size in group_sizes if size == 1)
    multi_image_groups = [group for group in groups if len(group) > 1]
    
    return {
        "total_groups": len(groups),
        "total_images": sum(group_sizes),
        "singleton_count": singleton_count,
        "multi_image_groups": multi_image_groups,
        "similar_pairs": similar_pairs,
        "similarity_pairs_count": len(similar_pairs)
    }