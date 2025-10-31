#!/usr/bin/env python3
"""
Standalone script to detect similar/duplicate images in a directory.
Uses the same logic as the Streamlit app's Image Similarity Check feature (DINOv2 embeddings).

Usage:
    python check_similarity.py <directory_path> [--threshold 0.8] [--output results.json]
"""

import argparse
import os
import glob
import cv2
import json
import shutil
from similarity_api import (
    compute_embeddings_for_uploaded_images,
    find_similar_pairs,
    group_similar_images_by_embeddings,
    summarize_similarity_results
)


def load_images_from_directory(directory_path):
    """
    Load all supported image files from a directory.

    Returns:
        list: List of tuples (filename, image_array)
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found: {directory_path}")
        return []

    supported_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.JPG', '*.JPEG', '*.PNG', '*.TIFF']
    image_files = []

    for ext in supported_extensions:
        image_files.extend(glob.glob(os.path.join(directory_path, ext)))

    if not image_files:
        print(f"Warning: No images found in directory: {directory_path}")
        return []

    loaded_images = []
    for img_path in sorted(image_files):
        try:
            image = cv2.imread(img_path)
            if image is not None:
                loaded_images.append((os.path.basename(img_path), image))
            else:
                print(f"Warning: Could not load {os.path.basename(img_path)}")
        except Exception as e:
            print(f"Warning: Failed to load {os.path.basename(img_path)}: {e}")

    return loaded_images


def organize_similar_images(directory_path, groups, similar_pairs, output_folder="similar_images"):
    """
    Copy similar images into organized folders.

    Args:
        directory_path: Source directory containing images
        groups: List of image groups
        similar_pairs: List of similar image pairs
        output_folder: Name of the output folder to create
    """
    # Create output directory
    output_dir = os.path.join(directory_path, output_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Find groups with more than one image (actual similar groups)
    multi_groups = [group for group in groups if len(group) > 1]

    if not multi_groups:
        print(f"\n⚠ No similar images to organize (no groups above threshold)")
        return

    print(f"\nOrganizing {len(multi_groups)} groups of similar images into: {output_dir}")

    # Copy each group into its own subfolder
    for idx, group in enumerate(multi_groups, 1):
        group_folder = os.path.join(output_dir, f"group_{idx:03d}")
        os.makedirs(group_folder, exist_ok=True)

        print(f"\n  Group {idx} ({len(group)} images):")

        for image_name in group:
            src_path = os.path.join(directory_path, image_name)
            dst_path = os.path.join(group_folder, image_name)

            try:
                shutil.copy2(src_path, dst_path)
                print(f"    ✓ Copied: {image_name}")
            except Exception as e:
                print(f"    ✗ Failed to copy {image_name}: {e}")

    print(f"\n✓ Successfully organized {len(multi_groups)} groups into {output_dir}/")


def check_similarity(directory_path, similarity_threshold=0.8):
    """
    Find similar images in a directory using DINOv2 embeddings.

    Returns:
        dict: Results containing groups and similar pairs
    """
    # Load images
    images = load_images_from_directory(directory_path)

    if len(images) < 2:
        print("Error: Need at least 2 images to detect similarities")
        return None

    print(f"Loaded {len(images)} images")
    print("\nStep 1: Computing DINOv2 embeddings...")
    print("(This may take a moment on first run as the model is downloaded)")
    print()

    # Compute embeddings and find similar pairs
    embeddings = compute_embeddings_for_uploaded_images(images, verbose=True)

    print(f"\n✓ All embeddings computed successfully!")
    print(f"\nStep 2: Finding similar pairs (threshold: {similarity_threshold})...")

    similar_pairs = find_similar_pairs(embeddings, similarity_threshold)

    print(f"✓ Found {len(similar_pairs)} similar pairs")
    print(f"\nStep 3: Grouping similar images...")

    # Build groups using the already computed embeddings
    # (We use the image names from the embeddings dictionary)
    from similarity_api import build_groups_from_pairs
    image_names = [name for name, _ in images]
    groups = build_groups_from_pairs(image_names, similar_pairs)

    # Get summary statistics
    summary = summarize_similarity_results(groups, similar_pairs)

    return {
        "summary": summary,
        "similar_pairs": similar_pairs,
        "groups": groups
    }


def main():
    parser = argparse.ArgumentParser(
        description="Detect similar/duplicate images in a directory using DINOv2 embeddings"
    )
    parser.add_argument("directory", help="Path to directory containing images")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold (0.0-1.0). Higher = more strict (default: 0.8)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Save results to JSON file"
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Copy similar images into organized folders (creates 'similar_images' directory)"
    )
    parser.add_argument(
        "--organize-folder",
        type=str,
        default="similar_images",
        help="Name of folder to organize similar images into (default: similar_images)"
    )

    args = parser.parse_args()

    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print("Error: Threshold must be between 0.0 and 1.0")
        return

    print(f"Scanning directory: {args.directory}")
    print(f"Similarity threshold: {args.threshold}")
    print("="*80 + "\n")

    # Check similarity
    results = check_similarity(args.directory, args.threshold)

    if results is None:
        return

    summary = results["summary"]
    similar_pairs = results["similar_pairs"]
    groups = results["groups"]

    # Display results
    print("\n" + "="*80)
    print("SIMILARITY DETECTION RESULTS")
    print("="*80)

    print(f"\nSummary:")
    print(f"  Total images: {summary['total_images']}")
    print(f"  Total groups: {summary['total_groups']}")
    print(f"  Similar pairs found: {summary['similarity_pairs_count']}")
    print(f"  Unique images (no matches): {summary['singleton_count']}")

    # Display similar pairs with scores
    if similar_pairs:
        print(f"\n{'='*80}")
        print("SIMILAR IMAGE PAIRS")
        print("="*80)
        for img1, img2, score in similar_pairs:
            print(f"  {img1} ↔ {img2}")
            print(f"    Similarity: {score:.3f}")

    # Display grouped images
    multi_groups = [group for group in groups if len(group) > 1]
    if multi_groups:
        print(f"\n{'='*80}")
        print("IMAGE GROUPS")
        print("="*80)
        for idx, group in enumerate(multi_groups):
            print(f"\n  Group {idx + 1} ({len(group)} images):")
            for image_name in group:
                print(f"    - {image_name}")
    else:
        print(f"\n✓ No similar image groups found with threshold {args.threshold}")

    # Organize similar images into folders if requested
    if args.organize:
        organize_similar_images(args.directory, groups, similar_pairs, args.organize_folder)

    # Save to file if requested
    if args.output:
        # Convert groups to serializable format
        serializable_groups = [[img for img in group] for group in groups]

        output_data = {
            "parameters": {
                "directory": args.directory,
                "similarity_threshold": args.threshold
            },
            "summary": {
                "total_images": summary['total_images'],
                "total_groups": summary['total_groups'],
                "similarity_pairs_count": summary['similarity_pairs_count'],
                "singleton_count": summary['singleton_count']
            },
            "similar_pairs": [
                {"image1": p[0], "image2": p[1], "similarity": float(p[2])}
                for p in similar_pairs
            ],
            "groups": serializable_groups
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
