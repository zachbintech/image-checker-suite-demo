#!/usr/bin/env python3
"""
Standalone script to detect color artifacts (red, green, blue lines) in images.
Uses the same logic as the Streamlit app's Artifact Highlighting feature.

Usage:
    python check_artifacts.py <directory_path> [--threshold 1.5] [--output results.json] [--save-masks]
"""

import argparse
import os
import glob
import cv2
import numpy as np
import json


def get_red_mask(img, threshold=1.5):
    """Get red mask using ratio approach."""
    b, g, r = cv2.split(img.astype(np.float32))

    # Avoid divide by zero
    epsilon = 1e-6
    r_ratio = r / (g + b + epsilon)

    # Threshold for "too red"
    red_mask = (r_ratio > threshold).astype(np.uint8) * 255
    return red_mask


def get_green_mask(img, threshold=1.5):
    """Get green mask using ratio approach."""
    b, g, r = cv2.split(img.astype(np.float32))

    # Avoid divide by zero
    epsilon = 1e-6
    g_ratio = g / (r + b + epsilon)

    # Threshold for "too green"
    green_mask = (g_ratio > threshold).astype(np.uint8) * 255
    return green_mask


def get_blue_mask(img, threshold=1.5):
    """Get blue mask using ratio approach."""
    b, g, r = cv2.split(img.astype(np.float32))

    # Avoid divide by zero
    epsilon = 1e-6
    b_ratio = b / (r + g + epsilon)

    # Threshold for "too blue"
    blue_mask = (b_ratio > threshold).astype(np.uint8) * 255
    return blue_mask


def detect_artifacts(image, threshold=1.5):
    """
    Detect red, green, and blue line artifacts in an image.

    Returns:
        dict: Contains masks and pixel counts for each color
    """
    red_mask = get_red_mask(image, threshold)
    green_mask = get_green_mask(image, threshold)
    blue_mask = get_blue_mask(image, threshold)

    return {
        "red_mask": red_mask,
        "red_count": int(np.count_nonzero(red_mask)),
        "green_mask": green_mask,
        "green_count": int(np.count_nonzero(green_mask)),
        "blue_mask": blue_mask,
        "blue_count": int(np.count_nonzero(blue_mask))
    }


def load_images_from_directory(directory_path):
    """Load all supported image files from a directory."""
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

    return sorted(image_files)


def check_artifacts_in_directory(directory_path, threshold=1.5, save_masks=False):
    """
    Check for color artifacts in all images in a directory.

    Returns:
        list: Results for each image
    """
    image_files = load_images_from_directory(directory_path)

    if not image_files:
        return []

    results = []
    masks_dir = None

    if save_masks:
        masks_dir = os.path.join(directory_path, "artifact_masks")
        os.makedirs(masks_dir, exist_ok=True)
        print(f"Saving masks to: {masks_dir}\n")

    for img_path in image_files:
        image = cv2.imread(img_path)
        if image is None:
            results.append({
                "filename": os.path.basename(img_path),
                "error": "Failed to load image"
            })
            continue

        # Detect artifacts
        artifact_data = detect_artifacts(image, threshold)

        result = {
            "filename": os.path.basename(img_path),
            "red_artifact_pixels": artifact_data["red_count"],
            "green_artifact_pixels": artifact_data["green_count"],
            "blue_artifact_pixels": artifact_data["blue_count"],
            "total_artifact_pixels": artifact_data["red_count"] + artifact_data["green_count"] + artifact_data["blue_count"],
            "image_size": image.shape[0] * image.shape[1],
            "artifact_percentage": (artifact_data["red_count"] + artifact_data["green_count"] + artifact_data["blue_count"]) / (image.shape[0] * image.shape[1]) * 100
        }

        results.append(result)

        # Save masks if requested
        if save_masks:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(masks_dir, f"{base_name}_red_mask.png"), artifact_data["red_mask"])
            cv2.imwrite(os.path.join(masks_dir, f"{base_name}_green_mask.png"), artifact_data["green_mask"])
            cv2.imwrite(os.path.join(masks_dir, f"{base_name}_blue_mask.png"), artifact_data["blue_mask"])

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Detect color artifacts (red, green, blue lines) in images"
    )
    parser.add_argument("directory", help="Path to directory containing images")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="Color ratio threshold (default: 1.5). E.g., red/(green+blue) > threshold"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Save results to JSON file"
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save artifact mask images to artifact_masks subdirectory"
    )

    args = parser.parse_args()

    print(f"Scanning directory: {args.directory}")
    print(f"Color ratio threshold: {args.threshold}")
    print("="*80 + "\n")

    # Check artifacts
    results = check_artifacts_in_directory(
        args.directory,
        threshold=args.threshold,
        save_masks=args.save_masks
    )

    if not results:
        return

    # Display results
    print("ARTIFACT DETECTION RESULTS")
    print("="*80)

    for result in results:
        print(f"\n{result['filename']}")
        if "error" in result:
            print(f"  ❌ {result['error']}")
        else:
            print(f"  Red artifact pixels: {result['red_artifact_pixels']:,}")
            print(f"  Green artifact pixels: {result['green_artifact_pixels']:,}")
            print(f"  Blue artifact pixels: {result['blue_artifact_pixels']:,}")
            print(f"  Total artifact pixels: {result['total_artifact_pixels']:,}")
            print(f"  Artifact percentage: {result['artifact_percentage']:.3f}%")

    print("\n" + "="*80)

    # Summary statistics
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        images_with_red = sum(1 for r in valid_results if r['red_artifact_pixels'] > 0)
        images_with_green = sum(1 for r in valid_results if r['green_artifact_pixels'] > 0)
        images_with_blue = sum(1 for r in valid_results if r['blue_artifact_pixels'] > 0)
        images_with_any = sum(1 for r in valid_results if r['total_artifact_pixels'] > 0)

        print(f"\nSummary:")
        print(f"  Total images processed: {len(valid_results)}")
        print(f"  Images with red artifacts: {images_with_red}")
        print(f"  Images with green artifacts: {images_with_green}")
        print(f"  Images with blue artifacts: {images_with_blue}")
        print(f"  Images with any artifacts: {images_with_any}")
        print(f"  Clean images: {len(valid_results) - images_with_any}")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "parameters": {
                    "directory": args.directory,
                    "threshold": args.threshold
                },
                "results": results
            }, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
