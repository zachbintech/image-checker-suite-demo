#!/usr/bin/env python3
"""
Standalone script to check image quality (blur and exposure) for all images in a directory.
Uses the same logic as the Streamlit app's Basic Image Quality Check feature.

Usage:
    python check_image_quality.py <directory_path> [--blur-threshold 100] [--exposure-threshold 0.05] [--output results.json]
"""

import argparse
import os
import glob
import cv2
import numpy as np
import json
from image_quality_checks.basic_image_quality import detect_blur


def detect_exposure(image, overexposure_thresh=0.95, underexposure_thresh=0.05, pixel_ratio_thresh=0.05):
    """
    Detect if an image array is overexposed or underexposed.

    Args:
        image (numpy.ndarray): Image array (BGR format from OpenCV).
        overexposure_thresh (float): Pixel brightness threshold for overexposure (0.0-1.0).
        underexposure_thresh (float): Pixel brightness threshold for underexposure (0.0-1.0).
        pixel_ratio_thresh (float): Fraction of total pixels allowed to trigger a warning.

    Returns:
        tuple: (status, overexposed_ratio, underexposed_ratio)
    """
    # Convert to grayscale and normalize
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = gray_image / 255.0

    # Find very bright and very dark pixels
    overexposed_pixels = np.sum(normalized_image > overexposure_thresh)
    underexposed_pixels = np.sum(normalized_image < underexposure_thresh)
    total_pixels = normalized_image.size

    overexposed_ratio = overexposed_pixels / total_pixels
    underexposed_ratio = underexposed_pixels / total_pixels

    if overexposed_ratio > pixel_ratio_thresh:
        return "overexposed", overexposed_ratio, underexposed_ratio
    elif underexposed_ratio > pixel_ratio_thresh:
        return "underexposed", overexposed_ratio, underexposed_ratio
    else:
        return "normal", overexposed_ratio, underexposed_ratio


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


def check_image_quality(image_path, blur_threshold=100, exposure_threshold=0.05):
    """
    Check the quality of a single image.

    Returns:
        dict: Quality metrics for the image
    """
    image = cv2.imread(image_path)
    if image is None:
        return {
            "filename": os.path.basename(image_path),
            "error": "Failed to load image"
        }

    # Detect blur
    blur_score = detect_blur(image)
    blur_status = "good" if blur_score > blur_threshold else "blurry"

    # Detect exposure
    exposure_status, overexposed_ratio, underexposed_ratio = detect_exposure(
        image, pixel_ratio_thresh=exposure_threshold
    )

    return {
        "filename": os.path.basename(image_path),
        "blur_score": float(blur_score),
        "blur_status": blur_status,
        "exposure_status": exposure_status,
        "overexposed_ratio": float(overexposed_ratio),
        "underexposed_ratio": float(underexposed_ratio)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check image quality (blur and exposure) for all images in a directory"
    )
    parser.add_argument("directory", help="Path to directory containing images")
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100,
        help="Blur threshold (default: 100). Higher = sharper required"
    )
    parser.add_argument(
        "--exposure-threshold",
        type=float,
        default=0.05,
        help="Exposure pixel ratio threshold (default: 0.05). Lower = more sensitive"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional: Save results to JSON file"
    )

    args = parser.parse_args()

    # Load images
    print(f"Scanning directory: {args.directory}")
    image_files = load_images_from_directory(args.directory)

    if not image_files:
        return

    print(f"Found {len(image_files)} images\n")
    print("="*80)

    # Process each image
    results = []
    for img_path in image_files:
        result = check_image_quality(
            img_path,
            blur_threshold=args.blur_threshold,
            exposure_threshold=args.exposure_threshold
        )
        results.append(result)

        # Print results
        print(f"\n{result['filename']}")
        if "error" in result:
            print(f"  ❌ {result['error']}")
        else:
            print(f"  Blur Score: {result['blur_score']:.2f} ({result['blur_status']})")
            print(f"  Exposure: {result['exposure_status'].title()}")
            print(f"    - Overexposed pixels: {result['overexposed_ratio']:.1%}")
            print(f"    - Underexposed pixels: {result['underexposed_ratio']:.1%}")

    print("\n" + "="*80)

    # Summary statistics
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        blurry_count = sum(1 for r in valid_results if r['blur_status'] == 'blurry')
        overexposed_count = sum(1 for r in valid_results if r['exposure_status'] == 'overexposed')
        underexposed_count = sum(1 for r in valid_results if r['exposure_status'] == 'underexposed')

        print(f"\nSummary:")
        print(f"  Total images processed: {len(valid_results)}")
        print(f"  Blurry images: {blurry_count}")
        print(f"  Overexposed images: {overexposed_count}")
        print(f"  Underexposed images: {underexposed_count}")
        print(f"  Good quality images: {len(valid_results) - blurry_count - overexposed_count - underexposed_count}")

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "parameters": {
                    "blur_threshold": args.blur_threshold,
                    "exposure_threshold": args.exposure_threshold
                },
                "results": results
            }, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()
