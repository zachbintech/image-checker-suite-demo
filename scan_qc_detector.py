# scan_qc_detector.py
import cv2
import numpy as np
import os
from pathlib import Path
import csv


def has_large_border(image_path, border_threshold=0.1):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    edges = cv2.Canny(img, 100, 200)
    h, w = img.shape

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    border_area = (w * h) - (cw * ch)
    border_ratio = border_area / (w * h)

    return border_ratio > border_threshold


def is_blurry(image_path, blur_threshold=100.0):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < blur_threshold


def detect_multiple_photos(image_path, area_threshold=5000, border_ratio_threshold=0.90, min_aspect=0.5, max_aspect=2.0):
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    h, w = img.shape[:2]
    image_area = h * w

    filtered_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < area_threshold or area / image_area > border_ratio_threshold:
            continue  # Skip small or overly large areas

        x, y, cw, ch = cv2.boundingRect(c)
        aspect_ratio = cw / ch if ch != 0 else 0

        # Ignore very skinny or weird shapes
        if not (min_aspect <= aspect_ratio <= max_aspect):
            continue

        # Optional: Rectangularity check
        rect_area = cw * ch
        solidity = area / rect_area if rect_area != 0 else 0
        if solidity < 0.7:
            continue  # very irregular shape

        filtered_contours.append(c)

    return len(filtered_contours) > 1




def save_qc_overlay(image_path, output_dir, area_threshold=5000):
    img = cv2.imread(str(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    photo_like_contours = [c for c in contours if cv2.contourArea(c) > area_threshold]

    for cnt in photo_like_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    output_path = Path(output_dir) / Path(image_path).name
    cv2.imwrite(str(output_path), img)



def analyze_folder(input_dir, output_csv):
    image_paths = list(Path(input_dir).rglob("*.jpg")) + list(Path(input_dir).rglob("*.png"))

    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "has_large_border", "is_blurry", "contains_multiple_photos"])

        for path in image_paths:
            border = has_large_border(path)
            blurry = is_blurry(path)
            multi = detect_multiple_photos(path)
            writer.writerow([path.name, border, blurry, multi])

            # Optionally generate QC overlays
            qc_output_dir = Path("qc_overlays")
            qc_output_dir.mkdir(exist_ok=True)

            if multi:  # Only if multiple photos detected
                save_qc_overlay(path, qc_output_dir)

    print(f"Analysis complete. Results saved to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detect artifacts and multi-photo scans in an image folder.")
    parser.add_argument("input_dir", help="Path to input folder of scanned images")
    parser.add_argument("output_csv", help="Path to save CSV report")

    args = parser.parse_args()

    analyze_folder(args.input_dir, args.output_csv)
