import os
import shutil
import numpy as np
from PIL import Image

INPUT_DIR = '../data/oldphotos'
PROBLEM_DIR = 'problem_images'
STD_THRESHOLD = 15  # tweak this if needed
EDGE_SCAN_WIDTH = 50  # only scan first 50 cols/rows

os.makedirs(PROBLEM_DIR, exist_ok=True)

def detect_defect(img_path):
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)

    height, width, _ = img_array.shape

    column_stddevs = []
    row_stddevs = []

    for col in range(width):
        col_pixels = img_array[:, col, :]
        std = np.std(col_pixels, axis=0)
        column_stddevs.append(np.mean(std))

    for row in range(height):
        row_pixels = img_array[row, :, :]
        std = np.std(row_pixels, axis=0)
        row_stddevs.append(np.mean(std))

    # Only look at leftmost and topmost part
    suspicious_columns = column_stddevs[:EDGE_SCAN_WIDTH]
    suspicious_rows = row_stddevs[:EDGE_SCAN_WIDTH]

    # Look for any very flat columns/rows
    if any(std < STD_THRESHOLD for std in suspicious_columns) or \
       any(std < STD_THRESHOLD for std in suspicious_rows):
        return True
    else:
        return False

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    file_path = os.path.join(INPUT_DIR, filename)
    if detect_defect(file_path):
        print(f"Problem detected: {filename}")
        shutil.copy(file_path, os.path.join(PROBLEM_DIR, filename))
    else:
        print(f"No problem: {filename}")
