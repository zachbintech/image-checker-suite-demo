import os
import shutil
from PIL import Image, ImageDraw
import numpy as np

def detect_subtle_defects(img_array, std_threshold=10, green_bias=20):
    """
    Detect very subtle vertical and horizontal lines where color varies very little and green is dominant.
    """
    height, width, _ = img_array.shape
    problems = []
    defect_columns = []
    defect_rows = []

    # Check each column
    for col in range(width):
        column = img_array[:, col, :]
        std = np.std(column, axis=0)
        mean = np.mean(column, axis=0)

        # If color doesn't vary much vertically
        if np.mean(std) < std_threshold:
            # Check if green is noticeably higher than red and blue
            if mean[1] > mean[0] + green_bias and mean[1] > mean[2] + green_bias:
                problems.append('green')
                defect_columns.append(col)
            elif mean[2] > mean[0] + green_bias and mean[2] > mean[1] + green_bias:
                problems.append('blue')
                defect_columns.append(col)
            elif mean[0] > mean[1] + green_bias and mean[0] > mean[2] + green_bias:
                problems.append('red')
                defect_columns.append(col)

    # Check each row
    for row in range(height):
        line = img_array[row, :, :]
        std = np.std(line, axis=0)
        mean = np.mean(line, axis=0)

        if np.mean(std) < std_threshold:
            if mean[1] > mean[0] + green_bias and mean[1] > mean[2] + green_bias:
                problems.append('green')
                defect_rows.append(row)
            elif mean[2] > mean[0] + green_bias and mean[2] > mean[1] + green_bias:
                problems.append('blue')
                defect_rows.append(row)
            elif mean[0] > mean[1] + green_bias and mean[0] > mean[2] + green_bias:
                problems.append('red')
                defect_rows.append(row)

    return list(set(problems)), defect_columns, defect_rows

def mark_defect_lines(img, defect_columns, defect_rows, highlight_color=(255, 255, 0)):
    """
    Mark defective columns and rows on a copy of the image.
    """
    img_marked = img.copy()
    draw = ImageDraw.Draw(img_marked)
    width, height = img.size

    for col in defect_columns:
        draw.line([(col, 0), (col, height)], fill=highlight_color, width=1)

    for row in defect_rows:
        draw.line([(0, row), (width, row)], fill=highlight_color, width=1)

    return img_marked

def process_directory(source_dir, problem_dir, marked_dir=None):
    """
    Process all images in a directory, copying problem images if defective lines found.
    """
    os.makedirs(problem_dir, exist_ok=True)
    if marked_dir:
        os.makedirs(marked_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(source_dir, filename)
            try:
                img = Image.open(file_path).convert('RGB')
                img_array = np.array(img)

                detected_colors, defect_columns, defect_rows = detect_subtle_defects(img_array)

                if detected_colors:
                    print(f"{filename}: Problem detected ({', '.join(detected_colors)})")
                    shutil.copy(file_path, os.path.join(problem_dir, filename))

                    if marked_dir:
                        highlight_color = {
                            'green': (0, 255, 0),
                            'blue': (0, 0, 255),
                            'red': (255, 0, 0)
                        }.get(detected_colors[0], (255, 255, 0))

                        img_marked = mark_defect_lines(img, defect_columns, defect_rows, highlight_color=highlight_color)
                        img_marked.save(os.path.join(marked_dir, filename))
                else:
                    print(f"{filename}: No problem detected.")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")


# === HOW TO USE ===
source_dir = "../data/"
problem_dir = "problem/defect_images"
marked_dir = "marked_images/"  # Optional; can be None if you don't want marked copies

process_directory(source_dir, problem_dir, marked_dir)














