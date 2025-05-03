import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_defect_scores(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    height, width, _ = img_array.shape

    column_scores = []
    row_scores = []

    # Calculate flatness scores (stddev) for columns
    for col in range(width):
        column = img_array[:, col, :]
        std = np.std(column, axis=0)
        column_scores.append(np.mean(std))

    # Calculate flatness scores for rows
    for row in range(height):
        line = img_array[row, :, :]
        std = np.std(line, axis=0)
        row_scores.append(np.mean(std))

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    ax1.plot(column_scores)
    ax1.set_title('Column Flatness Scores')
    ax1.set_xlabel('Column Index')
    ax1.set_ylabel('Mean Std Dev')

    ax2.plot(row_scores)
    ax2.set_title('Row Flatness Scores')
    ax2.set_xlabel('Row Index')
    ax2.set_ylabel('Mean Std Dev')

    plt.tight_layout()
    plt.show()

# === HOW TO USE ===
visualize_defect_scores('../data/green_line.jpg')
