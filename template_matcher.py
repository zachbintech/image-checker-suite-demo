import cv2
import numpy as np
import matplotlib.pyplot as plt


# --- 1. Load image ---
image_path = '../data/green_line.jpg'  # Your cropped artifact region
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


template = cv2.imread('Screenshot 2025-04-28 at 8.19.52 AM.png', 0)  # Template to search for
result = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.6  # Confidence threshold for match
locations = np.where(result >= threshold)

for pt in zip(*locations[::-1]):
    cv2.rectangle(image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
