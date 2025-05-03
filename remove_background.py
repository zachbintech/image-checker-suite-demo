import cv2
import numpy as np

# Load image
img = cv2.imread('template.png')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Create mask of near-white values
lower = np.array([240, 240, 240], dtype=np.uint8)
upper = np.array([255, 255, 255], dtype=np.uint8)
mask = cv2.inRange(img_rgb, lower, upper)

# Invert mask to get the object
mask_inv = cv2.bitwise_not(mask)

# Add alpha channel
b, g, r = cv2.split(img)
alpha = mask_inv
rgba = cv2.merge([b, g, r, alpha])

# Save result
cv2.imwrite('template_with_transparency.png', rgba)
