import numpy as np
import cv2
import matplotlib.pyplot as plt

# --- 1. Load image ---
image_path = '../data/green_line.jpg'  # Your cropped artifact region
image = cv2.imread(image_path)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- 2. Preprocessing ---
border_crop = 20  # Ignore 20px on each side
image_gray_cropped = image_gray[border_crop:-border_crop, border_crop:-border_crop]

# Light denoising
image_gray_cropped = cv2.GaussianBlur(image_gray_cropped, (5, 5), 0)

# --- 3. Compute FFT ---
f = np.fft.fft2(image_gray_cropped)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)  # Avoid log(0)

# --- 4. Analyze vertical frequencies ---
rows, cols = magnitude_spectrum.shape
center_row, center_col = rows // 2, cols // 2

# Take a vertical band around the center column
bandwidth = 5  # How many pixels wide to check
vertical_band = magnitude_spectrum[:, center_col-bandwidth:center_col+bandwidth]

# Sum vertical energy
vertical_strength = np.sum(vertical_band, axis=1)

# Normalize vertical strength
vertical_strength_normalized = (vertical_strength - vertical_strength.min()) / (vertical_strength.max() - vertical_strength.min())

# Create a vertical heatmap
heatmap = np.tile(vertical_strength_normalized[:, np.newaxis], (1, cols))

# --- 5. Generate Confidence Score ---
confidence_score = vertical_strength_normalized.mean() * 100  # 0–100%

print(f"Artifact Likelihood Score: {confidence_score:.2f}%")

# --- 6. Visualization ---

# Resize heatmap back to original image size
heatmap_full = np.zeros_like(image_gray, dtype=np.float32)
heatmap_full[border_crop:-border_crop, border_crop:-border_crop] = heatmap

# Display everything
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('FFT Magnitude Spectrum')
plt.imshow(magnitude_spectrum, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Vertical Artifact Heatmap')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.imshow(heatmap_full, cmap='jet', alpha=0.5)  # Overlay heatmap
plt.axis('off')

plt.tight_layout()
plt.show()
