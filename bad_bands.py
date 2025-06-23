import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_bad_bands(image_path, debug=False):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)

    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('FFT Magnitude')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{image_path}_fft_output.png")  # Save instead of show


# detect_bad_bands("real_images/Badger Canyon 1989.jpg", debug=True)
# detect_bad_bands("bad_bands.png", debug=True)
# detect_bad_bands("bad_bands_og.jpg", debug=True)

from tqdm import tqdm

def extract_fft_vertical_score(patch, band_width=4, low_cut=20, high_cut=200):
    # Convert to float and apply FFT
    f = np.fft.fft2(patch)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)

    h, w = patch.shape
    center_x = w // 2
    center_y = h // 2

    # Suppress low frequencies
    mask = np.zeros_like(mag)
    for y in range(h):
        for x in range(w):
            dist = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            if low_cut < dist < high_cut:
                mask[y, x] = 1

    mag *= mask  # apply band-pass mask

    # Get vertical slice near center
    vertical_band = mag[:, center_x - band_width // 2:center_x + band_width // 2 + 1]
    vertical_energy = np.sum(vertical_band)
    total_energy = np.sum(mag) + 1e-8  # prevent div-by-zero

    return vertical_energy / total_energy  # score between 0 and 1


def sliding_fft_artifact_map(image, patch_size=128, stride=64, threshold=None):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape
    heatmap = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            score = extract_fft_vertical_score(patch)
            heatmap[y:y+patch_size, x:x+patch_size] += score

    # Normalize heatmap to 0–1
    heatmap /= np.max(heatmap) + 1e-8

    if threshold is not None:
        binary_mask = (heatmap > threshold).astype(np.uint8) * 255
        return heatmap, binary_mask
    return heatmap, None


def generate_artifact_map(image_path, patch_size=128, stride=64, threshold=None):
    # Create a temporary directory if it doesn't exist
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    heatmap, mask = sliding_fft_artifact_map(img, patch_size=patch_size, stride=stride, threshold=threshold)

    # Save the heatmap and mask in the tmp directory
    heatmap_path = os.path.join(tmp_dir, f"artifact_heatmap_{os.path.basename(image_path)}")
    cv2.imwrite(heatmap_path, (heatmap * 255).astype(np.uint8))
    if mask is not None:
        mask_path = os.path.join(tmp_dir, f"artifact_mask_{os.path.basename(image_path)}")
        cv2.imwrite(mask_path, mask)

    # Combine original image and heatmap horizontally
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    combined = cv2.hconcat([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), heatmap_color])

    # Save the combined image in the tmp directory
    combined_path = os.path.join(tmp_dir, f"combined_artifact_{os.path.basename(image_path)}")
    cv2.imwrite(combined_path, combined)
    print(f"Saved combined image: {combined_path}")

generate_artifact_map("bad_bands.png", patch_size=128, stride=64, threshold=0.3)
generate_artifact_map("real_images/Badger Canyon 1989.jpg", patch_size=128, stride=64, threshold=0.3)

def process_directory(directory_path, patch_size=128, stride=64, threshold=0.3):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                print(f"Processing: {image_path}")
                generate_artifact_map(image_path, patch_size, stride, threshold)

# Example usage:
process_directory("real_images", patch_size=128, stride=64, threshold=0.3)

def save_heatmap_legend(output_path="heatmap_legend.png"):
    # Create a dummy heatmap for the legend
    heatmap = np.linspace(0, 1, 256).reshape(1, -1)
    heatmap = np.vstack([heatmap] * 20)  # Repeat to make it visible

    # Save the heatmap legend
    plt.figure(figsize=(6, 1))
    plt.imshow(heatmap, aspect="auto", cmap="jet")
    plt.gca().set_visible(False)
    cbar = plt.colorbar(orientation="horizontal", pad=0.2)
    cbar.set_label("Artifact Intensity (0 = Low, 1 = High)")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap legend: {output_path}")

# Example usage:
save_heatmap_legend()
