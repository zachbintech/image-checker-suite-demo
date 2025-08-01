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


def extract_fft_horizontal_score(patch, band_width=4, low_cut=20, high_cut=200):
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

    # Get horizontal slice near center
    horizontal_band = mag[center_y - band_width // 2:center_y + band_width // 2 + 1, :]
    horizontal_energy = np.sum(horizontal_band)
    total_energy = np.sum(mag) + 1e-8  # prevent div-by-zero

    return horizontal_energy / total_energy  # score between 0 and 1


def sliding_fft_artifact_map(image, patch_size=128, stride=64, threshold=None):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape
    vertical_heatmap = np.zeros((h, w), dtype=np.float32)
    horizontal_heatmap = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            vertical_score = extract_fft_vertical_score(patch)
            horizontal_score = extract_fft_horizontal_score(patch)
            vertical_heatmap[y:y+patch_size, x:x+patch_size] += vertical_score
            horizontal_heatmap[y:y+patch_size, x:x+patch_size] += horizontal_score

    # Normalize heatmaps to 0–1
    vertical_heatmap /= np.max(vertical_heatmap) + 1e-8
    horizontal_heatmap /= np.max(horizontal_heatmap) + 1e-8

    # Combine vertical and horizontal heatmaps
    combined_heatmap = np.maximum(vertical_heatmap, horizontal_heatmap)

    if threshold is not None:
        binary_mask = (combined_heatmap > threshold).astype(np.uint8) * 255
        return combined_heatmap, binary_mask
    return combined_heatmap, None


def generate_artifact_map(image_path, patch_size=128, stride=64, threshold=None):
    # Create necessary directories
    tmp_dir = "tmp"
    review_dir = "review"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(review_dir, exist_ok=True)

    # Read the image in color
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        print(f"Warning: Could not read image {image_path}. Skipping.")
        return

    # Extract the red channel (index 2 in BGR)
    red_channel = img_color[:, :, 2]

    # Create a blank 3-channel image where only red is filled
    red_vis = np.zeros_like(img_color)
    red_vis[:, :, 2] = red_channel  # OpenCV uses BGR, so channel 2 is red

    # Read the image in color

    # Split channels

    plot_histograms(image_path)

    img = cv2.imread(image_path).astype(np.float32)
    red_mask = get_red_mask(image_path, img)  
    save_artifact_mask(image_path, red_mask)

    # Vertically stack the color image and the red mask
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    red_mask_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    stacked = cv2.vconcat([img_color, red_mask_bgr])
    os.makedirs("masks", exist_ok=True)
    stacked_path = os.path.join("masks", f"color_and_redmask_{os.path.basename(image_path)}")
    cv2.imwrite(stacked_path, stacked)

    detect_scanner_artifacts(red_mask, show_debug=False)

def save_artifact_mask(image_path, red_mask):
    mask_output_dir = os.path.join("tmp", "masks")
    os.makedirs(mask_output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(mask_output_dir, f"red_artifacts_mask_{os.path.basename(image_path)}.png"), red_mask)


def detect_scanner_artifacts(binary_img: np.ndarray,
                              show_debug: bool = True,
                              component_thresh: dict = None,
                              stripe_thresh: dict = None):
    """
    Detects two types of artifacts in a binary image:
    1. Clustered vertical banding (connected component analysis).
    2. Vertical stripe artifacts with horizontal ripple (sliding window).

    Args:
        binary_img (np.ndarray): Binary (thresholded) image, 8-bit.
        show_debug (bool): Whether to show a visual output.
        component_thresh (dict): Tuning thresholds for component filtering.
        stripe_thresh (dict): Tuning thresholds for stripe detection.

    Returns:
        dict: Contains detected regions and types.
    """
    if component_thresh is None:
        component_thresh = {
            "min_aspect_ratio": 2.0,
            "min_density": 0.3,
            "max_height": 100,
        }

    if stripe_thresh is None:
        stripe_thresh = {
            "window_width": 20,
            "step_size": 5,
            "min_variation": 20,
            "min_density": 5,
        }

    artifacts = {
        "component_clusters": [],
        "vertical_stripes": []
    }

    h, w = binary_img.shape
    binary = binary_img.copy()

    # --- Step 1: Connected Component Analysis ---
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    for i in range(1, num_labels):  # skip background
        x, y, bw, bh, area = stats[i]
        aspect = bw / bh if bh != 0 else 0
        density = area / (bw * bh) if bw * bh != 0 else 0

        if (aspect > component_thresh["min_aspect_ratio"] and
            density > component_thresh["min_density"] and
            bh < component_thresh["max_height"]):
            artifacts["component_clusters"].append((x, y, bw, bh))

    # --- Step 2: Vertical Stripe Analysis ---
    win_w = stripe_thresh["window_width"]
    step = stripe_thresh["step_size"]
    for x in range(0, w - win_w, step):
        window = binary[:, x:x + win_w]
        row_sums = np.sum(window == 255, axis=1)
        row_std = np.std(row_sums)
        density = np.mean(row_sums) / win_w

        if row_std > stripe_thresh["min_variation"] and density > stripe_thresh["min_density"]:
            artifacts["vertical_stripes"].append((x, 0, win_w, h))

    # --- Debug visualization ---
    if show_debug:
        output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        for (x, y, bw, bh) in artifacts["component_clusters"]:
            cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        for (x, y, bw, bh) in artifacts["vertical_stripes"]:
            cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        cv2.imshow("Detected Artifacts", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return artifacts

  

def get_red_mask(image_path, img):
    b, g, r = cv2.split(img)

    # Avoid divide by zero
    epsilon = 1e-6
    r_ratio = r / (g + b + epsilon)

    print(os.path.basename(image_path))

    # Threshold for "too red"
    red_mask = (r_ratio > 1.5).astype(np.uint8) * 255 # adjust 1.5 to be more/less sensitive
    return red_mask
    

def plot_histograms(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    blue, green, red = cv2.split(img)

    plt.figure(figsize=(10, 5))
    plt.title("Color Channel Histograms")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Pixel Count")

    plt.hist(blue.ravel(), bins=256, range=[0,256], color='blue', alpha=0.5, label='Blue')
    plt.hist(green.ravel(), bins=256, range=[0,256], color='green', alpha=0.5, label='Green')
    plt.hist(red.ravel(), bins=256, range=[0,256], color='red', alpha=0.5, label='Red')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("histograms", exist_ok=True)  # <-- Add this line
    plt.savefig(f"histograms/channel_histogram_{image_path[-6:]}.png")
    print("Saved histogram to channel_histogram.png")


    

def extract_all_edge_regions(edge_width=25, heatmap=None):
    top_edge = heatmap[:edge_width, :]
    bottom_edge = heatmap[-edge_width:, :]
    left_edge = heatmap[:, :edge_width]
    right_edge = heatmap[:, -edge_width:]

def review_needed_ON_IMAGE(image_path):
    # Placeholder for the actual condition to check if review is needed
    # This should return True if the image has significant artifacts
    # For now, we will assume it always returns False
    return False

def save_for_review_if_over_thresh(image_path, review_dir ="review"):
    if (review_needed_ON_IMAGE(image_path)):        
        review_path = os.path.join(review_dir, f"review_{os.path.basename(image_path)}")
        cv2.imwrite(review_path, combined)
        print(f"Saved to review folder: {review_path}")

# Example usage:
# generate_artifact_map("bad_bands.png", patch_size=128, stride=64, threshold=0.3)
# generate_artifact_map("real_images/Badger Canyon 1989.jpg", patch_size=128, stride=64, threshold=0.3)

def process_directory(directory_path, patch_size=128, stride=64, threshold=0.3):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                print(f"Processing: {image_path}")
                generate_artifact_map(image_path, patch_size, stride, threshold)

# Example usage:
process_directory("/home/zach/Desktop/PhotoData", patch_size=128, stride=64, threshold=0.3)





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