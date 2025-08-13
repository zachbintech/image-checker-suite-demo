import cv2
import numpy as np
import os

# Threshold constants
DUST_MIN_AREA = 200  # Minimum contour area for dust detection
DUST_MAX_AREA = 500  # Maximum contour area for dust detection
FFT_LINE_SCORE_THRESHOLD = 0.04  # Threshold for FFT banding detection
SCRATCH_DENSITY_THRESHOLD = 0.6  # Threshold for scratch detection
CHANNEL_ANOMALY_THRESHOLD = 0.9  # Threshold for channel anomaly detection
COLOR_STREAK_THRESHOLD = 30  # Adjust as needed for channel diff threshold
WIDE_BAND_THRESHOLD = 25  # Threshold for wide color banding detection


def detect_wide_color_banding(image, debug=False):
    h, w, _ = image.shape
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)

    # Normalize each channel to account for global exposure
    b, g, r = cv2.split(image_blur)
    b_mean = np.mean(b, axis=0)
    g_mean = np.mean(g, axis=0)
    r_mean = np.mean(r, axis=0)

    # Stack as (w, 3)
    col_means = np.stack([b_mean, g_mean, r_mean], axis=-1)

    # Smooth with wider kernel to find slow trends
    col_means_smooth = cv2.blur(col_means.astype(np.float32), (25, 1))  # horizontal blur

    # Difference from smoothed baseline
    diff = np.abs(col_means - col_means_smooth)

    # Threshold: any column with > threshold diff in R, G, or B
    band_mask = np.any(diff > WIDE_BAND_THRESHOLD, axis=-1)  # shape: (w,)
    print("DIFF SHAPE:", diff.shape, "BAND MASK SHAPE:", band_mask.shape)
    print("DIFF: ", diff)

    if debug:
        band_vis = image.copy()
        for x in range(w):
            if band_mask[x]:
                cv2.line(band_vis, (x, 0), (x, h-1), (0, 0, 255), 1)
        cv2.imshow("Wide Band Detection", band_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # If 10+ vertical lines deviate, we assume a band exists
    return np.count_nonzero(band_mask) > WIDE_BAND_THRESHOLD


def detect_fft_artifacts(gray, debug=False):
    rows, cols = gray.shape
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

    # Suppress low frequencies around center
    mask = np.ones_like(fshift, dtype=np.uint8)
    center_row, center_col = rows // 2, cols // 2
    band_size = 10
    mask[center_row - band_size:center_row + band_size,
         center_col - band_size:center_col + band_size] = 0

    fshift_filtered = fshift * mask
    img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
    img_back = np.abs(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges = cv2.Canny(img_back, 50, 150)
    line_score = np.mean(edges) / 255



    if debug:
        cv2.imshow("FFT Suppressed Back Projection", img_back)
        cv2.imshow("FFT Band Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"FFT Line Score: {line_score:.2f}")

    return line_score > FFT_LINE_SCORE_THRESHOLD


def filter_lines_by_angle(lines, angle_range=(80, 100)):
    filtered = []
    if lines is None:
        return filtered
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angle = abs(angle)
        if angle_range[0] <= angle <= angle_range[1]:
            filtered.append(line)
    return filtered


def detect_channel_artifacts(image, debug=False):
    b, g, r = cv2.split(image)
    diff_bg = cv2.absdiff(b, g)
    diff_gr = cv2.absdiff(g, r)
    diff_rb = cv2.absdiff(r, b)
    anomaly_mask = ((diff_bg > 60) | (diff_gr > 60) | (diff_rb > 60))

    anomaly_score = np.mean(anomaly_mask.astype(np.uint8))

    if debug:
        vis = image.copy()
        vis[anomaly_mask] = [255, 0, 255]
        cv2.imshow("Channel Anomalies", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return anomaly_score > CHANNEL_ANOMALY_THRESHOLD



def detect_colored_vertical_streaks(image, debug=False):
    h, w, _ = image.shape
    streak_mask = np.zeros((h, w), dtype=np.uint8)

    b, g, r = cv2.split(image)

    for channel, name in zip([b, g, r], ['blue', 'green', 'red']):
        diff = cv2.absdiff(channel, cv2.medianBlur(channel, 5))
        _, binary = cv2.threshold(diff, COLOR_STREAK_THRESHOLD, 255, cv2.THRESH_BINARY)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

        streak_mask = cv2.bitwise_or(streak_mask, vertical_lines)

        # Visualization on top of the original image
        if debug:
            vis = image.copy()
            vis[vertical_lines > 0] = [0, 255, 255]  # Highlight streaks in yellow
            cv2.imshow(f"{name} channel streaks", vis)
            cv2.waitKey(0)

    return np.count_nonzero(streak_mask) > 0


def detect_dust_or_artifacts(image_path, debug=False, save_vis=False, output_dir=None):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"⚠️ Could not read: {image_path}")
        return False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

    issues_detected = False
    vis = image.copy()

    

        # 7. Faint wide vertical banding
    wide_band_issue = detect_wide_color_banding(image, debug=debug)
    if wide_band_issue:
        issues_detected = True
        cv2.putText(vis, "Wide Color Band", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 2)


   

    if save_vis and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, vis)

        # Create a side-by-side comparison
        comparison = np.hstack((image, vis))
        comparison_path = os.path.join(output_dir, f"comparison_{os.path.basename(image_path)}")
        cv2.imwrite(comparison_path, comparison)

    return issues_detected


def get_all_image_paths_recursive(root_dir):
    image_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(dirpath, f))
    return image_paths


if __name__ == "__main__":
    INPUT_DIR = "/home/zach/Desktop/PhotoData"
    OUTPUT_DIR = "artifact_overlays"

    image_paths = get_all_image_paths_recursive(INPUT_DIR)

    print(f"🔍 Scanning {len(image_paths)} images...\n")
    for path in image_paths:
        has_issues = detect_dust_or_artifacts(path, debug=True, save_vis=True, output_dir=OUTPUT_DIR)
        status = "❌ Issue Detected" if has_issues else "✅ Clean"
        print(f"{status}: {path}")
