import cv2
import numpy as np
import os

# Threshold constants
DUST_MIN_AREA = 200  # Minimum contour area for dust detection
DUST_MAX_AREA = 500  # Maximum contour area for dust detection
FFT_LINE_SCORE_THRESHOLD = 0.6  # Threshold for FFT banding detection
SCRATCH_DENSITY_THRESHOLD = 0.6  # Threshold for scratch detection
CHANNEL_ANOMALY_THRESHOLD = 0.9  # Threshold for channel anomaly detection

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

    # # 1. Dust detection
    # dust_mask = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                   cv2.THRESH_BINARY_INV, 15, 10)
    # dust_mask = cv2.morphologyEx(dust_mask, cv2.MORPH_OPEN,
    #                               np.ones((3, 3), np.uint8), iterations=1)
    # contours, _ = cv2.findContours(dust_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # dust_contours = [cnt for cnt in contours if DUST_MIN_AREA < cv2.contourArea(cnt) < DUST_MAX_AREA]
    # if dust_contours:
    #     issues_detected = True
    #     for cnt in dust_contours:
    #         cv2.drawContours(vis, [cnt], -1, (0, 0, 255), 1)

    # 2. FFT banding detection
    fft_issue = detect_fft_artifacts(gray, debug=debug)
    if fft_issue:
        issues_detected = True
        cv2.putText(vis, "FFT Banding", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2)

    # 3. Scanner banding: edge + line orientation filtering
    # sobelx = cv2.Sobel(norm, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(norm, cv2.CV_64F, 0, 1, ksize=3)
    # edge_mag = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)

    # lines = cv2.HoughLinesP(edge_mag, 1, np.pi/180, threshold=500,
    #                         minLineLength=100, maxLineGap=5)
    
    # filtered_lines = filter_lines_by_angle(lines, angle_range=(88, 92))  # vertical lines TODO should also check for horizontal lines
    # if filtered_lines:
    #     issues_detected = True
    #     for line in filtered_lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
    # End of scanner banding detection

    # # 4. Channel anomalies
    # channel_issue = detect_channel_artifacts(image, debug=debug)
    # if channel_issue:
    #     issues_detected = True
    #     cv2.putText(vis, "Channel Artifact", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                 (255, 0, 255), 2)

    # # 5. Scratches / tear edges (basic)
    # edges = cv2.Canny(norm, 50, 150)
    # scratch_density = np.mean(edges) / 255
    # if scratch_density > SCRATCH_DENSITY_THRESHOLD:
    #     issues_detected = True
    #     vis[edges > 0] = [255, 0, 255]

    if debug:
        cv2.imshow("Detected Artifacts", cv2.resize(vis, (800, int(800 * image.shape[0] / image.shape[1]))))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
