import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def detect_artifacts(image, border_crop=20, bandwidth=5):
    """
    Detect vertical artifacts in an image using FFT analysis.
    
    Args:
        image (numpy.ndarray): Input image as BGR numpy array
        border_crop (int): Pixels to ignore on each side
        bandwidth (int): Width of vertical band to analyze
        
    Returns:
        float: Confidence score (0-100) indicating likelihood of artifacts
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()
    
    # Preprocessing - crop borders and denoise
    image_gray_cropped = image_gray[border_crop:-border_crop, border_crop:-border_crop]
    image_gray_cropped = cv2.GaussianBlur(image_gray_cropped, (5, 5), 0)
    
    # Compute FFT
    f = np.fft.fft2(image_gray_cropped)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)  # Avoid log(0)
    
    # Analyze vertical frequencies
    rows, cols = magnitude_spectrum.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Take a vertical band around the center column
    vertical_band = magnitude_spectrum[:, center_col-bandwidth:center_col+bandwidth]
    
    # Sum vertical energy
    vertical_strength = np.sum(vertical_band, axis=1)
    
    # Normalize vertical strength
    if vertical_strength.max() > vertical_strength.min():
        vertical_strength_normalized = (vertical_strength - vertical_strength.min()) / (vertical_strength.max() - vertical_strength.min())
    else:
        vertical_strength_normalized = np.zeros_like(vertical_strength)
    
    # Generate confidence score
    confidence_score = vertical_strength_normalized.mean() * 100  # 0–100%
    
    return confidence_score


def create_artifact_heatmap(image, border_crop=20, bandwidth=5):
    """
    Create a heatmap visualization of detected artifacts.
    
    Args:
        image (numpy.ndarray): Input image as BGR numpy array
        border_crop (int): Pixels to ignore on each side
        bandwidth (int): Width of vertical band to analyze
        
    Returns:
        tuple: (heatmap, confidence_score)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()
    
    # Preprocessing
    image_gray_cropped = image_gray[border_crop:-border_crop, border_crop:-border_crop]
    image_gray_cropped = cv2.GaussianBlur(image_gray_cropped, (5, 5), 0)
    
    # Compute FFT
    f = np.fft.fft2(image_gray_cropped)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)
    
    # Analyze vertical frequencies
    rows, cols = magnitude_spectrum.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Take a vertical band around the center column
    vertical_band = magnitude_spectrum[:, center_col-bandwidth:center_col+bandwidth]
    vertical_strength = np.sum(vertical_band, axis=1)
    
    # Normalize vertical strength
    if vertical_strength.max() > vertical_strength.min():
        vertical_strength_normalized = (vertical_strength - vertical_strength.min()) / (vertical_strength.max() - vertical_strength.min())
    else:
        vertical_strength_normalized = np.zeros_like(vertical_strength)
    
    # Create a vertical heatmap
    heatmap = np.tile(vertical_strength_normalized[:, np.newaxis], (1, cols))
    
    # Resize heatmap back to original image size
    heatmap_full = np.zeros_like(image_gray, dtype=np.float32)
    heatmap_full[border_crop:-border_crop, border_crop:-border_crop] = heatmap
    
    # Generate confidence score
    confidence_score = vertical_strength_normalized.mean() * 100
    
    return heatmap_full, confidence_score


def visualize_artifact_detection(image_path, save_path=None):
    """
    Load an image, detect artifacts, and create visualization.
    
    Args:
        image_path (str): Path to input image
        save_path (str, optional): Path to save visualization
        
    Returns:
        float: Confidence score
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Detect artifacts and create heatmap
    heatmap, confidence_score = create_artifact_heatmap(image)
    
    # Compute magnitude spectrum for visualization
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_cropped = image_gray[20:-20, 20:-20]  # Use same crop as detection
    image_gray_cropped = cv2.GaussianBlur(image_gray_cropped, (5, 5), 0)
    
    f = np.fft.fft2(image_gray_cropped)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)
    
    # Create visualization
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
    plt.title(f'Artifact Heatmap (Score: {confidence_score:.2f}%)')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    return confidence_score


def process_image_file(image_path):
    """
    Process a single image file for artifact detection.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: Results containing confidence score and other metrics
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Could not load image: {image_path}"}
        
        confidence_score = detect_artifacts(image)
        
        return {
            "image_path": image_path,
            "confidence_score": confidence_score,
            "artifact_likelihood": "High" if confidence_score > 50 else "Medium" if confidence_score > 25 else "Low"
        }
    except Exception as e:
        return {"error": f"Error processing {image_path}: {str(e)}"}


# Example usage when run as script
if __name__ == "__main__":
    # Example usage - replace with actual image path
    image_path = '../data/green_line.jpg'  # Your cropped artifact region
    
    if os.path.exists(image_path):
        try:
            confidence_score = visualize_artifact_detection(image_path)
            print(f"Artifact Likelihood Score: {confidence_score:.2f}%")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Example image not found: {image_path}")
        print("To use this script, provide a valid image path in the __main__ section")