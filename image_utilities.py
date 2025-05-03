import cv2
from PIL import Image
import numpy as np


# Placeholder: photo detection & cropping (mock)
def dummy_crop(image):
    h, w = image.shape[:2]
    return [image[0:h//2, 0:w//2], image[h//2:, w//2:]]  # Fake 2 crops

# Utility: Convert to OpenCV format
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)