import unittest
import numpy as np
import cv2
import os
import sys

# Import artifact detection functions from app.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import get_red_mask, get_green_mask, get_blue_mask, detect_red_lines, detect_green_lines, detect_blue_lines


class TestArtifacts(unittest.TestCase):

    def setUp(self):
        """Create test images with color artifacts"""
        # Pure black image (no artifacts)
        self.black_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Image with red line artifact
        self.red_artifact_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.red_artifact_image[50, :] = [0, 0, 255]  # Red line (BGR format)

        # Image with green line artifact
        self.green_artifact_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.green_artifact_image[50, :] = [0, 255, 0]  # Green line

        # Image with blue line artifact
        self.blue_artifact_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.blue_artifact_image[50, :] = [255, 0, 0]  # Blue line

    def test_red_mask_detection(self):
        """Test red artifact mask generation"""
        red_mask = get_red_mask(self.red_artifact_image, threshold=1.5)

        self.assertIsInstance(red_mask, np.ndarray)
        self.assertEqual(red_mask.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(red_mask), 0, "Should detect red artifacts")

    def test_green_mask_detection(self):
        """Test green artifact mask generation"""
        green_mask = get_green_mask(self.green_artifact_image, threshold=1.5)

        self.assertIsInstance(green_mask, np.ndarray)
        self.assertEqual(green_mask.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(green_mask), 0, "Should detect green artifacts")

    def test_blue_mask_detection(self):
        """Test blue artifact mask generation"""
        blue_mask = get_blue_mask(self.blue_artifact_image, threshold=1.5)

        self.assertIsInstance(blue_mask, np.ndarray)
        self.assertEqual(blue_mask.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(blue_mask), 0, "Should detect blue artifacts")

    def test_no_red_artifacts_in_black_image(self):
        """Test that black image has no red artifacts"""
        red_mask = get_red_mask(self.black_image, threshold=1.5)
        self.assertEqual(np.count_nonzero(red_mask), 0, "Black image should have no red artifacts")

    def test_detect_red_lines_returns_tuple(self):
        """Test that detect_red_lines returns mask and count"""
        mask, count = detect_red_lines(self.red_artifact_image, threshold=1.5)

        self.assertIsInstance(mask, np.ndarray)
        self.assertIsInstance(count, (int, np.integer))
        self.assertGreater(count, 0)

    def test_detect_green_lines_returns_tuple(self):
        """Test that detect_green_lines returns mask and count"""
        mask, count = detect_green_lines(self.green_artifact_image, threshold=1.5)

        self.assertIsInstance(mask, np.ndarray)
        self.assertIsInstance(count, (int, np.integer))
        self.assertGreater(count, 0)

    def test_detect_blue_lines_returns_tuple(self):
        """Test that detect_blue_lines returns mask and count"""
        mask, count = detect_blue_lines(self.blue_artifact_image, threshold=1.5)

        self.assertIsInstance(mask, np.ndarray)
        self.assertIsInstance(count, (int, np.integer))
        self.assertGreater(count, 0)

    def test_threshold_sensitivity(self):
        """Test that higher thresholds detect fewer artifacts"""
        low_threshold_mask = get_red_mask(self.red_artifact_image, threshold=1.0)
        high_threshold_mask = get_red_mask(self.red_artifact_image, threshold=3.0)

        low_count = np.count_nonzero(low_threshold_mask)
        high_count = np.count_nonzero(high_threshold_mask)

        # Lower threshold should detect at least as many artifacts as higher threshold
        self.assertGreaterEqual(low_count, high_count)

    def test_bad_bands_reference_image_loading(self):
        """Test that the bad_bands.png reference image can be loaded if it exists"""
        bad_bands_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'bad_bands.png')
        if os.path.exists(bad_bands_path):
            img = cv2.imread(bad_bands_path)
            self.assertIsNotNone(img, "Should be able to load bad_bands.png")
            self.assertGreater(img.shape[0], 0, "Image should have height > 0")
            self.assertGreater(img.shape[1], 0, "Image should have width > 0")
            self.assertEqual(len(img.shape), 3, "Image should be 3D (color)")
        else:
            self.skipTest("bad_bands.png reference image not found")


if __name__ == "__main__":
    unittest.main()