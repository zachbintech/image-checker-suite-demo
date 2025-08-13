import unittest
import numpy as np
import cv2
from image_quality_checks.basic_image_quality import detect_blur


class TestImageQuality(unittest.TestCase):
    
    def setUp(self):
        """Create test images with different blur levels"""
        # Sharp image with high contrast edges
        self.sharp_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.sharp_image, (25, 25), (75, 75), (255, 255, 255), -1)
        cv2.rectangle(self.sharp_image, (40, 40), (60, 60), (0, 0, 0), -1)
        
        # Blurry version of the same image
        self.blurry_image = cv2.GaussianBlur(self.sharp_image, (15, 15), 0)
        
        # Extremely blurry image
        self.very_blurry_image = cv2.GaussianBlur(self.sharp_image, (31, 31), 0)
        
        # Uniform image (minimal variance)
        self.uniform_image = np.full((100, 100, 3), 128, dtype=np.uint8)
    
    def test_detect_blur_sharp_image(self):
        """Test that sharp images return high blur scores"""
        blur_score = detect_blur(self.sharp_image)
        self.assertGreater(blur_score, 100, "Sharp image should have high blur score")
    
    def test_detect_blur_blurry_image(self):
        """Test that blurry images return lower blur scores than sharp images"""
        sharp_score = detect_blur(self.sharp_image)
        blurry_score = detect_blur(self.blurry_image)
        self.assertGreater(sharp_score, blurry_score, "Sharp image should have higher score than blurry")
    
    def test_detect_blur_very_blurry_image(self):
        """Test that very blurry images return very low blur scores"""
        very_blurry_score = detect_blur(self.very_blurry_image)
        self.assertLess(very_blurry_score, 50, "Very blurry image should have very low score")
    
    def test_detect_blur_uniform_image(self):
        """Test that uniform images return very low blur scores"""
        uniform_score = detect_blur(self.uniform_image)
        self.assertLess(uniform_score, 10, "Uniform image should have very low score")
    
    def test_detect_blur_relative_ordering(self):
        """Test that blur scores are ordered correctly relative to actual blur levels"""
        sharp_score = detect_blur(self.sharp_image)
        blurry_score = detect_blur(self.blurry_image)
        very_blurry_score = detect_blur(self.very_blurry_image)
        uniform_score = detect_blur(self.uniform_image)
        
        # Verify ordering: sharp > blurry > very_blurry > uniform
        self.assertGreater(sharp_score, blurry_score)
        self.assertGreater(blurry_score, very_blurry_score)
        self.assertGreater(very_blurry_score, uniform_score)
    
    def test_detect_blur_returns_float(self):
        """Test that detect_blur returns a numeric value"""
        blur_score = detect_blur(self.sharp_image)
        self.assertIsInstance(blur_score, (int, float))
        self.assertGreaterEqual(blur_score, 0)


if __name__ == "__main__":
    unittest.main()