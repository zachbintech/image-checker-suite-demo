import unittest
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
from io import BytesIO
from image_utilities import load_image, dummy_crop


class TestImageUtilities(unittest.TestCase):
    
    def setUp(self):
        """Create test images and files"""
        # Create a test image array
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (25, 25), (75, 75), (255, 255, 255), -1)
        
        # Create temporary image files
        self.temp_dir = tempfile.mkdtemp()
        
        # Save as different formats
        self.jpg_path = os.path.join(self.temp_dir, "test.jpg")
        self.png_path = os.path.join(self.temp_dir, "test.png")
        
        cv2.imwrite(self.jpg_path, self.test_image)
        cv2.imwrite(self.png_path, self.test_image)
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.jpg_path):
            os.remove(self.jpg_path)
        if os.path.exists(self.png_path):
            os.remove(self.png_path)
        os.rmdir(self.temp_dir)
    
    def test_load_image_from_path(self):
        """Test loading image from file path"""
        loaded = load_image(self.jpg_path)
        self.assertIsInstance(loaded, np.ndarray)
        self.assertEqual(len(loaded.shape), 3)  # Should be 3D (H, W, C)
        self.assertEqual(loaded.shape[2], 3)    # Should have 3 channels
    
    def test_load_image_different_formats(self):
        """Test loading different image formats"""
        jpg_loaded = load_image(self.jpg_path)
        png_loaded = load_image(self.png_path)
        
        self.assertIsInstance(jpg_loaded, np.ndarray)
        self.assertIsInstance(png_loaded, np.ndarray)
        self.assertEqual(jpg_loaded.shape, png_loaded.shape)
    
    def test_load_image_from_uploaded_file(self):
        """Test loading image from uploaded file-like object"""
        # Create a file-like object
        pil_image = Image.fromarray(cv2.cvtColor(self.test_image, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG')
        buffer.seek(0)
        
        # Mock uploaded file object
        class MockUploadedFile:
            def __init__(self, buffer):
                self.buffer = buffer
            
            def read(self):
                return self.buffer.read()
        
        mock_file = MockUploadedFile(buffer)
        loaded = load_image(mock_file)
        
        self.assertIsInstance(loaded, np.ndarray)
        self.assertEqual(len(loaded.shape), 3)
        self.assertEqual(loaded.shape[2], 3)
    
    def test_dummy_crop_functionality(self):
        """Test that dummy_crop function exists and is callable"""
        # This test assumes dummy_crop exists - if not, it should be implemented
        try:
            result = dummy_crop(self.test_image)
            self.assertIsNotNone(result)
        except (NameError, AttributeError):
            self.skipTest("dummy_crop function not implemented yet")
    
    def test_load_image_nonexistent_file(self):
        """Test loading from nonexistent file raises appropriate error"""
        with self.assertRaises((FileNotFoundError, cv2.error)):
            load_image("/nonexistent/path/image.jpg")
    
    def test_load_image_returns_bgr_format(self):
        """Test that loaded images are in BGR format (OpenCV standard)"""
        loaded = load_image(self.jpg_path)
        
        # Create a test with known colors
        red_image = np.zeros((50, 50, 3), dtype=np.uint8)
        red_image[:, :, 2] = 255  # Red in BGR
        
        red_path = os.path.join(self.temp_dir, "red.jpg")
        cv2.imwrite(red_path, red_image)
        
        loaded_red = load_image(red_path)
        
        # Check that red channel (index 2 in BGR) has highest values
        self.assertGreater(np.mean(loaded_red[:, :, 2]), np.mean(loaded_red[:, :, 0]))
        self.assertGreater(np.mean(loaded_red[:, :, 2]), np.mean(loaded_red[:, :, 1]))
        
        os.remove(red_path)


if __name__ == "__main__":
    unittest.main()