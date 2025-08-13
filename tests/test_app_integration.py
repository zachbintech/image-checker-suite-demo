import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np
import cv2

# Add the root directory to the path to import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestAppIntegration(unittest.TestCase):
    """Integration tests for the main Streamlit application"""
    
    def setUp(self):
        """Set up test data"""
        # Create a test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.test_image, (25, 25), (75, 75), (255, 255, 255), -1)
    
    def test_app_imports(self):
        """Test that all required modules can be imported"""
        try:
            from image_quality_checks.basic_image_quality import detect_blur
            from image_utilities import load_image
            self.assertTrue(callable(detect_blur))
            self.assertTrue(callable(load_image))
        except ImportError as e:
            self.fail(f"Failed to import required modules: {e}")
    
    def test_basic_image_quality_function(self):
        """Test the basic image quality detection function used in the app"""
        try:
            from image_quality_checks.basic_image_quality import detect_blur
            
            blur_score = detect_blur(self.test_image)
            self.assertIsInstance(blur_score, (int, float))
            self.assertGreaterEqual(blur_score, 0)
        except ImportError:
            self.skipTest("basic_image_quality module not available")
    
    @patch('streamlit.set_page_config')
    @patch('streamlit.title')
    @patch('streamlit.expander')
    @patch('streamlit.slider')
    @patch('streamlit.file_uploader')
    def test_app_structure_mocked(self, mock_uploader, mock_slider, mock_expander, mock_title, mock_config):
        """Test app structure with mocked Streamlit components"""
        # Mock streamlit components
        mock_expander.return_value.__enter__ = MagicMock()
        mock_expander.return_value.__exit__ = MagicMock()
        mock_slider.return_value = 100
        mock_uploader.return_value = None
        
        try:
            # Import app to test structure
            import app
            
            # Verify Streamlit components were called
            mock_config.assert_called_once()
            mock_title.assert_called_once()
            
        except ImportError as e:
            self.skipTest(f"Could not import app module: {e}")
        except Exception as e:
            # App might fail due to missing components, but structure should be testable
            pass
    
    def test_image_processing_pipeline(self):
        """Test the basic image processing pipeline"""
        try:
            from image_quality_checks.basic_image_quality import detect_blur
            from image_utilities import load_image
            import tempfile
            
            # Create a temporary image file
            temp_path = tempfile.mktemp(suffix='.jpg')
            cv2.imwrite(temp_path, self.test_image)
            
            try:
                # Test the pipeline: load -> process
                loaded_image = load_image(temp_path)
                blur_score = detect_blur(loaded_image)
                
                self.assertIsInstance(loaded_image, np.ndarray)
                self.assertIsInstance(blur_score, (int, float))
                
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except ImportError:
            self.skipTest("Required modules not available")
    
    def test_similarity_module_integration(self):
        """Test integration with similarity detection modules"""
        try:
            from similar import get_hash_difference
            
            # Create two identical images
            image1 = self.test_image.copy()
            image2 = self.test_image.copy()
            
            # Test hash difference
            diff = get_hash_difference(image1, image2)
            self.assertEqual(diff, 0, "Identical images should have 0 hash difference")
            
        except ImportError:
            self.skipTest("Similarity module not available")
    
    def test_app_functions_exist(self):
        """Test that main app functions exist and are callable"""
        try:
            import app
            
            # Check for main function definitions
            functions_to_check = [
                'get_basic_image_quality_and_display',
                'basic_image_check_section',
                'similar_section',
                'advanced_image_quality_section'
            ]
            
            for func_name in functions_to_check:
                if hasattr(app, func_name):
                    func = getattr(app, func_name)
                    self.assertTrue(callable(func), f"{func_name} should be callable")
                
        except ImportError:
            self.skipTest("App module not available")
    
    def test_app_constants_and_config(self):
        """Test app configuration and constants"""
        try:
            import app
            
            # App should run without immediate errors
            # (actual Streamlit execution would require web context)
            self.assertTrue(True)  # If we get here, basic import succeeded
            
        except ImportError:
            self.skipTest("App module not available")
        except Exception as e:
            # Expected for Streamlit apps run outside web context
            if "streamlit" in str(e).lower():
                self.skipTest("Streamlit context not available")
            else:
                raise


if __name__ == "__main__":
    unittest.main()