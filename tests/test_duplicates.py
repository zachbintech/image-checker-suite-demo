import os
import shutil
import unittest
from duplicate.duplicate import find_duplicate_images_in_directory
from PIL import Image
import numpy as np

class TestFindDuplicateImagesInDirectory(unittest.TestCase):
    def setUp(self):
        """
        Set up a directory named 'similar_images' with test images.
        """
        self.test_dir = "similar_images"
        os.makedirs(self.test_dir, exist_ok=True)

        # Create dummy images
        img1 = Image.new('RGB', (100, 100), color='red')
        img2 = Image.new('RGB', (100, 100), color='red')  # Duplicate of img1
        img3 = Image.new('RGB', (100, 100), color='blue')

        img1.save(os.path.join(self.test_dir, "image1.jpg"))
        img2.save(os.path.join(self.test_dir, "image2.jpg"))
        img3.save(os.path.join(self.test_dir, "image3.jpg"))

    def tearDown(self):
        """
        Clean up the 'similar_images' directory after tests.
        """
        shutil.rmtree(self.test_dir)

    def test_find_duplicate_images_in_directory(self):
        """
        Test the find_duplicate_images_in_directory function.
        """
        similarities = find_duplicate_images_in_directory(self.test_dir)

        # Normalize paths for comparison
        image1_path = os.path.abspath(os.path.join(self.test_dir, "image1.jpg"))
        image2_path = os.path.abspath(os.path.join(self.test_dir, "image2.jpg"))
        image3_path = os.path.abspath(os.path.join(self.test_dir, "image3.jpg"))

        # Check that the similarity between image1 and image2 is high (close to 1)
        self.assertGreater(similarities[(image1_path, image2_path)], 0.9)

        # Check that the similarity between image1 and image3 is low
        self.assertLess(similarities[(image1_path, image3_path)], 0.5)

if __name__ == "__main__":
    unittest.main()