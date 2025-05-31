import unittest
import cv2
import numpy as np
import os
import shutil
from similar import (compute_and_store_hashes, group_images_by_hash,
                     group_similar_images_in_memory, get_hash_difference, process_and_group_images)


class TestImageSimilarity(unittest.TestCase):

    def setUp(self):
        # Create dummy images for testing
        self.image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.image1, (25, 25), (75, 75), (255, 255, 255), -1)

        self.image2 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.image2, (25, 25), (75, 75), (255, 255, 255), -1)

        self.image3 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(self.image3, (50, 50), 25, (255, 255, 255), -1)

        self.images = [
            ("image1", self.image1),
            ("image2", self.image2),
            ("image3", self.image3)
        ]

    def display_images(self):
        # Display images for inspection
        for name, image in self.images:
            cv2.imshow(name, image)
        print("Press any key to close the images...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def test_no_similar_images(self):
        # Test when no images are similar
        groups = group_similar_images_in_memory(self.images, threshold_matches=100)
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0], ["image1"])
        self.assertEqual(groups[1], ["image2"])
        self.assertEqual(groups[2], ["image3"])

    def test_all_images_similar(self):
        # Test when all images are considered similar
        groups = group_similar_images_in_memory(self.images, threshold_matches=0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(set(groups[0]), {"image1", "image2", "image3"})

    def test_hash_difference(self):
        """
        Test the hash difference function with similar and dissimilar images.
        """
        # Hash difference between identical images
        diff1 = get_hash_difference(self.image1, self.image2)
        self.assertEqual(diff1, 0, "Hash difference for identical images should be 0")

        # Hash difference between different images
        diff2 = get_hash_difference(self.image1, self.image3)
        self.assertGreater(diff2, 0, "Hash difference for different images should be greater than 0")

        print(f"Hash difference between image1 and image2: {diff1}")
        print(f"Hash difference between image1 and image3: {diff2}")

    def test_group_images_by_hash(self):
        """
        Test grouping images by perceptual hash from a directory.
        """
        # Create a temporary directory with dummy images
        temp_dir = "temp_test_images"
        os.makedirs(temp_dir, exist_ok=True)

        # Save dummy images to the directory
        cv2.imwrite(os.path.join(temp_dir, "image1.jpg"), self.image1)
        cv2.imwrite(os.path.join(temp_dir, "image2.jpg"), self.image2)
        cv2.imwrite(os.path.join(temp_dir, "image3.jpg"), self.image3)

        try:
            # Use the new function to process and group images
            groups = process_and_group_images(temp_dir, threshold=5)

            # Display grouped images
            print("Grouped Images:")
            for i, group in enumerate(groups):
                print(f"Group {i + 1}: {group}")
                for image_name in group:
                    image_path = os.path.join(temp_dir, image_name)
                    image = cv2.imread(image_path)
                    cv2.imshow(f"Group {i + 1} - {image_name}", image)
                print("Press any key to close the images for this group...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Assertions to verify grouping
            self.assertEqual(len(groups), 2)
            self.assertIn("image1.jpg", groups[0])
            self.assertIn("image2.jpg", groups[0])
            self.assertIn("image3.jpg", groups[1])

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()