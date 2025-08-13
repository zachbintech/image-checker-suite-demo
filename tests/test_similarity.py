import unittest
import cv2
import numpy as np
import os
import shutil

from similar import (
    get_hash_difference,
    process_and_group_images,
    find_duplicate_images_in_directory,
)


class TestImageSimilarity(unittest.TestCase):

    def setUp(self):
        """
        Create synthetic images for testing.
        """
        self.image1 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.image1, (25, 25), (75, 75), (255, 255, 255), -1)

        self.image2 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(self.image2, (25, 25), (75, 75), (255, 255, 255), -1)

        self.image3 = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(self.image3, (50, 50), 25, (255, 255, 255), -1)

    def test_hash_difference(self):
        """
        Test that hash difference between identical and different images behaves as expected.
        """
        diff1 = get_hash_difference(self.image1, self.image2)
        diff2 = get_hash_difference(self.image1, self.image3)

        print(f"Hash difference between image1 and image2 (identical): {diff1}")
        print(f"Hash difference between image1 and image3 (different): {diff2}")

        self.assertEqual(diff1, 0, "Identical images should have 0 hash difference")
        self.assertGreater(diff2, 0, "Different images should have non-zero hash difference")

    # def test_group_images_by_hash(self):
    #     """
    #     Test perceptual hash grouping with synthetic images.
    #     """
    #     temp_dir = "temp_test_images"
    #     os.makedirs(temp_dir, exist_ok=True)

    #     # Save images
    #     cv2.imwrite(os.path.join(temp_dir, "image1.jpg"), self.image1)
    #     cv2.imwrite(os.path.join(temp_dir, "image2.jpg"), self.image2)
    #     cv2.imwrite(os.path.join(temp_dir, "image3.jpg"), self.image3)

    #     try:
    #         groups = process_and_group_images(temp_dir, threshold=5)

    #         print("Grouped images based on phash:")
    #         for i, group in enumerate(groups):
    #             print(f"Group {i + 1}: {group}")

    #         # Should be two groups: one for image1 & image2, one for image3
    #         self.assertEqual(len(groups), 2)
    #         all_grouped = sum(groups, [])
    #         self.assertIn("image1.jpg", all_grouped)
    #         self.assertIn("image2.jpg", all_grouped)
    #         self.assertIn("image3.jpg", all_grouped)

    #     finally:
    #         shutil.rmtree(temp_dir)

    def test_real_image_similarity(self):
        """
        Test cosine similarity using DINOv2 embeddings on real images.
        Make sure to populate 'real_images/' with sample images first.
        """
        real_images_dir = os.path.abspath("real_images")
        if not os.path.exists(real_images_dir):
            self.skipTest(f"Directory '{real_images_dir}' does not exist.")

        similarities = find_duplicate_images_in_directory(real_images_dir)

        if similarities:
            avg_similarity = sum(similarities.values()) / len(similarities)
            print(f"Average cosine similarity in directory: {avg_similarity:.4f}")
        else:
            print("No similar pairs found or not enough images.")

        self.assertIsInstance(similarities, dict)


if __name__ == "__main__":
    unittest.main()
