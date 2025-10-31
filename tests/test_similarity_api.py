"""
Tests for similarity_api.py which is used by the Streamlit app.
This tests the actual image similarity detection pipeline.
"""
import unittest
import numpy as np
import cv2
import tempfile
import os

from similarity_api import (
    compute_embeddings_for_uploaded_images,
    find_similar_pairs,
    group_similar_images_by_embeddings,
    summarize_similarity_results
)


class TestSimilarityAPI(unittest.TestCase):

    def setUp(self):
        """Create test images for similarity detection"""
        # Create two similar images
        self.image1 = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.rectangle(self.image1, (50, 50), (174, 174), (255, 255, 255), -1)
        cv2.circle(self.image1, (112, 112), 30, (100, 100, 100), -1)

        # Almost identical image
        self.image2 = self.image1.copy()
        cv2.circle(self.image2, (112, 112), 31, (100, 100, 100), -1)

        # Very different image
        self.image3 = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.ellipse(self.image3, (112, 112), (80, 40), 45, 0, 360, (200, 150, 100), -1)

        # Prepare images as (name, array) tuples
        self.images = [
            ("image1.jpg", self.image1),
            ("image2.jpg", self.image2),
            ("image3.jpg", self.image3)
        ]

    def test_compute_embeddings_returns_dict(self):
        """Test that compute_embeddings returns a dictionary"""
        embeddings = compute_embeddings_for_uploaded_images(self.images)

        self.assertIsInstance(embeddings, dict)
        self.assertEqual(len(embeddings), 3)
        self.assertIn("image1.jpg", embeddings)
        self.assertIn("image2.jpg", embeddings)
        self.assertIn("image3.jpg", embeddings)

    def test_embeddings_are_numpy_arrays(self):
        """Test that embeddings are numpy arrays with consistent dimensions"""
        embeddings = compute_embeddings_for_uploaded_images(self.images)

        for image_name, embedding in embeddings.items():
            self.assertIsInstance(embedding, np.ndarray)
            self.assertGreater(embedding.size, 0)

        # All embeddings should have the same shape
        shapes = [emb.shape for emb in embeddings.values()]
        self.assertEqual(len(set([str(s) for s in shapes])), 1)

    def test_find_similar_pairs_high_threshold(self):
        """Test finding similar pairs with high threshold"""
        embeddings = compute_embeddings_for_uploaded_images(self.images)
        similar_pairs = find_similar_pairs(embeddings, similarity_threshold=0.95)

        # Should find at least the similar pair (image1, image2)
        self.assertIsInstance(similar_pairs, list)

        # Each pair should be a tuple of (name1, name2, score)
        for pair in similar_pairs:
            self.assertEqual(len(pair), 3)
            self.assertIsInstance(pair[0], str)
            self.assertIsInstance(pair[1], str)
            self.assertIsInstance(pair[2], (float, np.floating))
            self.assertGreaterEqual(pair[2], 0.95)

    def test_find_similar_pairs_low_threshold(self):
        """Test that low threshold finds more pairs"""
        embeddings = compute_embeddings_for_uploaded_images(self.images)

        high_threshold_pairs = find_similar_pairs(embeddings, similarity_threshold=0.95)
        low_threshold_pairs = find_similar_pairs(embeddings, similarity_threshold=0.5)

        # Lower threshold should find at least as many pairs
        self.assertGreaterEqual(len(low_threshold_pairs), len(high_threshold_pairs))

    def test_group_similar_images_creates_groups(self):
        """Test that grouping creates appropriate groups"""
        groups = group_similar_images_by_embeddings(self.images, similarity_threshold=0.8)

        self.assertIsInstance(groups, list)
        # Should have at least one group
        self.assertGreater(len(groups), 0)

        # All images should be in some group
        all_images = set()
        for group in groups:
            self.assertIsInstance(group, list)
            all_images.update(group)

        self.assertEqual(len(all_images), 3)

    def test_group_similar_images_empty_input(self):
        """Test grouping with empty input"""
        groups = group_similar_images_by_embeddings([], similarity_threshold=0.8)
        self.assertEqual(groups, [])

    def test_summarize_similarity_results(self):
        """Test summary statistics generation"""
        groups = group_similar_images_by_embeddings(self.images, similarity_threshold=0.8)
        embeddings = compute_embeddings_for_uploaded_images(self.images)
        similar_pairs = find_similar_pairs(embeddings, similarity_threshold=0.8)

        summary = summarize_similarity_results(groups, similar_pairs)

        self.assertIsInstance(summary, dict)
        self.assertIn("total_groups", summary)
        self.assertIn("total_images", summary)
        self.assertIn("singleton_count", summary)
        self.assertIn("similarity_pairs_count", summary)

        # Total images should be 3
        self.assertEqual(summary["total_images"], 3)

        # Number of pairs should match
        self.assertEqual(summary["similarity_pairs_count"], len(similar_pairs))

    def test_identical_images_have_high_similarity(self):
        """Test that identical images have very high similarity"""
        identical_images = [
            ("img1.jpg", self.image1),
            ("img2.jpg", self.image1.copy())  # Exact copy
        ]

        embeddings = compute_embeddings_for_uploaded_images(identical_images)
        similar_pairs = find_similar_pairs(embeddings, similarity_threshold=0.99)

        # Should find the identical pair
        self.assertEqual(len(similar_pairs), 1)
        self.assertGreater(similar_pairs[0][2], 0.99)


if __name__ == "__main__":
    unittest.main()
