import unittest
import os
import shutil
from unittest.mock import patch, call
from similar.evaluate_groupings import summarize_grouping, export_grouped_images, evaluate_image_grouping

class TestEvaluateGroupings(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_export_output"
        os.makedirs(self.test_dir, exist_ok=True)
        # Create dummy image files
        self.img1 = os.path.join(self.test_dir, "img1.jpg")
        self.img2 = os.path.join(self.test_dir, "img2.jpg")
        self.img3 = os.path.join(self.test_dir, "img3.jpg")
        for img in [self.img1, self.img2, self.img3]:
            with open(img, "w") as f:
                f.write("dummy")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if os.path.exists("exported_groups"):
            shutil.rmtree("exported_groups")
        if os.path.exists("another_output"):
            shutil.rmtree("another_output")

    def test_summarize_grouping_empty(self):
        with patch("builtins.print") as mock_print:
            summarize_grouping([])
            mock_print.assert_any_call("\n📊 Grouping Summary:")
            mock_print.assert_any_call("Total Groups: 0")
            mock_print.assert_any_call("Group sizes: []")
            mock_print.assert_any_call("Singleton groups (likely unique): 0")
            mock_print.assert_any_call("Large groups (>3 images): 0")

    def test_summarize_grouping_mixed(self):
        groups = [
            ["img1.jpg"],  # singleton
            ["img2.jpg", "img3.jpg"],  # size 2
            ["img4.jpg", "img5.jpg", "img6.jpg", "img7.jpg"],  # size 4
        ]
        with patch("builtins.print") as mock_print:
            summarize_grouping(groups)
            mock_print.assert_any_call("Total Groups: 3")
            mock_print.assert_any_call("Group sizes: [1, 2, 4]")
            mock_print.assert_any_call("Singleton groups (likely unique): 1")
            mock_print.assert_any_call("Large groups (>3 images): 1")

    def test_summarize_grouping_all_singletons(self):
        groups = [["a.jpg"], ["b.jpg"], ["c.jpg"]]
        with patch("builtins.print") as mock_print:
            summarize_grouping(groups)
            mock_print.assert_any_call("Total Groups: 3")
            mock_print.assert_any_call("Group sizes: [1, 1, 1]")
            mock_print.assert_any_call("Singleton groups (likely unique): 3")
            mock_print.assert_any_call("Large groups (>3 images): 0")

    def test_export_grouped_images_exports_only_groups_gt1(self):
        groups = [
            [self.img1],  # singleton, should not be exported
            [self.img2, self.img3],  # should be exported
        ]
        output_dir = "exported_groups"
        export_grouped_images(groups, output_dir)
        group_dirs = [d for d in os.listdir(output_dir) if d.startswith("group_")]
        self.assertEqual(len(group_dirs), 1)
        group1_path = os.path.join(output_dir, group_dirs[0])
        self.assertTrue(os.path.exists(os.path.join(group1_path, "img2.jpg")))
        self.assertTrue(os.path.exists(os.path.join(group1_path, "img3.jpg")))
        self.assertFalse(os.path.exists(os.path.join(output_dir, "group_1", "img1.jpg")))

    def test_export_grouped_images_creates_output_dir(self):
        groups = [[self.img2, self.img3]]
        output_dir = "another_output"
        export_grouped_images(groups, output_dir)
        self.assertTrue(os.path.exists(output_dir))
        group_dirs = [d for d in os.listdir(output_dir) if d.startswith("group_")]
        self.assertEqual(len(group_dirs), 1)

    def test_export_grouped_images_handles_copy_error(self):
        groups = [[self.img2, "/nonexistent/path.jpg"]]
        output_dir = "exported_groups"
        with patch("builtins.print") as mock_print:
            export_grouped_images(groups, output_dir)
            mock_print.assert_any_call(f"⚠️ Could not copy /nonexistent/path.jpg: [Errno 2] No such file or directory: '/nonexistent/path.jpg'")

    @patch("similar.evaluate_groupings.process_and_group_images_recursive")
    @patch("similar.evaluate_groupings.export_grouped_images")
    @patch("similar.evaluate_groupings.summarize_grouping")
    def test_evaluate_image_grouping_calls(self, mock_summarize, mock_export, mock_process):
        mock_process.return_value = [["a.jpg"], ["b.jpg", "c.jpg"]]
        evaluate_image_grouping("indir", 5, "outdir")
        mock_process.assert_called_once_with("indir", 5)
        mock_summarize.assert_called_once_with([["a.jpg"], ["b.jpg", "c.jpg"]])
        mock_export.assert_called_once_with([["a.jpg"], ["b.jpg", "c.jpg"]], "outdir")

    def test_export_grouped_images_no_groups(self):
        output_dir = "exported_groups"
        export_grouped_images([], output_dir)
        self.assertTrue(os.path.exists(output_dir))
        self.assertEqual(os.listdir(output_dir), [])

    def test_export_grouped_images_all_singletons(self):
        groups = [[self.img1], [self.img2]]
        output_dir = "exported_groups"
        export_grouped_images(groups, output_dir)
        self.assertTrue(os.path.exists(output_dir))
        self.assertEqual(os.listdir(output_dir), [])

if __name__ == "__main__":
    unittest.main()