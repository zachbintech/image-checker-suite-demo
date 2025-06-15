import os
import shutil
from similar import process_and_group_images_recursive


def summarize_grouping(groups):
    """
    Print a summary of the image groups.
    """
    print("\n📊 Grouping Summary:")
    print(f"Total Groups: {len(groups)}")

    group_sizes = [len(group) for group in groups]
    print(f"Group sizes: {group_sizes}")

    singleton_count = sum(1 for size in group_sizes if size == 1)
    large_groups = [size for size in group_sizes if size > 3]

    print(f"Singleton groups (likely unique): {singleton_count}")
    print(f"Large groups (>3 images): {len(large_groups)}")


def export_grouped_images(groups, output_dir):
    """
    Export grouped images to subdirectories for manual inspection.
    Each group gets its own folder named 'group_<n>'.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, group in enumerate(groups):
        group_dir = os.path.join(output_dir, f"group_{i+1}")
        os.makedirs(group_dir, exist_ok=True)

        for image_path in group:
            filename = os.path.basename(image_path)
            dest_path = os.path.join(group_dir, filename)
            try:
                shutil.copy(image_path, dest_path)
            except Exception as e:
                print(f"⚠️ Could not copy {image_path}: {e}")


def evaluate_image_grouping(input_dir, threshold, output_dir):
    """
    Full evaluation pipeline:
    - Runs image grouping
    - Prints summary
    - Exports image groups
    """
    print(f"📁 Scanning: {input_dir}")
    print(f"🔍 Using threshold: {threshold}")
    groups = process_and_group_images_recursive(input_dir, threshold)

    summarize_grouping(groups)
    export_grouped_images(groups, output_dir)

    print(f"\n✅ Exported groups to: {output_dir}")


if __name__ == "__main__":
    # Update these paths as needed
    INPUT_DIR = "/home/zach/Desktop/PhotoData"         # Your top-level photo directory
    OUTPUT_DIR = "grouped_output"            # Where to save output
    HASH_THRESHOLD = 6                       # Adjust for stricter/looser grouping

    evaluate_image_grouping(INPUT_DIR, HASH_THRESHOLD, OUTPUT_DIR)
