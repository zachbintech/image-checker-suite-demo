# Similar image detection package
import os
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import functions from similar_core module
try:
    from .similar_core import (
        get_hash_difference,
        process_and_group_images,
        find_duplicate_images_in_directory,
        compute_and_store_hashes,
        group_images_by_hash,
        process_and_group_images_recursive
    )
except ImportError:
    try:
        import similar_core as similar_module
        get_hash_difference = similar_module.get_hash_difference
        process_and_group_images = similar_module.process_and_group_images
        find_duplicate_images_in_directory = similar_module.find_duplicate_images_in_directory
        compute_and_store_hashes = similar_module.compute_and_store_hashes
        group_images_by_hash = similar_module.group_images_by_hash
        process_and_group_images_recursive = similar_module.process_and_group_images_recursive
    except ImportError:
        pass

# Import functions from evaluate_groupings module
try:
    from .evaluate_groupings import (
        summarize_grouping,
        export_grouped_images,
        evaluate_image_grouping
    )
except ImportError:
    try:
        import evaluate_groupings as eval_module
        summarize_grouping = eval_module.summarize_grouping
        export_grouped_images = eval_module.export_grouped_images
        evaluate_image_grouping = eval_module.evaluate_image_grouping
    except ImportError:
        pass