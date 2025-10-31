# Command-Line Interface Usage Guide

This guide explains how to use the standalone command-line scripts for batch processing images. These scripts provide the same functionality as the Streamlit web interface but are designed for automation, scripting, and processing large directories of images.

## Overview

Three standalone scripts are available:

1. **check_image_quality.py** - Detect blur and exposure issues
2. **check_similarity.py** - Find similar/duplicate images
3. **check_artifacts.py** - Detect color artifacts (red, green, blue lines)

All scripts can be run locally (with Python environment) or via Docker.

---

## 1. Image Quality Checker (`check_image_quality.py`)

Analyzes images for blur and exposure issues.

### Basic Usage

```bash
python check_image_quality.py /path/to/images
```

### With Docker

```bash
docker compose run --rm app python check_image_quality.py /app/your_directory
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `directory` | positional | required | Path to directory containing images |
| `--blur-threshold` | float | 100 | Blur threshold (higher = sharper required) |
| `--exposure-threshold` | float | 0.05 | Exposure pixel ratio threshold (lower = more sensitive) |
| `--output` | string | none | Save results to JSON file |

### Examples

```bash
# Check images with default settings
python check_image_quality.py ./photos

# Use stricter blur detection (require sharper images)
python check_image_quality.py ./photos --blur-threshold 150

# More sensitive exposure detection
python check_image_quality.py ./photos --exposure-threshold 0.03

# Save results to JSON
python check_image_quality.py ./photos --output quality_report.json
```

### Output Format

Terminal output shows:
- Filename
- Blur score and status (good/blurry)
- Exposure status (normal/overexposed/underexposed)
- Overexposed and underexposed pixel percentages
- Summary statistics

JSON output includes all metrics for programmatic processing.

---

## 2. Similarity Detector (`check_similarity.py`)

Finds similar or duplicate images using DINOv2 deep learning embeddings.

### Basic Usage

```bash
python check_similarity.py /path/to/images
```

### With Docker

```bash
docker compose run --rm app python check_similarity.py /app/your_directory
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `directory` | positional | required | Path to directory containing images |
| `--threshold` | float | 0.8 | Cosine similarity threshold (0.0-1.0, higher = stricter) |
| `--output` | string | none | Save results to JSON file |
| `--organize` | flag | false | Copy similar images into organized folders |
| `--organize-folder` | string | similar_images | Name of folder to organize similar images into |

### Examples

```bash
# Find similar images with default threshold
python check_similarity.py ./photos

# Stricter similarity (only very similar images)
python check_similarity.py ./photos --threshold 0.9

# More lenient (find more potential matches)
python check_similarity.py ./photos --threshold 0.7

# Save results to JSON
python check_similarity.py ./photos --output similarity_report.json

# Organize similar images into folders (copies images above threshold)
python check_similarity.py ./photos --threshold 0.8 --organize

# Organize with custom folder name
python check_similarity.py ./photos --organize --organize-folder "duplicates"

# Full workflow: find, organize, and save report
python check_similarity.py ./photos --threshold 0.85 --organize --output report.json
```

### Output Format

Terminal output shows:
- Summary statistics (total images, groups, unique images)
- Similar image pairs with similarity scores
- Grouped images (images that are similar to each other)

JSON output includes structured data with all similarity pairs and groups.

### Performance Notes

- First run downloads the DINOv2 model (~300MB)
- Processing time depends on number of images and hardware
- GPU acceleration is used if available

---

## 3. Artifact Detector (`check_artifacts.py`)

Detects color line artifacts (common in scanned photos).

### Basic Usage

```bash
python check_artifacts.py /path/to/images
```

### With Docker

```bash
docker compose run --rm app python check_artifacts.py /app/your_directory
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `directory` | positional | required | Path to directory containing images |
| `--threshold` | float | 1.5 | Color ratio threshold for artifact detection |
| `--save-masks` | flag | false | Save artifact mask images to subdirectory |
| `--output` | string | none | Save results to JSON file |

### Examples

```bash
# Detect artifacts with default settings
python check_artifacts.py ./photos

# More sensitive detection (lower threshold)
python check_artifacts.py ./photos --threshold 1.3

# Save visual masks showing artifacts
python check_artifacts.py ./photos --save-masks

# Save results to JSON
python check_artifacts.py ./photos --output artifacts_report.json

# Full analysis with masks and JSON
python check_artifacts.py ./photos --threshold 1.4 --save-masks --output report.json
```

### Output Format

Terminal output shows:
- Filename
- Red/green/blue artifact pixel counts
- Total artifact pixels and percentage
- Summary statistics

When `--save-masks` is used, creates an `artifact_masks/` subdirectory containing:
- `{filename}_red_mask.png` - Red artifacts highlighted
- `{filename}_green_mask.png` - Green artifacts highlighted
- `{filename}_blue_mask.png` - Blue artifacts highlighted

---

## Common Workflows

### Batch Quality Check

Process a large directory and save results:

```bash
python check_image_quality.py /path/to/scanned_photos --output quality.json
```

### Find Duplicates

Identify duplicate or near-duplicate images for cleanup:

```bash
python check_similarity.py /path/to/photos --threshold 0.9 --output duplicates.json
```

### Complete Image Analysis

Run all three checks on a directory:

```bash
# Check quality
python check_image_quality.py ./batch_001 --output batch_001_quality.json

# Find similarities
python check_similarity.py ./batch_001 --output batch_001_similarity.json

# Check for artifacts
python check_artifacts.py ./batch_001 --save-masks --output batch_001_artifacts.json
```

### Using with Docker

When running in Docker, mount your local directory:

```bash
# Add volume mount in docker-compose.yml or use docker compose run with full paths
docker compose run --rm -v /local/path:/data app python check_image_quality.py /data
```

---

## Supported Image Formats

All scripts support:
- JPEG (.jpg, .jpeg)
- PNG (.png)
- TIFF (.tiff)

Both uppercase and lowercase extensions are recognized.

---

## Tips

1. **Start with defaults** - The default thresholds work well for most cases
2. **Use JSON output** - For automated pipelines, save to JSON for easy parsing
3. **Test thresholds** - Run on a small sample first to tune thresholds
4. **Batch processing** - Process large directories efficiently by running overnight
5. **Combine results** - Use multiple scripts to get comprehensive analysis

---

## Exit Codes

All scripts use standard exit codes:
- `0` - Success
- `1` - Error (invalid arguments, directory not found, etc.)

---

## Getting Help

For detailed help on any script:

```bash
python check_image_quality.py --help
python check_similarity.py --help
python check_artifacts.py --help
```
