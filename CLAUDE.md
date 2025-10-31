# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Smart Photo Digitization Demo - an image quality checking and similarity detection suite designed for processing and analyzing digitized photos. The application is built with Python and Streamlit, providing a web interface for various image analysis tasks.

## Development Commands

### Docker Setup (Recommended)

**Start the application:**
```bash
docker compose up
# Or run in background: docker compose up -d
```

**Rebuild after changes:**
```bash
docker compose up --build
```

**Stop the application:**
```bash
docker compose down
```

**Run tests:**
```bash
docker compose run --rm app python3 -m unittest discover -s tests
```

**Test image groupings:**
```bash
docker compose run --rm app python /app/evaluate_groupsing.py
```

**Interactive development shell:**
```bash
docker compose --profile dev run --rm dev
```

**View logs:**
```bash
docker compose logs -f
```

### Local Environment Setup (Alternative)

**Environment Setup:**
```bash
python3 -m venv photoenv
source photoenv/bin/activate
pip install -r requirements.txt
```

**Running the Application:**
```bash
streamlit run app.py
```

**Running Tests:**
```bash
python3 -m unittest discover -s tests
```

**Test Image Groupings (Main functionality):**
```bash
python /home/zach/Documents/dev/image-checker-suite-demo/evaluate_groupsing.py
```

## Architecture Overview

### Core Application Structure
- **app.py**: Main Streamlit application with three primary modules:
  1. Basic Image Quality Check (blur detection)
  2. Image Similarity Detection
  3. Advanced Image Quality (artifact detection - placeholder)

### Key Modules
- **image_quality_checks/**: Basic image quality assessment
  - `basic_image_quality.py`: Blur detection using Laplacian variance
- **face_detection.py**: Standalone OpenCV Haar cascade face detection (removed from main app, available for future use)
- **similar/**: Image similarity and grouping functionality
  - `similar.py`: Main similarity detection with multiple approaches (SIFT, perceptual hashing, DINOv2 embeddings)
  - `dino_embeddings.py`: DINOv2 deep learning embeddings for advanced similarity
  - `evaluate_groupings.py`: Testing and evaluation of grouping algorithms
- **image_utilities.py**: Common image loading and processing utilities
- **artifacts/**: Artifact detection and removal (CNN-based)
- **defects/**: Various defect detection algorithms
- **data/**: Data utilities and helpers

### Image Processing Pipeline
The application uses a multi-tier approach for image analysis:
1. **Perceptual Hashing**: Fast initial filtering using phash for near-duplicates
2. **SIFT Features**: Computer vision keypoint matching for structural similarity
3. **DINOv2 Embeddings**: Deep learning embeddings for semantic similarity using cosine similarity

### Data Organization
- **real_images/**: Source images for processing
- **masks/**: Generated masks and overlays
- **qc_overlays/**: Quality control visualization outputs
- **review/**: Images flagged for manual review
- **grouped_output/**: Results from similarity grouping algorithms

## Key Dependencies
- **streamlit**: Web interface framework
- **opencv-python-headless**: Computer vision operations
- **torch/torchvision**: Deep learning framework for DINOv2
- **scikit-learn**: Cosine similarity calculations
- **ImageHash**: Perceptual hashing for duplicate detection
- **timm**: Transformer models including DINOv2

## Testing Strategy
Tests are located in the `tests/` directory and provide comprehensive coverage:

### Core Test Files
- **test_groupings.py**: Image grouping algorithm validation and export functionality
- **test_similarity.py**: Similarity detection accuracy using hash differences and DINO embeddings
- **test_image_quality.py**: Blur detection algorithm testing with various image conditions
- **test_image_utilities.py**: Image loading and utility function validation
- **test_artifacts.py**: Artifact detection functionality (CNN-based)
- **test_dino_embeddings.py**: DINOv2 embedding computation and consistency
- **test_app_integration.py**: End-to-end application integration testing

### Test Coverage Areas
- **Image Quality**: Blur detection with sharp, blurry, and uniform images
- **Similarity Detection**: Hash-based and embedding-based similarity algorithms
- **Utility Functions**: Image loading from files and file-like objects
- **Error Handling**: Invalid inputs, missing files, and edge cases
- **Integration**: Full pipeline testing from image load to processing

### Running Tests
Use the standard unittest discovery:
```bash
python3 -m unittest discover -s tests
```

Individual test files can be run directly:
```bash
python3 -m unittest tests.test_image_quality
```

The main evaluation script tests real image groupings and creates output folders for manual verification of algorithm performance.