# Smart Photo Digitization Demo

An image quality checking and similarity detection suite designed for processing and analyzing digitized photos. Built with Python and Streamlit.

## Quick Start with Docker (Recommended)

### Prerequisites
- Docker
- Docker Compose

### Running the Application

**Start the Streamlit web interface:**
```bash
docker compose up
```

Then open your browser to http://localhost:8501

**Rebuild after code changes:**
```bash
docker compose up --build
```

**Stop the application:**
```bash
docker compose down
```

### Development with Docker

**Run tests inside Docker:**
```bash
docker compose run --rm app python3 -m unittest discover -s tests
```

**Test real image groupings:**
```bash
docker compose run --rm app python evaluate_groupsing.py
```

**Run artifact detection:**
```bash
docker compose run --rm app python artifacts/bad_bands.py
```

**Interactive development shell:**
```bash
docker compose --profile dev run --rm dev
```

This gives you a bash shell inside the container where you can run any commands.

### Docker Architecture

- **app service**: Runs the Streamlit web interface on port 8501
- **dev service**: Provides an interactive shell for running tests and scripts
- Volume mounts ensure code changes are reflected immediately
- Output directories (grouped_output, qc_overlays, review, masks, temp) persist on your host

## Local Development (Without Docker)

### Environment Setup
```bash
python3 -m venv photoenv
source photoenv/bin/activate
pip install -r requirements.txt
```

### Running Locally
```bash
streamlit run app.py
```

### Run Tests
```bash
python3 -m unittest discover -s tests
```

### Test Image Groupings
```bash
python evaluate_groupsing.py
```

### Artifact Detection
```bash
python artifacts/bad_bands.py
```

## Project Structure

- **app.py**: Main Streamlit application
- **similar/**: Image similarity and grouping algorithms
- **image_quality_checks/**: Blur detection and quality assessment
- **artifacts/**: CNN-based artifact detection
- **defects/**: Various defect detection algorithms
- **tests/**: Comprehensive test suite

## Key Features

1. **Basic Image Quality Check**: Blur detection using Laplacian variance
2. **Image Similarity Detection**: Multi-tier approach using perceptual hashing, SIFT features, and DINOv2 embeddings
3. **Artifact Detection**: CNN-based detection for digitization artifacts 


