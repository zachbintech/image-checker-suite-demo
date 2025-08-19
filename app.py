from image_quality_checks.basic_image_quality import detect_blur
from image_quality_checks.exposure import detect_exposure
from image_utilities import dummy_crop, load_image
from similarity_api import group_similar_images_by_embeddings, find_similar_pairs, compute_embeddings_for_uploaded_images, summarize_similarity_results
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile

# --- Streamlit App ---
st.set_page_config(page_title="Smart Photo Digitization Demo", layout="centered")
st.title("Photo-to-Digital ML Demo")

# 1. Basic Image Quality Check (Blur / Exposure)
def detect_exposure_from_array(image, overexposure_thresh=0.95, underexposure_thresh=0.05, pixel_ratio_thresh=0.05):
    """
    Detect if an image array is overexposed or underexposed.
    
    Args:
        image (numpy.ndarray): Image array (BGR format from OpenCV).
        overexposure_thresh (float): Pixel brightness threshold for overexposure (0.0-1.0).
        underexposure_thresh (float): Pixel brightness threshold for underexposure (0.0-1.0).
        pixel_ratio_thresh (float): Fraction of total pixels allowed to trigger a warning.
    
    Returns:
        tuple: (status, overexposed_ratio, underexposed_ratio)
    """
    # Convert to grayscale and normalize
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    normalized_image = gray_image / 255.0
    
    # Find very bright and very dark pixels
    overexposed_pixels = np.sum(normalized_image > overexposure_thresh)
    underexposed_pixels = np.sum(normalized_image < underexposure_thresh)
    total_pixels = normalized_image.size
    
    overexposed_ratio = overexposed_pixels / total_pixels
    underexposed_ratio = underexposed_pixels / total_pixels
    
    if overexposed_ratio > pixel_ratio_thresh:
        return "overexposed", overexposed_ratio, underexposed_ratio
    elif underexposed_ratio > pixel_ratio_thresh:
        return "underexposed", overexposed_ratio, underexposed_ratio
    else:
        return "normal", overexposed_ratio, underexposed_ratio

def get_basic_image_quality_and_display(image, blur_threshold=100, exposure_threshold=0.05):
    blur_score = detect_blur(image)
    blur_label = "Good" if blur_score > blur_threshold else "Blurry"
    
    exposure_status, overexposed_ratio, underexposed_ratio = detect_exposure_from_array(
        image, pixel_ratio_thresh=exposure_threshold
    )
    
    # Display metrics in columns
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Blur Score:** {blur_score:.2f} ({blur_label})")
    with col2:
        st.write(f"**Exposure:** {exposure_status.title()}")
        st.write(f"- Overexposed pixels: {overexposed_ratio:.1%}")
        st.write(f"- Underexposed pixels: {underexposed_ratio:.1%}")

def basic_image_check_section(get_basic_image_quality_and_display):
    with st.expander("1. Basic Image Quality Check (Blur / Exposure)", expanded=True):

        threshold_blur = st.slider(
            "Set blur threshold", 
            min_value=10, 
            max_value=300, 
            value=100, 
            step=5,
            key="blur_slider",
            help="Adjust the threshold to control how similar images need to be to be grouped."
        )

        threshold_exposure = st.slider(
            "Set exposure threshold", 
            min_value=0.01, 
            max_value=0.20, 
            value=0.05, 
            step=0.01,
            key="exposure_slider",
            help="Pixel ratio threshold for exposure detection (0.01-0.20). Lower values = more sensitive."
        )


        basic_quality_files = st.file_uploader(
            "Upload images for basic quality checks (blur, exposure)", 
            type=["jpg", "jpeg", "png", "tiff"], 
            accept_multiple_files=True,
            key="basic_quality"
        )

        if basic_quality_files:
            for uploaded_file in basic_quality_files:
                st.markdown(f"**Image: {uploaded_file.name}**")
                image = load_image(uploaded_file)
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

            # Blur and Exposure Detection
                get_basic_image_quality_and_display(image, threshold_blur, threshold_exposure)

basic_image_check_section(get_basic_image_quality_and_display)

# 2. Image Similarity
def get_similar_images_and_display(similarity_files, similarity_threshold):
    """
    Process uploaded images for similarity using DINOv2 embeddings and display grouped results.
    """
    # Load images into memory as (name, image_array) tuples
    images = []
    for uploaded_file in similarity_files:
        image = load_image(uploaded_file)  # Convert to numpy array
        images.append((uploaded_file.name, image))

    with st.spinner("Computing image embeddings and finding similarities..."):
        # Compute embeddings and find similar pairs
        embeddings = compute_embeddings_for_uploaded_images(images)
        similar_pairs = find_similar_pairs(embeddings, similarity_threshold)
        
        # Group similar images
        groups = group_similar_images_by_embeddings(images, similarity_threshold)
        
        # Get summary statistics
        summary = summarize_similarity_results(groups, similar_pairs)

    # Display results
    st.markdown("### Similarity Detection Results")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Groups", summary["total_groups"])
    with col2:
        st.metric("Similar Pairs Found", summary["similarity_pairs_count"])
    with col3:
        st.metric("Unique Images", summary["singleton_count"])

    # Display similar pairs with scores
    if similar_pairs:
        st.markdown("### Similar Image Pairs")
        for img1, img2, score in similar_pairs:
            st.write(f"**{img1}** ↔ **{img2}** (similarity: {score:.3f})")

    # Display grouped images
    multi_groups = [group for group in groups if len(group) > 1]
    if multi_groups:
        st.markdown("### Image Groups")
        for idx, group in enumerate(multi_groups):
            st.markdown(f"**Group {idx + 1}** ({len(group)} images)")
            for image_name in group:
                st.write(f"- {image_name}")
    else:
        st.info("No similar image groups found with the current threshold.")

def similar_section():
    with st.expander("2. Image Similarity Check", expanded=False):
        # Add a slider to adjust the similarity threshold dynamically
        similarity_threshold = st.slider(
            "Set similarity threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.8, 
            step=0.05,
            help="Cosine similarity threshold (0.0-1.0). Higher values = more strict similarity."
        )

        similarity_files = st.file_uploader(
            "Upload images for similarity checking", 
            type=["jpg", "jpeg", "png", "tiff"], 
            accept_multiple_files=True,
            key="similarity"
        )

        if similarity_files and len(similarity_files) > 1:
            get_similar_images_and_display(similarity_files, similarity_threshold)
        elif similarity_files and len(similarity_files) == 1:
            st.info("Upload at least 2 images to detect similarities.")
        else:
            st.info("Upload images to begin similarity detection.")


similar_section()

# 3. Advanced Image Quality (Artifact Detection and Removal)
def advanced_image_quality_section():
    with st.expander("3. Advanced Image Quality (Artifact Detection and Removal)", expanded=False):
        advanced_quality_files = st.file_uploader(
        "Upload images for advanced quality checks", 
        type=["jpg", "jpeg", "png", "tiff"], 
        accept_multiple_files=True,
        key="advanced_quality"
    )

        if advanced_quality_files:
            st.warning("Artifact detection and removal functionality is not implemented yet.")
            for uploaded_file in advanced_quality_files:
                st.markdown(f"**Uploaded: {uploaded_file.name}**")
                image = load_image(uploaded_file)
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

advanced_image_quality_section()
