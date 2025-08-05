from basic_image_quality import detect_blur
from face_detection import detect_faces
from image_utilities import dummy_crop, load_image
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
def get_basic_image_quality_and_display(image):
    blur_score = detect_blur(image)
    blur_label = "Good" if blur_score > 100 else "Blurry"
    st.write(f"Blur Score: {blur_score:.2f} ({blur_label})")

def basic_image_check_section(get_basic_image_quality_and_display):
    with st.expander("1. Basic Image Quality Check (Blur / Exposure)", expanded=False):
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

            # Blur Detection
                get_basic_image_quality_and_display(image)

basic_image_check_section(get_basic_image_quality_and_display)

# 2. Face Detection
def get_faces_and_display(uploaded_file):
    image = load_image(uploaded_file)
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Face Detection
    face_annotated, num_faces = detect_faces(image.copy())
    st.image(cv2.cvtColor(face_annotated, cv2.COLOR_BGR2RGB), 
             caption=f"{num_faces} face(s) detected", 
             use_container_width=True)

def face_detection_section(get_faces_and_display):
    with st.expander("2. Face Detection", expanded=False):
        face_detection_files = st.file_uploader(
        "Upload images for face detection", 
        type=["jpg", "jpeg", "png", "tiff"], 
        accept_multiple_files=True,
        key="face_detection"
    )

        if face_detection_files:
            for uploaded_file in face_detection_files:
                st.markdown(f"**Image: {uploaded_file.name}**")
                get_faces_and_display(uploaded_file)

face_detection_section(get_faces_and_display)

# 3. Image Similarity
def get_similar_images_and_display(similarity_files, threshold_matches):
    """
    Process uploaded images for similarity and display grouped results.
    """
    # Load images into memory as (name, image_array) tuples
    images = []
    for uploaded_file in similarity_files:
        image = load_image(uploaded_file)  # Convert to numpy array
        images.append((uploaded_file.name, image))

    # Group similar images
    groups = group_similar_images_in_memory(images, threshold_matches)

    # Display grouped images
    for idx, group in enumerate(groups):
        st.markdown(f"### Group {idx + 1}")
        for image_name in group:
            st.write(f"- {image_name}")

def similar_section():
    with st.expander("3. Image Similarity Check", expanded=False):
        # Add a slider to adjust the threshold dynamically
        threshold_matches = st.slider(
            "Set similarity threshold", 
            min_value=10, 
            max_value=300, 
            value=33, 
            step=5,
            help="Adjust the threshold to control how similar images need to be to be grouped."
        )

        similarity_files = st.file_uploader(
            "Upload images for similarity checking", 
            type=["jpg", "jpeg", "png", "tiff"], 
            accept_multiple_files=True,
            key="similarity"
        )


similar_section()

# 4. Advanced Image Quality (Artifact Detection and Removal)
def advanced_image_quality_section():
    with st.expander("4. Advanced Image Quality (Artifact Detection and Removal)", expanded=False):
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
