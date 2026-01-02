import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from backend import ComicTranslator

# Page configuration
st.set_page_config(
    page_title="Comic Translator",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("📚 Comic Translator")
st.markdown("Translate comic book pages from English to Spanish (Spain)")

# Initialize session state
if 'translator' not in st.session_state:
    try:
        st.session_state.translator = ComicTranslator()
    except Exception as e:
        st.error(f"Error initializing translator: {e}")
        st.stop()

if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

if 'original_image' not in st.session_state:
    st.session_state.original_image = None

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Comic Page")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a comic book page image (.jpg or .png)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image info
        st.info(f"File: {uploaded_file.name}")
        st.info(f"Size: {uploaded_file.size / 1024:.2f} KB")
        
        # Process button
        # NOTE: Buttons still use use_container_width usually, but if this warns too, change to width="stretch"
        if st.button("🚀 Process Image", type="primary", use_container_width=True):
            with st.spinner("Processing image... This may take a moment."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process image
                    original, processed = st.session_state.translator.process_image(tmp_path)
                    
                    # Store in session state
                    st.session_state.original_image = original
                    st.session_state.processed_image = processed
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    st.success("Image processed successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing image: {e}")
                    st.exception(e)

# Main content area
if st.session_state.original_image is not None and st.session_state.processed_image is not None:
    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📖 Original")
        # Convert BGR to RGB for display
        original_rgb = cv2.cvtColor(st.session_state.original_image, cv2.COLOR_BGR2RGB)
        # FIX APPLIED: use_container_width=True -> width="stretch"
        st.image(original_rgb, width="stretch", channels="RGB")
    
    with col2:
        st.subheader("✨ Translated")
        # Convert BGR to RGB for display
        processed_rgb = cv2.cvtColor(st.session_state.processed_image, cv2.COLOR_BGR2RGB)
        # FIX APPLIED: use_container_width=True -> width="stretch"
        st.image(processed_rgb, width="stretch", channels="RGB")
    
    # Download button for translated image
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Convert to PIL Image for download
        processed_pil = Image.fromarray(processed_rgb)
        
        # Save to bytes
        from io import BytesIO
        buf = BytesIO()
        processed_pil.save(buf, format='PNG')
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Translated Image",
            data=buf,
            file_name="translated_comic.png",
            mime="image/png",
            use_container_width=True
        )

else:
    # Welcome message
    st.info("👈 Please upload a comic book page image using the sidebar to get started.")
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Upload an image**: Click on the file uploader in the sidebar and select a comic book page (.jpg or .png)
        2. **Process**: Click the "Process Image" button
        3. **View results**: See the original and translated images side by side
        4. **Download**: Download your translated comic page
        
        **Features:**
        - Automatically detects speech bubbles
        - Ignores titles and large text boxes
        - Translates English to Spanish (Spain)
        - Preserves comic book style with custom font
        """)
