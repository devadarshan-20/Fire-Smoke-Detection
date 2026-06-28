import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide"
)

# Title
st.title("🔥 Fire & Smoke Detection System")

st.markdown("""
Upload an image to detect **Fire** or **Smoke** using a trained YOLOv8 model.
""")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
