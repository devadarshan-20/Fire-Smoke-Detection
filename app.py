import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fire & Smoke Detection",
    page_icon="🔥",
    layout="wide"
)

# -----------------------------
# Load YOLO Model
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("model.pt")

model = load_model()

# -----------------------------
# Title
# -----------------------------
st.title("🔥 Fire & Smoke Detection System")

st.markdown("""
This application uses a trained **YOLOv8** model to detect **Fire** and **Smoke** from uploaded images.
""")

st.divider()

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# If image uploaded
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_container_width=True)

    if st.button("🔥 Detect Fire / Smoke", use_container_width=True):

        with st.spinner("Running Detection..."):

            # Save image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            # Prediction
            results = model.predict(
                source=temp_path,
                conf=0.5
            )

            result = results[0]

            # Draw detections
            plotted = result.plot()

            with col2:
                st.subheader("🤖 Detection Result")
                st.image(plotted, use_container_width=True)

            st.divider()

            # ----------------------------------
            # Detection Information
            # ----------------------------------

            if len(result.boxes) == 0:

                st.success("✅ No Fire or Smoke Detected")

            else:

                st.subheader("📊 Detection Summary")

                for box in result.boxes:

                    cls = int(box.cls[0])

                    label = result.names[cls]

                    confidence = float(box.conf[0])

                    st.write(f"### 🔹 {label.upper()}")

                    st.write(f"**Confidence:** {confidence:.2%}")

                    # Risk Level
                    if confidence >= 0.90:
                        risk = "🔴 HIGH"

                    elif confidence >= 0.70:
                        risk = "🟠 MEDIUM"

                    else:
                        risk = "🟢 LOW"

                    st.write(f"**Risk Level:** {risk}")

                    # Recommendation
                    st.write("### 💡 Recommendation")

                    if label.lower() == "fire":

                        st.error("""
- Evacuate the area immediately.
- Call the Fire Department.
- Do not attempt to extinguish large fires yourself.
""")

                    elif label.lower() == "smoke":

                        st.warning("""
- Check the area for possible fire.
- Ensure proper ventilation.
- Monitor the surroundings carefully.
""")

st.divider()

st.caption("Developed using YOLOv8 + Streamlit")
