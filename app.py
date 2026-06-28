import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import numpy as np
import cv2
import os
from collections import Counter

# ----------------------------------
# Page Configuration
# ----------------------------------

st.set_page_config(
    page_title="🔥 FireVision AI",
    page_icon="🔥",
    layout="wide"
)

# ----------------------------------
# Load Model
# ----------------------------------

@st.cache_resource
def load_model():
    return YOLO("model.pt")

model = load_model()

# ----------------------------------
# Header
# ----------------------------------

st.markdown("""
# 🔥 FireVision AI

### Intelligent Fire & Smoke Detection using YOLOv8
""")

st.success("🟢 AI Model Loaded Successfully")

st.divider()

# ----------------------------------
# Sidebar
# ----------------------------------

with st.sidebar:

    st.title("⚙ System")

    st.markdown("---")

    st.info("""
This application detects

🔥 Fire

💨 Smoke

from uploaded images using
a trained YOLOv8 model.
""")

    st.markdown("---")

    st.write("Model")
    st.code("YOLOv8")

    st.write("Framework")
    st.code("Ultralytics")

    st.write("Status")
    st.success("Ready")

# ----------------------------------
# Upload Image
# ----------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📷 Original Image")

        st.image(
            image,
            use_container_width=True
        )

    detect = st.button(
        "🔥 Detect Fire / Smoke",
        use_container_width=True
    )

    if detect:

        with st.spinner("Analyzing image..."):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".jpg"
            ) as tmp:

                image.save(tmp.name)

                image_path = tmp.name

            results = model.predict(
                source=image_path,
                conf=0.5
            )

            result = results[0]

            plotted = result.plot()

            plotted = cv2.cvtColor(
                plotted,
                cv2.COLOR_BGR2RGB
            )

            with col2:

                st.subheader("🤖 Detection Result")

                st.image(
                    plotted,
                    use_container_width=True
                )

            st.divider()

            # -----------------------------
            # Collect detections
            # -----------------------------

            detected_classes = []

            highest_conf = 0

            highest_label = None

            for box in result.boxes:

                cls = int(box.cls[0])

                label = result.names[cls]

                confidence = float(box.conf[0])

                detected_classes.append(label)

                if confidence > highest_conf:

                    highest_conf = confidence

                    highest_label = label

            counter = Counter(detected_classes)
                        # ------------------------------------------
            # No Detection
            # ------------------------------------------

            if len(detected_classes) == 0:

                st.success("✅ No Fire or Smoke Detected")

                st.info("""
The uploaded image does not appear to contain
Fire or Smoke.

The environment looks safe.
""")

            else:

                st.subheader("📊 AI Detection Summary")

                colA, colB, colC = st.columns(3)

                with colA:

                    st.metric(
                        "Objects Detected",
                        len(detected_classes)
                    )

                with colB:

                    st.metric(
                        "Highest Confidence",
                        f"{highest_conf:.2%}"
                    )

                with colC:

                    if highest_label:

                        st.metric(
                            "Primary Detection",
                            highest_label.upper()
                        )

                st.divider()

                st.subheader("🔥 Detected Objects")

                for label, count in counter.items():

                    emoji = "🔥" if label.lower() == "fire" else "💨"

                    st.write(f"{emoji} **{label.title()}** : {count}")

                st.divider()

                # -----------------------------
                # Risk Analysis
                # -----------------------------

                st.subheader("🚨 Risk Assessment")

                if highest_conf >= 0.90:

                    st.error("🔴 HIGH RISK")

                elif highest_conf >= 0.70:

                    st.warning("🟠 MEDIUM RISK")

                else:

                    st.info("🟢 LOW RISK")

                st.progress(int(highest_conf * 100))

                st.write(f"Confidence Score : **{highest_conf:.2%}**")

                st.divider()

                # -----------------------------
                # Recommendation
                # -----------------------------

                st.subheader("💡 AI Recommendation")

                if "fire" in [x.lower() for x in detected_classes]:

                    st.error("""
### 🔥 Fire Detected

Immediate Actions

• Evacuate everyone from the area.

• Contact the Fire Department immediately.

• Do NOT attempt to extinguish large fires.

• Switch off electrical power if it is safe.

• Keep away from hazardous materials.
""")

                elif "smoke" in [x.lower() for x in detected_classes]:

                    st.warning("""
### 💨 Smoke Detected

Suggested Actions

• Inspect the area carefully.

• Check for possible fire sources.

• Improve ventilation if safe.

• Continue monitoring the surroundings.
""")

                else:

                    st.success("""
### ✅ Environment Safe

No fire or smoke detected.
""")

                st.divider()

                # -----------------------------
                # Download Result
                # -----------------------------

                success, encoded = cv2.imencode(".jpg", cv2.cvtColor(plotted, cv2.COLOR_RGB2BGR))

                if success:
                    st.download_button(
                        label="📥 Download Detection Result",
                        data=encoded.tobytes(),
                        file_name="fire_detection_result.jpg",
                        mime="image/jpeg",
                        use_container_width=True
                    )

            os.remove(image_path)

# ------------------------------------------
# Footer
# ------------------------------------------

st.divider()

st.markdown("### 📌 About")

st.info("""
FireVision AI is an intelligent Fire & Smoke Detection System developed using:

• YOLOv8 Object Detection
• Streamlit
• PyTorch
• OpenCV

This project is intended for educational and demonstration purposes.
""")

st.caption("© 2026 FireVision AI | Powered by YOLOv8 & Streamlit")
