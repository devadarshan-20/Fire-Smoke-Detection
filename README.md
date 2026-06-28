# 🔥 Fire & Smoke Detection using YOLOv8

## 📌 Overview

This project is an AI-powered Fire and Smoke Detection System developed using the YOLOv8 object detection model and Streamlit.

The application allows users to upload an image, detects the presence of fire or smoke, draws bounding boxes around detected objects, and displays the confidence score.

---

## 🚀 Features

- 🔥 Fire Detection
- 💨 Smoke Detection
- 📤 Upload an image
- 📦 Bounding box visualization
- 📊 Confidence score display
- ⚡ Fast YOLOv8 inference
- 🌐 Interactive Streamlit web interface

---

## 🛠 Technologies Used

- Python
- YOLOv8 (Ultralytics)
- Streamlit
- OpenCV
- Pillow
- PyTorch

---

## 📂 Project Structure

```
Fire-Smoke-Detection/
│
├── app.py
├── best.pt
├── requirements.txt
├── README.md
└── sample_images/
```

---

## ▶️ How to Run

1. Clone the repository.

2. Install the required libraries:

```bash
pip install -r requirements.txt
```

3. Start the Streamlit application:

```bash
streamlit run app.py
```

---

## 📸 Demo

Upload an image containing fire or smoke.

The application will:

- Detect fire and/or smoke
- Draw bounding boxes
- Display confidence scores
- Show the processed image

---

## 📈 Model

- Model: YOLOv8
- Framework: Ultralytics
- Classes:
  - Fire
  - Smoke

---

## 📄 License

This project is intended for educational and research purposes.
