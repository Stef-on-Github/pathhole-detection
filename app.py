import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load models using ultralytics
last_model_path = "models/last.pt"  # Replace with your local path
best_model_path = "models/best.pt"  # Replace with your local path

# Streamlit UI for model selection
st.title("Real-Time Pothole Detection")
st.write("Upload an image or video to detect potholes.")

model_choice = st.radio("Select Model", ("last.pt", "best.pt"))
model_path = last_model_path if model_choice == "last.pt" else best_model_path
model = YOLO(model_path)  # Load YOLO model

# File uploader
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "mp4"])

def detect_and_display(image, model):
    results = model(image)
    st.image(np.squeeze(results[0].plot()), caption="Detection Result", use_column_width=True)
    pothole_detected = any(
        'pothole' in result['name'].lower() for result in results[0].boxes.data.tolist()
    )
    return pothole_detected

if uploaded_file:
    if uploaded_file.type.startswith("image"):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pothole_detected = detect_and_display(img_rgb, model)
        if pothole_detected:
            st.warning("⚠️ Pothole Detected! Alerting...")

    elif uploaded_file.type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        pothole_detected = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            stframe.image(np.squeeze(results[0].plot()), channels="RGB")

            pothole_detected = pothole_detected or any(
                'pothole' in result['name'].lower() for result in results[0].boxes.data.tolist()
            )

        cap.release()
        tfile.close()
        os.unlink(tfile.name)

        if pothole_detected:
            st.warning("⚠️ Pothole(s) Detected in the Video! Alerting...")

st.write("Developed by Team Pothole")
