import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Load the best.pt model
model_path = "models/best.pt"  # Replace with the actual path on your PC
model = YOLO(model_path)

# Streamlit app
st.title("Pothole Detection using YOLOv5")
st.write("Upload an image or video to detect potholes.")

# File uploader
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "mp4"])

def detect_and_display(image, model):
    results = model(image)
    st.image(np.squeeze(results[0].plot()), caption="Detection Result", use_column_width=True)
    
    # Check if any potholes were detected
    pothole_detected = any(
        'pothole' in result['name'].lower() for result in results[0].boxes.data.tolist()
    )
    return pothole_detected

if uploaded_file:
    # Process uploaded image
    if uploaded_file.type.startswith("image"):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect and display results
        pothole_detected = detect_and_display(img_rgb, model)
        
        if pothole_detected:
            st.warning("⚠️ Pothole Detected! Please take necessary action.")

    # Process uploaded video
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
            st.warning("⚠️ Pothole(s) Detected in the Video! Please take necessary action.")

# Footer
st.write("Developed by Team Pothole")
