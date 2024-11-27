import os
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import tempfile

# Load models from local files
last_model_path = "models/last.pt"  # Replace with your local path
best_model_path = "models/best.pt"  # Replace with your local path

last_model = torch.load(last_model_path, map_location=torch.device('cpu'))
best_model = torch.load(best_model_path, map_location=torch.device('cpu'))

# Function to select model
def load_model(model_choice):
    if model_choice == "last.pt":
        return last_model
    elif model_choice == "best.pt":
        return best_model

# Streamlit app
st.title("Real-Time Pothole Detection")
st.write("Upload an image or video to detect potholes.")

# Model selection
model_choice = st.radio("Select Model", ("last.pt", "best.pt"))
model = load_model(model_choice)

# File uploader
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "mp4"])

def detect_and_display(image, model):
    # Ensure image is in the correct format for model input
    results = model(image)
    st.image(np.squeeze(results.render()), caption="Detection Result", use_column_width=True)
    pothole_detected = any(
        'pothole' in result['name'].lower() for result in results.pandas().xyxy[0].to_dict(orient="records")
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
            st.warning("⚠️ Pothole Detected! Alerting...")

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
            stframe.image(np.squeeze(results.render()), channels="RGB")

            pothole_detected = pothole_detected or any(
                'pothole' in result['name'].lower() for result in results.pandas().xyxy[0].to_dict(orient="records")
            )

        cap.release()
        tfile.close()
        os.unlink(tfile.name)

        if pothole_detected:
            st.warning("⚠️ Pothole(s) Detected in the Video! Alerting...")

# Footer
st.write("Developed by Team Pothole")
