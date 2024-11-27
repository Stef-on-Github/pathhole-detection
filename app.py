import streamlit as st
import torch
from PIL import Image
import requests
from io import BytesIO

# Load models
@st.cache_resource
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Pothole detection function
def detect_potholes(model, image):
    with torch.no_grad():
        result = model(image)
    return result

# Streamlit UI
st.title("Pothole Detection App")
st.sidebar.title("Select Model")
model_choice = st.sidebar.selectbox("Choose a model", ("last.pt", "best.pt"))

# Load selected model
model_path = f"models/{model_choice}"  # Assuming models are stored in ./models/
model = load_model(model_path)

# File upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Detecting potholes...")
    # Assuming model expects a tensor input
    input_image = torch.tensor(image).unsqueeze(0)  # Modify based on model requirement
    result = detect_potholes(model, input_image)

    # Display result
    st.write("Detection Result:")
    st.image(result, caption="Detection Output", use_column_width=True)  # Adjust as needed
