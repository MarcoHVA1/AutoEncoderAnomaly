# app.py
import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image

# ==============================
# Config
# ==============================
IMG_SIZE = 64
SAVE_DIR = "models/saved_models"

# ==============================
# Load model & scaler & threshold
# ==============================
@st.cache_resource
def load_model_scaler():
    autoencoder = joblib.load(os.path.join(SAVE_DIR, "autoencoder.pkl"))
    scaler = joblib.load(os.path.join(SAVE_DIR, "scaler.pkl"))
    threshold = np.load(os.path.join(SAVE_DIR, "threshold.npy"))
    return autoencoder, scaler, threshold

autoencoder, scaler, threshold = load_model_scaler()

# ==============================
# Helpers
# ==============================
def preprocess_image(image: Image.Image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).astype(np.float32) / 255.0
    return img.flatten()

def detect_anomaly(image_array: np.ndarray):
    img_scaled = scaler.transform(image_array.reshape(1, -1))
    recon = autoencoder.predict(img_scaled)
    error = np.mean((img_scaled - recon)**2)
    is_anomaly = error > threshold
    return error, is_anomaly

# ==============================
# Streamlit UI
# ==============================
st.title("Autoencoder Anomaly Detection")

st.write("Upload een afbeelding of kies een voorbeeld uit de testset.")

# Upload
uploaded_file = st.file_uploader("Upload een afbeelding", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    img_array = preprocess_image(img)
    error, is_anomaly = detect_anomaly(img_array)
    
    st.write(f"Reconstructie fout: {error:.6f}")
    st.write("⚠️ **Anomaly gedetecteerd!**" if is_anomaly else "✅ Normal")

# Optioneel: kies een testmap voorbeeld
if st.checkbox("Gebruik voorbeeld uit testset"):
    TEST_DIR = os.path.join("dataset", "test", "normal")
    test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))]
    selected_file = st.selectbox("Kies een testbeeld", test_files)
    if selected_file:
        img_path = os.path.join(TEST_DIR, selected_file)
        img = Image.open(img_path).convert("RGB")
        st.image(img, caption="Test Image", use_column_width=True)
        
        img_array = preprocess_image(img)
        error, is_anomaly = detect_anomaly(img_array)
        
        st.write(f"Reconstructie fout: {error:.6f}")
        st.write("⚠️ **Anomaly gedetecteerd!**" if is_anomaly else "✅ Normal")
