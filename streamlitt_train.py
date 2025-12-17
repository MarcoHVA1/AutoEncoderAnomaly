import streamlit as st
import cv2
import numpy as np
import joblib
import os

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 64
MODEL_PATH = "autoencoder.pkl"
SCALER_PATH = "scaler.pkl"
THRESHOLD = 0.02  # <-- zet hier je berekende threshold

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

autoencoder, scaler = load_model()

# ==============================
# IMAGE FUNCTIONS
# ==============================
def preprocess_image(image):
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = img.flatten().reshape(1, -1)
    img = scaler.transform(img)
    return img


def pixelate(img, blocks=16):
    h, w = img.shape[:2]
    temp = cv2.resize(img, (blocks, blocks))
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


# ==============================
# STREAMLIT UI
# ==============================
st.set_page_config(page_title="Anomaly Detection", layout="centered")

st.title("🧠 Anomaly Detection Dashboard")
st.write("Upload een afbeelding om te controleren of deze **normaal** of een **anomaly** is.")

uploaded_file = st.file_uploader("Upload afbeelding", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("Originele afbeelding")
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Predict
    x = preprocess_image(image)
    x_hat = autoencoder.predict(x)

    error = np.mean((x - x_hat) ** 2)
    label = "Anomaly 🚨" if error > THRESHOLD else "Normal ✅"

    st.subheader("Resultaat")
    st.write(f"**Classificatie:** {label}")
    st.write(f"**Reconstructiefout (MSE):** {error:.6f}")
    st.write(f"**Threshold:** {THRESHOLD}")

    # Blur option
    if error > THRESHOLD:
        if st.button("🔒 Blur anomaly"):
            blurred = pixelate(image)
            st.subheader("Vervaagde afbeelding")
            st.image(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB), use_container_width=True)
