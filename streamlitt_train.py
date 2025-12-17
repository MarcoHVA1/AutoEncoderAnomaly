# streamlit_app.py
import streamlit as st
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==============================
# Config
# ==============================
SAVE_DIR = "models/saved_models"

# ==============================
# Model laden
# ==============================
@st.cache_resource
def load_model():
    ae_path = os.path.join(SAVE_DIR, "autoencoder.pkl")
    scaler_path = os.path.join(SAVE_DIR, "scaler.pkl")
    threshold_path = os.path.join(SAVE_DIR, "threshold.npy")
    metrics_path = os.path.join(SAVE_DIR, "metrics.json")

    if not all(os.path.exists(p) for p in [ae_path, scaler_path, threshold_path, metrics_path]):
        st.error("Modelbestanden ontbreken! Train eerst het model met train.py")
        return None, None, None, None

    autoencoder = joblib.load(ae_path)
    scaler = joblib.load(scaler_path)
    threshold = np.load(threshold_path)
    with open(metrics_path) as f:
        metrics = json.load(f)

    return autoencoder, scaler, threshold, metrics

autoencoder, scaler, threshold, metrics = load_model()
if autoencoder is None:
    st.stop()

# ==============================
# Sidebar
# ==============================
st.sidebar.title("Autoencoder Anomaly Detection")
st.sidebar.write("Upload een afbeelding of gebruik de testset")

# ==============================
# Testset laden
# ==============================
import cv2

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64,64))
    img = img.astype("float32") / 255.0
    return img.flatten()

TEST_DIR = os.path.join("dataset","test","normal")
TEST_ANOMALY_DIR = os.path.join("dataset","test","anomaly")

test_files_normal = [os.path.join(TEST_DIR,f) for f in os.listdir(TEST_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))]
test_files_anomaly = [os.path.join(TEST_ANOMALY_DIR,f) for f in os.listdir(TEST_ANOMALY_DIR) if f.lower().endswith((".png",".jpg",".jpeg"))]

test_files = test_files_normal + test_files_anomaly
y_test = np.array([0]*len(test_files_normal) + [1]*len(test_files_anomaly))

# ==============================
# Afbeelding selecteren
# ==============================
selected_file = st.selectbox("Selecteer een testbeeld", test_files)
st.image(cv2.cvtColor(cv2.imread(selected_file), cv2.COLOR_BGR2RGB), caption="Geselecteerde afbeelding", use_column_width=True)

# ==============================
# Detectie
# ==============================
x = preprocess_image(selected_file)
x_scaled = scaler.transform(x.reshape(1,-1))
recon = autoencoder.predict(x_scaled)
error = np.mean((x_scaled - recon)**2)
is_anomaly = error > threshold

st.write(f"Reconstructie fout: {error:.6f}")
st.write("⚠️ Anomaly gedetecteerd!" if is_anomaly else "✅ Normal")

# ==============================
# Confusion matrix
# ==============================
st.subheader("Confusion Matrix op testset")
X_test_scaled = scaler.transform(np.array([preprocess_image(f) for f in test_files]))
test_pred = autoencoder.predict(X_test_scaled)
test_errors = np.mean((X_test_scaled - test_pred)**2, axis=1)
y_pred = (test_errors > threshold).astype(int)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal","Anomaly"])
fig, ax = plt.subplots()
disp.plot(ax=ax)
st.pyplot(fig)

# ==============================
# Loss curve (approx per iteration)
# ==============================
if hasattr(autoencoder, 'loss_curve_'):
    st.subheader("Training loss per iteratie")
    fig2, ax2 = plt.subplots()
    ax2.plot(autoencoder.loss_curve_)
    ax2.set_xlabel("Iteratie")
    ax2.set_ylabel("Loss")
    st.pyplot(fig2)

# ==============================
# Belangrijke metrics
# ==============================
st.subheader("Belangrijke statistieken")
st.write(f"Accuracy: {metrics['accuracy']:.3f}")
st.write(f"Precision: {metrics['precision']:.3f}")
st.write(f"Recall: {metrics['recall']:.3f}")
st.write(f"F1-score: {metrics['f1']:.3f}")
st.write(f"MSE Normal: {metrics['mse_normal']:.6f}")
st.write(f"MSE Anomaly: {metrics['mse_anomaly']:.6f}")

st.markdown("""
**Tips voor een goed cijfer:**
- Train met genoeg normale beelden voor generalisatie
- Kies juiste hidden layer grootte (niet te groot voor GitHub)
- Controleer threshold (95e percentile of pas aan)
- Analyseer confusion matrix voor false positives/negatives
- Gebruik loss_curve_ om overfitting te vermijden
""")
