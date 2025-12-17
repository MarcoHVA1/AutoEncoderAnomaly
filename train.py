"""
Kleine autoencoder training voor anomaly detection (onder 25MB)
Train op TRAIN/normal.
Valideer op VAL/normal.
Test op TEST/normal + TEST/anomaly.
"""

import os
import numpy as np
import joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import json

# ==============================
# Config
# ==============================
IMG_SIZE = 64
DATASET_DIR = "dataset"
SAVE_DIR = "models/saved_models"
MAX_ITER = 80             # minder iteraties voor kleiner model
HIDDEN_LAYERS = (64, 16, 64)  # kleiner dan voorheen
THRESHOLD_PERCENTILE = 95
RANDOM_STATE = 42

# ==============================
# Data loading
# ==============================
def load_images_from_folder(folder):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Map bestaat niet: {folder}")
    images = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype("float32") / 255.0
        images.append(img.flatten())  # flatten voor MLP
    return np.array(images, dtype=np.float32)

# ==============================
# Main
# ==============================
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Train / Val / Test data laden ---
    X_train = load_images_from_folder(os.path.join(DATASET_DIR, "train", "normal"))
    X_val = load_images_from_folder(os.path.join(DATASET_DIR, "val", "normal"))
    X_test_normal = load_images_from_folder(os.path.join(DATASET_DIR, "test", "normal"))
    X_test_anomaly = load_images_from_folder(os.path.join(DATASET_DIR, "test", "anomaly"))

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.array([0]*len(X_test_normal) + [1]*len(X_test_anomaly))

    print(f"Train normal  : {len(X_train)}")
    print(f"Val normal    : {len(X_val)}")
    print(f"Test normal   : {len(X_test_normal)}")
    print(f"Test anomaly  : {len(X_test_anomaly)}")

    # --- Feature scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Kleine Autoencoder trainen ---
    autoencoder = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation='relu',
        solver='adam',
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        verbose=True
    )

    autoencoder.fit(X_train_scaled, X_train_scaled)

    # --- Threshold bepalen (val set) ---
    val_pred = autoencoder.predict(X_val_scaled)
    val_errors = np.mean((X_val_scaled - val_pred)**2, axis=1)
    threshold = np.percentile(val_errors, THRESHOLD_PERCENTILE)
    np.save(os.path.join(SAVE_DIR, "threshold.npy"), threshold)
    print(f"Threshold: {threshold:.6f}")

    # --- Test evaluatie ---
    test_pred = autoencoder.predict(X_test_scaled)
    test_errors = np.mean((X_test_scaled - test_pred)**2, axis=1)
    y_pred = (test_errors > threshold).astype(int)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:")
    print(cm)

    # --- Metrics opslaan ---
    accuracy = float((cm[0,0]+cm[1,1])/np.sum(cm))
    precision = float(cm[1,1]/max(cm[0,1]+cm[1,1],1))
    recall = float(cm[1,1]/max(cm[1,0]+cm[1,1],1))
    f1 = float(2*precision*recall/max(precision+recall,1e-8))

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mse_normal": float(test_errors[y_test==0].mean()),
        "mse_anomaly": float(test_errors[y_test==1].mean())
    }

    with open(os.path.join(SAVE_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Model & scaler opslaan ---
    joblib.dump(autoencoder, os.path.join(SAVE_DIR, "autoencoder.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

    print("Model en scaler opgeslagen in:", os.path.abspath(SAVE_DIR))
    print("Training afgerond.")

# ==============================
if __name__ == "__main__":
    main()
