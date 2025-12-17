import os
import cv2
import numpy as np
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# CONFIG
IMG_SIZE = 64
DATASET_DIR = "dataset"
MODEL_PATH = "autoencoder.pkl"
SCALER_PATH = "scaler.pkl"

RANDOM_STATE = 42

# IMAGE LOADING & PREPROCESSING
def load_image(path):
    """
    Load image, resize, normalize and flatten
    """
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return img.flatten()


def load_images_from_folder(folder):
    images = []
    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            images.append(load_image(os.path.join(folder, file)))
    return np.array(images)


# LOAD DATASET
print("[INFO] Loading dataset...")

X_train = load_images_from_folder(os.path.join(DATASET_DIR, "train", "normal"))
X_val   = load_images_from_folder(os.path.join(DATASET_DIR, "val", "normal"))

X_test_normal  = load_images_from_folder(os.path.join(DATASET_DIR, "test", "normal"))
X_test_anomaly = load_images_from_folder(os.path.join(DATASET_DIR, "test", "anomaly"))

X_test = np.vstack((X_test_normal, X_test_anomaly))
y_test = np.array(
    [0] * len(X_test_normal) + [1] * len(X_test_anomaly)
)  # 0 = normal, 1 = anomaly


# FEATURE SCALING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# AUTOENCODER MODEL
print("[INFO] Training autoencoder...")

autoencoder = MLPRegressor(
    hidden_layer_sizes=(512, 128, 512),
    activation="relu",
    solver="adam",
    max_iter=120,
    random_state=RANDOM_STATE,
    verbose=True
)

autoencoder.fit(X_train, X_train)

# RECONSTRUCTION ERROR
print("[INFO] Calculating reconstruction errors...")

X_train_pred = autoencoder.predict(X_train)
train_errors = np.mean((X_train - X_train_pred) ** 2, axis=1)

X_test_pred = autoencoder.predict(X_test)
test_errors = np.mean((X_test - X_test_pred) ** 2, axis=1)

# THRESHOLD (95 PERCENTILE)
threshold = np.percentile(train_errors, 95)

y_pred = (test_errors > threshold).astype(int)


# METRICS
mse  = mean_squared_error(X_test, X_test_pred)
rmse = np.sqrt(mse)

print("\n===== AUTOENCODER RESULTS =====")
print(f"MSE:  {mse:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"Threshold: {threshold:.6f}")
print(f"Detected anomalies: {np.sum(y_pred)} / {len(y_pred)}")


# K-MEANS COMPARISON
print("\n[INFO] Running K-Means comparison...")

kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
kmeans_labels = kmeans.fit_predict(X_test)


# DBSCAN COMPARISON
print("[INFO] Running DBSCAN comparison...")

dbscan = DBSCAN(eps=5, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_test)


# Print clustering results
print("\n[INFO] Saving model and scaler...")

joblib.dump(autoencoder, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("[DONE] Model training completed.")
