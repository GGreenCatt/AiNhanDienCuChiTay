# Train ASL Model Using Hand Landmarks (42D vector per sample)

import os
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import joblib

# Configuration
DATA_DIR = "landmark_data"  # Folder containing .npz files with 'landmarks' and 'labels'
MODEL_PATH = "asl_landmark_model.h5"
LABELS_PATH = "label_map_landmark.json"
SCALER_PATH = "scaler_landmark.pkl"

# Load dataset from .npz file
def load_data():
    data = np.load(os.path.join(DATA_DIR, "dataset.npz"))
    X = data['landmarks']  # shape: (num_samples, 42)
    y = data['labels']     # shape: (num_samples,)
    label_set = sorted(list(set(y)))
    label_map = {label: idx for idx, label in enumerate(label_set)}
    reverse_map = {v: k for k, v in label_map.items()}

    y_encoded = to_categorical([label_map[label] for label in y], num_classes=len(label_set))
    return X, y_encoded, reverse_map

# Build simple dense neural network
def build_model(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Training routine
def train():
    X, y, label_map = load_data()

    # Optional: normalize (0–1) or standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = build_model(input_dim=X.shape[1], num_classes=y.shape[1])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    model.save(MODEL_PATH)
    with open(LABELS_PATH, 'w') as f:
        json.dump(label_map, f)
    joblib.dump(scaler, SCALER_PATH)
    print("Model, labels, and scaler saved.")

if __name__ == "__main__":
    train()
