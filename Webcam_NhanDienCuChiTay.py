# Real-time ASL prediction from hand landmarks using webcam

import cv2
import numpy as np
import mediapipe as mp
import json
import joblib
from tensorflow.keras.models import load_model

# Load model, labels, scaler
MODEL_PATH = "asl_landmark_model.h5"
LABEL_MAP_PATH = "label_map_landmark.json"
SCALER_PATH = "scaler_landmark.pkl"

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

with open(LABEL_MAP_PATH, 'r') as f:
    label_map = json.load(f)
    labels = [label_map[str(i)] for i in range(len(label_map))]

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Extract landmarks from frame
def get_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = []
        for pt in hand.landmark:
            lm.extend([pt.x, pt.y])
        return np.array(lm), hand
    return None, None

# Predict gesture
def predict(landmarks):
    landmarks_scaled = scaler.transform([landmarks])
    preds = model.predict(landmarks_scaled)[0]
    max_idx = np.argmax(preds)
    return labels[max_idx], np.max(preds)

# Webcam loop
cap = cv2.VideoCapture(0)
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks, hand_landmarks = get_landmarks(frame)
    if landmarks is not None and landmarks.shape[0] == 42:
        label, conf = predict(landmarks)
        cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No hand detected", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("ASL Landmark Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
