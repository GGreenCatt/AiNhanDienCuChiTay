# Extract hand landmarks from ASL image dataset using MediaPipe

import os
import cv2
import numpy as np
import mediapipe as mp

# Configuration
DATASET_PATH = "asl_alphabet_train"  # root folder with A-Z subfolders
OUTPUT_PATH = "landmark_data/dataset.npz"
IMG_SIZE = 224  # resize image to this size (optional for uniformity)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def extract_landmarks_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y])  # Use only x and y
        return landmarks if len(landmarks) == 42 else None
    return None

def process_dataset():
    landmarks_list = []
    labels_list = []
    skipped = 0

    label_folders = sorted(os.listdir(DATASET_PATH))
    for label in label_folders:
        label_path = os.path.join(DATASET_PATH, label)
        if not os.path.isdir(label_path):
            continue

        print(f"Processing label: {label}")
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            landmarks = extract_landmarks_from_image(file_path)
            if landmarks:
                landmarks_list.append(landmarks)
                labels_list.append(label)
            else:
                skipped += 1

    print(f"Done. Total samples: {len(landmarks_list)}, Skipped: {skipped}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.savez(OUTPUT_PATH, landmarks=np.array(landmarks_list), labels=np.array(labels_list))
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_dataset()
