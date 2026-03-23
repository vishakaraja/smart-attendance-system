"""
encode_faces.py
Simple script to read dataset/<person> images and create encodings.pickle
Author: Vishaka
"""

import os
import face_recognition
import pickle

DATASET_DIR = "dataset"
OUTPUT_DIR = "encodings"
OUT_FILE = os.path.join(OUTPUT_DIR, "encodings.pickle")

os.makedirs(OUTPUT_DIR, exist_ok=True)

known_encodings = []
known_names = []

# iterate each person folder
for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Processing {person_name}")
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        # load image and convert to RGB
        image = face_recognition.load_image_file(img_path)
        # detect face encodings (returns list; we take first face)
        encs = face_recognition.face_encodings(image)
        if len(encs) == 0:
            print(f"  [WARN] No face found in {img_path}, skip.")
            continue
        encoding = encs[0]
        known_encodings.append(encoding)
        known_names.append(person_name)

# save to pickle
with open(OUT_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"[DONE] Saved {len(known_encodings)} encodings to {OUT_FILE}")
