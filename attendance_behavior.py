"""
attendance_behavior.py
Webcam face recognition + drowsiness (EAR) check + attendance CSV + tabular output
Author: Vishaka
"""

import cv2
import face_recognition
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime
import dlib
from tabulate import tabulate

# Paths
ENCODINGS_PATH = "encodings/encodings.pickle"
LANDMARK_MODEL = "models/shape_predictor_68_face_landmarks.dat"
OUTPUT_CSV = "outputs/attendance.csv"

# Make sure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load face encodings
with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

# Dlib predictor for landmarks
predictor = dlib.shape_predictor(LANDMARK_MODEL)

# Function to calculate EAR (Eye Aspect Ratio)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)

LEFT_EYE_IDX = list(range(42, 48))
RIGHT_EYE_IDX = list(range(36, 42))

attendance = {}
blink_counters = {}
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3
FRAME_RESIZE = 0.5

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing
    small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE, fy=FRAME_RESIZE)
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings
    boxes = face_recognition.face_locations(rgb_small)
    encs = face_recognition.face_encodings(rgb_small, boxes)

    for (box, enc) in zip(boxes, encs):
        matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
        name = "Unknown"
        if True in matches:
            first = matches.index(True)
            name = known_names[first]

        # Scale box back to original frame
        top, right, bottom, left = box
        top = int(top / FRAME_RESIZE)
        right = int(right / FRAME_RESIZE)
        bottom = int(bottom / FRAME_RESIZE)
        left = int(left / FRAME_RESIZE)

        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert to grayscale for dlib predictor
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dlib_rect = dlib.rectangle(left, top, right, bottom)
        try:
            shape = predictor(gray, dlib_rect)
            coords = np.zeros((68, 2), dtype=int)
            for i in range(68):
                coords[i] = (shape.part(i).x, shape.part(i).y)

            leftEye = coords[LEFT_EYE_IDX]
            rightEye = coords[RIGHT_EYE_IDX]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            cv2.putText(frame, f"EAR:{ear:.2f}", (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if ear < EAR_THRESHOLD:
                blink_counters[name] = blink_counters.get(name, 0) + 1
            else:
                blink_counters[name] = 0

            if blink_counters.get(name, 0) >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSY", (left, bottom + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                behavior = "Drowsy"
            else:
                behavior = "Attentive"
        except Exception:
            behavior = "Unknown"

        # Mark attendance
        today = datetime.now().strftime("%Y-%m-%d")
        key = (name, today)
        if name != "Unknown" and key not in attendance:
            attendance[key] = {
                "name": name,
                "date": today,
                "time": datetime.now().strftime("%H:%M:%S"),
                "behavior": behavior
            }
            print(f"[ATTEND] {name} marked present at {attendance[key]['time']} ({behavior})")

    # Show webcam feed
    cv2.imshow("Smart Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything
cap.release()
cv2.destroyAllWindows()

# Save and display attendance
rows = list(attendance.values())
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n✅ Attendance Summary:\n")
    print(tabulate(df, headers="keys", tablefmt="grid"))
    print(f"\n[DONE] Attendance saved to {OUTPUT_CSV}")
else:
    print("[DONE] No attendance recorded.")

