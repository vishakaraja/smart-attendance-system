# Smart Attendance System with Drowsiness Detection

## Overview
This project automates attendance using face recognition and also detects driver drowsiness using Eye Aspect Ratio (EAR).

## Features
- Face detection and recognition using OpenCV
- Automatic attendance marking with timestamp
- Drowsiness detection using EAR
- CSV output generation

## Technologies Used
- Python
- OpenCV
- face_recognition
- dlib
- NumPy, Pandas

## Project Structure
- encode_faces.py → encodes faces from dataset
- attendance_behavior.py → runs webcam + attendance + drowsiness
- encodings.pickle → stored face encodings

## How to Run

### Step 1: Install dependencies# Smart Attendance System with Drowsiness Detection

## Overview
This project automates attendance using face recognition and also detects driver drowsiness using Eye Aspect Ratio (EAR).

## Features
- Face detection and recognition using OpenCV
- Automatic attendance marking with timestamp
- Drowsiness detection using EAR
- CSV output generation

## Technologies Used
- Python
- OpenCV
- face_recognition
- dlib
- NumPy, Pandas

## Project Structure
- encode_faces.py → encodes faces from dataset
- attendance_behavior.py → runs webcam + attendance + drowsiness
- encodings.pickle → stored face encodings
pip install -r requirements.txt


### Step 2: Encode faces

python encode_faces.py


### Step 3: Run system

python attendance_behavior.py

## Output
- Webcam window opens
- Face is detected and name displayed
- Attendance saved in CSV file

## Note
Download dlib model:
https://github.com/davisking/dlib-models

## Future Improvements
- Web interface
- Real-time database
- Cloud deployment
