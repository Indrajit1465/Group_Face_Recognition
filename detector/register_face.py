import cv2
import mediapipe as mp
import numpy as np
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_embeddings, save_embeddings

mp_face = mp.solutions.face_detection.FaceDetection(0, 0.6)
cap = cv2.VideoCapture(0)

student_id = input("Enter Employee ID: ")
data = load_embeddings()

os.makedirs("temp_faces", exist_ok=True)
embeddings = []
count = 0

while count < 15:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin*w), int(bbox.ymin*h)
            bw, bh = int(bbox.width*w), int(bbox.height*h)

            face = frame[y:y+bh, x:x+bw]
            if face.size == 0:
                continue

            face = cv2.resize(face, (112,112))
            temp_path = f"temp_faces/{student_id}_{count}.jpg"
            cv2.imwrite(temp_path, face)
            count += 1
