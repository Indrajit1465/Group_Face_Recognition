import os
import cv2
import numpy as np
import pickle
from embedding_service import get_embedding

TEMP_FACE_DIR = "../detector/temp_faces"
EMBEDDING_FILE = "../embeddings/faces.pkl"

os.makedirs("../embeddings", exist_ok=True)

# Load existing embeddings if present
if os.path.exists(EMBEDDING_FILE):
    with open(EMBEDDING_FILE, "rb") as f:
        data = pickle.load(f)
else:
    data = {}

student_embeddings = []

student_id = input("Enter Student ID for these faces: ")

for img_name in os.listdir(TEMP_FACE_DIR):
    img_path = os.path.join(TEMP_FACE_DIR, img_name)
    face = cv2.imread(img_path)

    if face is None:
        continue

    emb = get_embedding(face)
    student_embeddings.append(emb)

if len(student_embeddings) == 0:
    print("No faces found. Exiting.")
    exit()

avg_embedding = np.mean(student_embeddings, axis=0)
data[student_id] = avg_embedding

with open(EMBEDDING_FILE, "wb") as f:
    pickle.dump(data, f)

print(f"Embeddings saved successfully for student: {student_id}")
