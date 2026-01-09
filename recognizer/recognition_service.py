from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
import cv2
from deepface import DeepFace

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_FILE = os.path.join(BASE_DIR, "..", "embeddings", "faces.pkl")

# --------------------------------------------------
# Utils
# --------------------------------------------------
def load_embeddings():
    if os.path.exists(EMBED_FILE) and os.path.getsize(EMBED_FILE) > 0:
        with open(EMBED_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --------------------------------------------------
# API
# --------------------------------------------------
@app.route("/recognize", methods=["POST"])
def recognize():
    # 🔥 RELOAD embeddings every request
    embeddings_db = load_embeddings()

    if not embeddings_db:
        return jsonify({"student_id": None, "confidence": 0.0})

    file = request.files["face"]
    img_bytes = file.read()

    img_array = np.frombuffer(img_bytes, np.uint8)
    face = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    emb = DeepFace.represent(
        img_path=face,
        model_name="ArcFace",
        enforce_detection=False,
        detector_backend="skip"
    )[0]["embedding"]

    emb = np.array(emb)

    best_match = None
    best_score = 0.0

    for sid, db_emb in embeddings_db.items():
        score = cosine_similarity(emb, db_emb)
        if score > best_score:
            best_score = score
            best_match = sid

    if best_score > 0.7:
        return jsonify({
            "student_id": best_match,
            "confidence": float(best_score)
        })

    return jsonify({"student_id": None, "confidence": float(best_score)})

# --------------------------------------------------
if __name__ == "__main__":
    app.run(port=8000)
