from flask import Flask, request, jsonify
import numpy as np
import pickle
from deepface import DeepFace

app = Flask(__name__)

EMBED_FILE = "../embeddings/faces.pkl"

with open(EMBED_FILE, "rb") as f:
    EMBEDDINGS_DB = pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files["face"]
    img_bytes = file.read()

    import cv2
    import numpy as np

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

    for sid, db_emb in EMBEDDINGS_DB.items():
        score = cosine_similarity(emb, db_emb)
        if score > best_score:
            best_score = score
            best_match = sid

    if best_score > 0.7:
        return jsonify({
            "student_id": best_match,
            "confidence": float(best_score)
        })

    return jsonify({"student_id": None})

if __name__ == "__main__":
    app.run(port=8000)
