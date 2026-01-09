import os
import pickle
import numpy as np
from embedding_service import get_embedding
import json
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_FACE_DIR = os.path.join(BASE_DIR, "..", "detector", "temp_faces")
EMBED_DIR = os.path.join(BASE_DIR, "..", "embeddings")
EMBED_FILE = os.path.join(EMBED_DIR, "faces.pkl")

os.makedirs(EMBED_DIR, exist_ok=True)

# --------------------------------------------------
# Load existing embeddings
# --------------------------------------------------
if os.path.exists(EMBED_FILE) and os.path.getsize(EMBED_FILE) > 0:
    with open(EMBED_FILE, "rb") as f:
        embeddings_db = pickle.load(f)
else:
    embeddings_db = {}

# --------------------------------------------------
# Scan users
# --------------------------------------------------
users = [
    d for d in os.listdir(TEMP_FACE_DIR)
    if os.path.isdir(os.path.join(TEMP_FACE_DIR, d))
]

total_users = len(users)
processed = 0

print(json.dumps({
    "type": "init",
    "total": total_users
}))
sys.stdout.flush()

# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------
for emp_id in users:
    emp_path = os.path.join(TEMP_FACE_DIR, emp_id)
    embeddings = []

    for img_name in os.listdir(emp_path):
        img_path = os.path.join(emp_path, img_name)
        try:
            emb = get_embedding(img_path)
            embeddings.append(emb)
        except Exception:
            pass

    if embeddings:
        embeddings_db[emp_id] = np.mean(embeddings, axis=0)

    processed += 1

    # 🔥 Emit progress
    print(json.dumps({
        "type": "progress",
        "processed": processed,
        "emp_id": emp_id
    }))
    sys.stdout.flush()

# --------------------------------------------------
# Save embeddings
# --------------------------------------------------
with open(EMBED_FILE, "wb") as f:
    pickle.dump(embeddings_db, f)

print(json.dumps({"type": "done"}))
sys.stdout.flush()
