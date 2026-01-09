import os
import pickle
import numpy as np
from embedding_service import get_embedding

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

print("🔍 Scanning registered faces...")

# --------------------------------------------------
# Generate embeddings
# --------------------------------------------------
for emp_id in os.listdir(TEMP_FACE_DIR):
    emp_path = os.path.join(TEMP_FACE_DIR, emp_id)

    if not os.path.isdir(emp_path):
        continue

    print(f"🧑 Processing ID: {emp_id}")
    embeddings = []

    for img_name in os.listdir(emp_path):
        img_path = os.path.join(emp_path, img_name)
        try:
            emb = get_embedding(img_path)
            embeddings.append(emb)
        except Exception as e:
            print(f"⚠️ Skipped {img_name}: {e}")

    if embeddings:
        embeddings_db[emp_id] = np.mean(embeddings, axis=0)
        print(f"✅ Embedding saved for {emp_id}")
    else:
        print(f"❌ No valid faces for {emp_id}")

# --------------------------------------------------
# Save embeddings
# --------------------------------------------------
with open(EMBED_FILE, "wb") as f:
    pickle.dump(embeddings_db, f)

print("🎉 All embeddings generated successfully!")
