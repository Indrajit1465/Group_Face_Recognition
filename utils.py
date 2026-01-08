import os
import pickle
import numpy as np

EMBEDDING_FILE = "embeddings/faces.pkl"

def load_embeddings():
    if os.path.exists(EMBEDDING_FILE):
        with open(EMBEDDING_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_embeddings(data):
    os.makedirs("embeddings", exist_ok=True)
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(data, f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
