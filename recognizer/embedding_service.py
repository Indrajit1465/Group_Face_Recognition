import numpy as np
from deepface import DeepFace

def get_embedding(face_img):
    emb = DeepFace.represent(
        img_path=face_img,
        model_name="ArcFace",
        enforce_detection=False,
        detector_backend="skip"
    )
    return np.array(emb[0]["embedding"])
