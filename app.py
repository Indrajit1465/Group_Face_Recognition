import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import requests
import pandas as pd
from datetime import datetime
import shutil
import time
import json
import subprocess

# --------------------------------------------------
# Config
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATT_DIR = os.path.join(BASE_DIR, "attendance")
ATT_FILE = os.path.join(ATT_DIR, "attendance.xlsx")
TEMP_FACE_DIR = os.path.join(BASE_DIR, "detector", "temp_faces")

os.makedirs(ATT_DIR, exist_ok=True)
os.makedirs(TEMP_FACE_DIR, exist_ok=True)

MIN_CHECKOUT_SECONDS = 60 * 30
MAX_REG_IMAGES = 15

# --------------------------------------------------
# Helpers
# --------------------------------------------------
COLUMNS = ["Employee ID", "Date", "Check-In", "Check-Out", "Duration"]

def load_attendance_df():
    if os.path.exists(ATT_FILE) and os.path.getsize(ATT_FILE) > 0:
        return pd.read_excel(ATT_FILE, engine="openpyxl")
    return pd.DataFrame(columns=COLUMNS)

def save_face_image(emp_id, face_img):
    emp_dir = os.path.join(TEMP_FACE_DIR, emp_id)
    os.makedirs(emp_dir, exist_ok=True)
    count = len(os.listdir(emp_dir))
    if count < MAX_REG_IMAGES:
        cv2.imwrite(os.path.join(emp_dir, f"{count}.jpg"), face_img)
    return count + 1

def clear_temp_faces():
    if os.path.exists(TEMP_FACE_DIR):
        shutil.rmtree(TEMP_FACE_DIR)
        os.makedirs(TEMP_FACE_DIR, exist_ok=True)

# --------------------------------------------------
# MediaPipe
# --------------------------------------------------
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Group Face Attendance", layout="wide")
st.title("👥 Group Face Attendance System")

menu = st.sidebar.radio(
    "Select Action",
    ["Register Face", "Generate Embeddings", "Mark Attendance"]
)

# --------------------------------------------------
# PAGE 1: FACE REGISTRATION
# --------------------------------------------------
if menu == "Register Face":
    st.subheader("🧑 Face Registration")

    emp_id = st.text_input("Enter Employee ID")

    if "run_register" not in st.session_state:
        st.session_state.run_register = False

    col1, col2 = st.columns(2)

    with col1:
        start = st.button("▶ Start Capture")
        stop = st.button("⏹ Stop Capture")
        frame_placeholder = st.empty()

    with col2:
        st.markdown("### 📊 Capture Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()

    if start and emp_id:
        st.session_state.run_register = True

    if stop:
        st.session_state.run_register = False

    if st.session_state.run_register and emp_id:
        cap = cv2.VideoCapture(0)
        os.makedirs(os.path.join(TEMP_FACE_DIR, emp_id), exist_ok=True)

        while st.session_state.run_register:
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

                    count = save_face_image(emp_id, face)
                    progress = min(count / MAX_REG_IMAGES, 1.0)

                    progress_bar.progress(progress)
                    status_text.success(f"Captured {count}/{MAX_REG_IMAGES}")

                    cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
                    cv2.putText(frame, f"{int(progress*100)}%",
                                (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                    if count >= MAX_REG_IMAGES:
                        st.session_state.run_register = False
                        break

            frame_placeholder.image(frame, channels="BGR")
            time.sleep(0.1)

        cap.release()
        progress_bar.progress(1.0)
        status_text.success("✅ Face registration completed!")

# --------------------------------------------------
# PAGE 2: GENERATE EMBEDDINGS (WITH PROGRESS BAR)
# --------------------------------------------------
elif menu == "Generate Embeddings":
    st.subheader("🧠 Generate Face Embeddings")

    registered_ids = [
        d for d in os.listdir(TEMP_FACE_DIR)
        if os.path.isdir(os.path.join(TEMP_FACE_DIR, d))
    ]

    if not registered_ids:
        st.warning("No registered faces found.")
    else:
        st.code(", ".join(registered_ids))

    if st.button("Generate Embeddings"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        cmd = [
            "cmd", "/c",
            "cd recognizer && fr_env\\Scripts\\activate && python generate_embeddings.py"
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        total = None

        for line in process.stdout:
            try:
                msg = json.loads(line.strip())
            except Exception:
                continue

            if msg["type"] == "init":
                total = msg["total"]
                status_text.info(f"Processing {total} users")

            elif msg["type"] == "progress":
                done = msg["processed"]
                emp = msg["emp_id"]
                progress_bar.progress(done / total)
                status_text.success(f"{emp} processed ({done}/{total})")

            elif msg["type"] == "done":
                progress_bar.progress(1.0)
                status_text.success("✅ Embeddings generated successfully!")
                break

        clear_temp_faces()

# --------------------------------------------------
# PAGE 3: MARK ATTENDANCE
# --------------------------------------------------
elif menu == "Mark Attendance":
    st.subheader("📸 Live Attendance")

    col1, col2 = st.columns([2, 1])

    if "run_attendance" not in st.session_state:
        st.session_state.run_attendance = False

    with col1:
        start_btn = st.button("▶ Start Attendance")
        stop_btn = st.button("⏹ Stop Attendance")
        frame_placeholder = st.empty()

    with col2:
        st.markdown("### 📋 Attendance Sheet")
        table_placeholder = st.empty()

    if start_btn:
        st.session_state.run_attendance = True

    if stop_btn:
        st.session_state.run_attendance = False

    if st.session_state.run_attendance:
        cap = cv2.VideoCapture(0)
        assigned_ids = set()

        while st.session_state.run_attendance:
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

                    _, img_encoded = cv2.imencode(".jpg", face)

                    response = requests.post(
                        "http://127.0.0.1:8000/recognize",
                        files={"face": img_encoded.tobytes()},
                        timeout=5
                    ).json()

                    emp_id = response.get("student_id")
                    if emp_id is None or emp_id in assigned_ids:
                        continue

                    assigned_ids.add(emp_id)

                    today = datetime.now().strftime("%Y-%m-%d")
                    now = datetime.now().strftime("%H:%M:%S")

                    df = load_attendance_df()
                    records = df[(df["Employee ID"] == emp_id) & (df["Date"] == today)]

                    if records.empty:
                        df.loc[len(df)] = [emp_id, today, now, "", ""]
                        message = "Check-In Marked"
                    else:
                        idx = records.index[0]
                        if df.at[idx, "Check-Out"] == "":
                            check_in = datetime.strptime(df.at[idx, "Check-In"], "%H:%M:%S")
                            curr = datetime.strptime(now, "%H:%M:%S")

                            if (curr - check_in).total_seconds() >= MIN_CHECKOUT_SECONDS:
                                df.at[idx, "Check-Out"] = now
                                df.at[idx, "Duration"] = str(curr - check_in)
                                message = "Check-Out Marked"
                            else:
                                message = "Checked-In"
                        else:
                            message = "Already Checked-Out"

                    df.to_excel(ATT_FILE, index=False)

                    cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
                    cv2.putText(frame, emp_id, (x,y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    cv2.putText(frame, message, (x,y+bh+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            frame_placeholder.image(frame, channels="BGR")
            table_placeholder.dataframe(load_attendance_df(), use_container_width=True)
            time.sleep(0.05)

        cap.release()
