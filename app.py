import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import requests
import pandas as pd
from datetime import datetime

# --------------------------------------------------
# Config
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ATT_DIR = os.path.join(BASE_DIR, "attendance")
ATT_FILE = os.path.join(ATT_DIR, "attendance.xlsx")
TEMP_FACE_DIR = os.path.join(BASE_DIR, "detector", "temp_faces")

os.makedirs(ATT_DIR, exist_ok=True)
os.makedirs(TEMP_FACE_DIR, exist_ok=True)

MIN_CHECKOUT_SECONDS = 60 * 30  # 30 minutes

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
    path = os.path.join(emp_dir, f"{count}.jpg")
    cv2.imwrite(path, face_img)

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
    run = st.checkbox("Start Camera")
    frame_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    if run and emp_id:
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_face.process(rgb)

            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    h, w, _ = frame.shape

                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    bw, bh = int(bbox.width * w), int(bbox.height * h)

                    face = frame[y:y+bh, x:x+bw]
                    if face.size != 0:
                        save_face_image(emp_id, face)
                        cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)
                        cv2.putText(frame, "Face Captured", (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            frame_placeholder.image(frame, channels="BGR")

    cap.release()
    st.info("Capture 5–10 images, then go to 'Generate Embeddings'.")

# --------------------------------------------------
# PAGE 2: GENERATE EMBEDDINGS
# --------------------------------------------------
elif menu == "Generate Embeddings":
    st.subheader("🧠 Generate Face Embeddings")

    if st.button("Generate Embeddings"):
        os.system("cd recognizer && fr_env\\Scripts\\activate && python generate_embeddings.py")
        st.success("Embeddings generated successfully!")

# --------------------------------------------------
# PAGE 3: MARK ATTENDANCE
# --------------------------------------------------
elif menu == "Mark Attendance":
    st.subheader("📸 Live Attendance")

    col1, col2 = st.columns([2, 1])

    # Session state init
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

                    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                    bw, bh = int(bbox.width * w), int(bbox.height * h)

                    face = frame[y:y+bh, x:x+bw]
                    if face.size == 0:
                        continue

                    _, img_encoded = cv2.imencode(".jpg", face)

                    response = requests.post(
                        "http://127.0.0.1:8000/recognize",
                        files={"face": img_encoded.tobytes()}
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

                    cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)
                    cv2.putText(frame, emp_id, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                    cv2.putText(frame, message, (x, y+bh+25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            frame_placeholder.image(frame, channels="BGR")

            table_placeholder.dataframe(
                load_attendance_df(),
                use_container_width=True
            )

        cap.release()

    # Download Excel
    if os.path.exists(ATT_FILE):
        with open(ATT_FILE, "rb") as f:
            st.download_button(
                "⬇️ Download Attendance Excel",
                data=f,
                file_name="attendance.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
