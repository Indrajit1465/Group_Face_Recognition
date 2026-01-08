import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import requests
from datetime import datetime
import sys

MIN_CHECKOUT_SECONDS = 60 * 30  # 30 minutes

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ATT_DIR = os.path.join(BASE_DIR, "attendance")
ATT_FILE = os.path.join(ATT_DIR, "attendance.xlsx")
os.makedirs(ATT_DIR, exist_ok=True)

# ---------------- Excel Init ----------------
columns = ["Employee ID", "Date", "Check-In", "Check-Out", "Duration"]

if os.path.exists(ATT_FILE) and os.path.getsize(ATT_FILE) > 0:
    df = pd.read_excel(ATT_FILE, engine="openpyxl")

    # Safety: ensure schema
    for col in columns:
        if col not in df.columns:
            df[col] = ""

    df = df[columns]
else:
    df = pd.DataFrame(columns=columns)

# ---------------- MediaPipe ----------------
mp_face = mp.solutions.face_detection.FaceDetection(0, 0.6)
cap = cv2.VideoCapture(0)

print("Press ESC to stop attendance")

# ---------------- Main Loop ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    assigned_ids = set()

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

            records = df[(df["Employee ID"] == emp_id) & (df["Date"] == today)]

            if records.empty:
                # ---------- CHECK-IN ----------
                df.loc[len(df)] = [emp_id, today, now, "", ""]
                df.to_excel(ATT_FILE, index=False)  # ✅ FIX
                message = "Check-In Marked"

            else:
                idx = records.index[0]

                # ❌ Prevent repeated check-out
                if df.at[idx, "Check-Out"] != "":
                    message = "Already Checked-Out"
                else:
                    check_in_time = datetime.strptime(df.at[idx, "Check-In"], "%H:%M:%S")
                    current_time = datetime.strptime(now, "%H:%M:%S")

                    elapsed_seconds = (current_time - check_in_time).total_seconds()

                    if elapsed_seconds >= MIN_CHECKOUT_SECONDS:
                        df.at[idx, "Check-Out"] = now
                        df.at[idx, "Duration"] = str(current_time - check_in_time)
                        df.to_excel(ATT_FILE, index=False)
                        message = "Check-Out Marked"
                    else:
                        message = "Checked-In"

            # ---------------- UI ----------------
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
            cv2.putText(frame, emp_id, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.putText(frame, message, (x, y+bh+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            print(f"[{message}] {emp_id}")

    cv2.imshow("Group Attendance", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
