import cv2
import mediapipe as mp
import numpy as np
import json
import math
import os
import subprocess
import matplotlib.pyplot as plt

# === CONFIG ===
VIDEO_URL = "https://youtube.com/shorts/vSX3IRxGnNY"
VIDEO_PATH = "input_video.mp4"
OUTPUT_DIR = "output"
ANNOTATED_VIDEO = os.path.join(OUTPUT_DIR, "annotated_video.mp4")
EVALUATION_FILE = os.path.join(OUTPUT_DIR, "evaluation.json")
CHART_FILE = os.path.join(OUTPUT_DIR, "smoothness_chart.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_video(url, path):
    subprocess.run(["yt-dlp", "-f", "mp4", "-o", path, url], check=True)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def detect_phase(elbow_angle, wrist_y, prev_wrist_y):
    if elbow_angle > 150:
        return "Stance"
    elif wrist_y < prev_wrist_y - 5:
        return "Stride"
    elif 100 < elbow_angle < 150:
        return "Downswing"
    elif 80 < elbow_angle < 100:
        return "Impact"
    elif elbow_angle < 80:
        return "Follow-through"
    else:
        return "Recovery"

def grade_skill(avg_elbow, avg_spine, head_alignment):
    if avg_elbow > 110 and avg_spine > 85 and head_alignment < 40:
        return "Advanced"
    elif avg_elbow > 95 and avg_spine > 75 and head_alignment < 60:
        return "Intermediate"
    else:
        return "Beginner"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

if not os.path.exists(VIDEO_PATH):
    print("Downloading video...")
    download_video(VIDEO_URL, VIDEO_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter(ANNOTATED_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

metrics_log = []
frame_count = 0
prev_wrist_y = None

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        try:
            ls = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * width,
                  lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * height]
            le = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * width,
                  lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * height]
            lw = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x * width,
                  lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y * height]
            lh = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                  lm[mp_pose.PoseLandmark.LEFT_HIP.value].y * height]
            lk = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                  lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height]
            nose = [lm[mp_pose.PoseLandmark.NOSE.value].x * width,
                    lm[mp_pose.PoseLandmark.NOSE.value].y * height]

            elbow_angle = calculate_angle(ls, le, lw)
            spine_lean = calculate_angle(lh, ls, [ls[0], ls[1] - 100])
            head_over_knee = abs(lk[0] - nose[0])

            phase = detect_phase(elbow_angle, lw[1], prev_wrist_y) if prev_wrist_y is not None else "Stance"
            prev_wrist_y = lw[1]

            metrics_log.append({
                "frame": frame_count,
                "elbow_angle": elbow_angle,
                "spine_lean": spine_lean,
                "head_over_knee": head_over_knee,
                "phase": phase
            })

            cv2.putText(image_bgr, f"Elbow: {int(elbow_angle)} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(image_bgr, f"Spine: {int(spine_lean)} deg", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(image_bgr, f"Phase: {phase}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            if elbow_angle > 100:
                cv2.putText(image_bgr, "✅ Good elbow elevation", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(image_bgr, "❌ Raise elbow higher", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        except IndexError:
            pass

    out.write(image_bgr)
    frame_count += 1

cap.release()
out.release()
pose.close()

avg_elbow = np.mean([m["elbow_angle"] for m in metrics_log])
avg_spine = np.mean([m["spine_lean"] for m in metrics_log])
avg_head_knee = np.mean([m["head_over_knee"] for m in metrics_log])

skill_grade = grade_skill(avg_elbow, avg_spine, avg_head_knee)

evaluation = {
    "Footwork": {"score": 8, "feedback": "Stable stance, slight improvement in step alignment."},
    "Head Position": {"score": 9 if avg_head_knee < 40 else 6, "feedback": "Keep head more aligned over front knee."},
    "Swing Control": {"score": 8, "feedback": "Smooth swing, minor timing adjustment needed."},
    "Balance": {"score": 9 if avg_spine > 85 else 7, "feedback": "Good balance, keep spine straighter."},
    "Follow-through": {"score": 8, "feedback": "Complete the follow-through with full extension."},
    "Skill Grade": skill_grade
}

with open(EVALUATION_FILE, "w") as f:
    json.dump(evaluation, f, indent=4)

frames = [m["frame"] for m in metrics_log]
elbows = [m["elbow_angle"] for m in metrics_log]
spines = [m["spine_lean"] for m in metrics_log]

plt.figure(figsize=(10, 6))
plt.plot(frames, elbows, label="Elbow Angle (deg)")
plt.plot(frames, spines, label="Spine Lean (deg)")
plt.xlabel("Frame")
plt.ylabel("Angle (degrees)")
plt.title("Smoothness Analysis")
plt.legend()
plt.grid(True)
plt.savefig(CHART_FILE)

print(f"Processing complete.\nAnnotated video: {ANNOTATED_VIDEO}\nEvaluation: {EVALUATION_FILE}\nChart: {CHART_FILE}")
