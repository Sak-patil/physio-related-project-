import cv2
import mediapipe as mp
import numpy as np
import math

# ── CONFIG ─────────────────────────────────────────────────────────────────────
TARGET_ANGLE = 75
TOLERANCE    = 10

# ── MATH ───────────────────────────────────────────────────────────────────────
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    denom   = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))

def normalize_vector(v):
    x, y, z = v
    mag = math.sqrt(x*x + y*y + z*z)
    if mag == 0:
        return (0.0, 0.0, 0.0)
    return (x/mag, y/mag, z/mag)

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def get_vectors(shoulder_norm, elbow_norm, wrist_norm):
    """Build and normalize arm vectors from 3D normalized coords."""
    upper_arm = normalize_vector((
        elbow_norm[0] - shoulder_norm[0],
        elbow_norm[1] - shoulder_norm[1],
        elbow_norm[2] - shoulder_norm[2]
    ))
    forearm = normalize_vector((
        wrist_norm[0] - elbow_norm[0],
        wrist_norm[1] - elbow_norm[1],
        wrist_norm[2] - elbow_norm[2]
    ))
    return upper_arm, forearm

# ── MEDIAPIPE ──────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose
pose    = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
h, w, _ = frame.shape

# ── DIAGRAM ────────────────────────────────────────────────────────────────────
arm_len   = 80
angle_rad = math.radians(TARGET_ANGLE)
EX, EY    = w - 150, 180

# Upper arm horizontal going left
SX, SY = EX - arm_len, EY

# Forearm at TARGET_ANGLE above horizontal
WX = int(EX - arm_len * math.cos(angle_rad))
WY = int(EY - arm_len * math.sin(angle_rad))

DIAG_S = (SX, SY)
DIAG_E = (EX, EY)
DIAG_W = (WX, WY)

# ── REFERENCE VECTORS (doctor defined, hardcoded) ──────────────────────────────
ref_upper_arm = (-0.4855, 0.0239, -0.8739)
ref_forearm   = (0.0850, -0.3752, -0.9230)
ref_captured  = True   # ← set True directly, no need to capture anymore

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    frame   = cv2.flip(frame, 1)

    # Diagram
    cv2.line(frame,   DIAG_S, DIAG_E, (255, 200, 0), 3)
    cv2.line(frame,   DIAG_E, DIAG_W, (255, 200, 0), 3)
    cv2.circle(frame, DIAG_S, 5, (255, 200, 0), -1)
    cv2.circle(frame, DIAG_E, 5, (255, 200, 0), -1)
    cv2.circle(frame, DIAG_W, 5, (255, 200, 0), -1)
    cv2.putText(frame, f"{TARGET_ANGLE}deg",
                (EX - 90, EY + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        r_elbow    = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
        r_wrist    = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

        # NORMALIZED — for vector math
        shoulder_norm = (r_shoulder.x, r_shoulder.y, r_shoulder.z)
        elbow_norm    = (r_elbow.x,    r_elbow.y,    r_elbow.z)
        wrist_norm    = (r_wrist.x,    r_wrist.y,    r_wrist.z)

        # PIXELS — for drawing
        shoulder_px = (int((1 - r_shoulder.x) * w), int(r_shoulder.y * h))
        elbow_px    = (int((1 - r_elbow.x) * w),    int(r_elbow.y * h))
        wrist_px    = (int((1 - r_wrist.x) * w),    int(r_wrist.y * h))

        # ANGLE from pixels
        angle = calculate_angle(shoulder_px, elbow_px, wrist_px)

        # Get live vectors
        live_upper, live_fore = get_vectors(shoulder_norm, elbow_norm, wrist_norm)


        # SCORE — cosine similarity vs reference
        if ref_captured:
            sim_upper = cosine_similarity(live_upper, ref_upper_arm)
            sim_fore  = cosine_similarity(live_fore,  ref_forearm)
            score     = (sim_upper + sim_fore) / 2.0
        else:
            score = 0.0

        # STATUS — angle is the truth, score is extra info
        if (abs(angle - TARGET_ANGLE) <= TOLERANCE and score > 0.8):
            status, color = "MATCHED!", (0, 255, 0)
        elif abs(angle - TARGET_ANGLE) <= TOLERANCE * 2:
            status, color = "CLOSE", (0, 165, 255)
        else:
            status, color = "TRY AGAIN", (0, 0, 255)

        # Draw
        cv2.line(frame,   shoulder_px, elbow_px, color, 3)
        cv2.line(frame,   elbow_px,    wrist_px, color, 3)
        cv2.circle(frame, shoulder_px, 7, (255,255,255), -1)
        cv2.circle(frame, elbow_px,    7, (255,255,255), -1)
        cv2.circle(frame, wrist_px,    7, (255,255,255), -1)

        cv2.putText(frame, status,
                    (30,50),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.putText(frame, f"Angle:  {int(angle)} deg",
                    (30,90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Score:  {score:.2f}",
                    (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f"Target: {TARGET_ANGLE} deg",
                    (30,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,200,0), 1)


    cv2.imshow("Pose Gate", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()