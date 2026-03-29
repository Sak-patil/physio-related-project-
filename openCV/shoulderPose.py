import cv2
import mediapipe as mp

# 1. Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5
)

# 2. Start webcam
cap = cv2.VideoCapture(0)

# Check camera
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Get frame dimensions
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

h, w, _ = frame.shape

# Function to convert normalized coords → mirrored pixel coords
def get_mirrored_coords(landmark, width, height):
    x_pixel = int(landmark.x * width)
    y_pixel = int(landmark.y * height)
    return (width - x_pixel, y_pixel)  # mirror X

# 3. Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pose detection
    results = pose.process(rgb)

    # Flip frame for mirror view
    frame = cv2.flip(frame, 1)

    # If body detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get right arm landmarks
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

        # Convert to mirrored pixel coordinates
        shoulder = get_mirrored_coords(r_shoulder, w, h)
        elbow = get_mirrored_coords(r_elbow, w, h)
        wrist = get_mirrored_coords(r_wrist, w, h)

        # Draw arm lines
        cv2.line(frame, shoulder, elbow, (0, 255, 0), 3)
        cv2.line(frame, elbow, wrist, (0, 255, 0), 3)

        # Draw joints
        cv2.circle(frame, shoulder, 6, (0, 0, 255), -1)
        cv2.circle(frame, elbow, 6, (0, 0, 255), -1)
        cv2.circle(frame, wrist, 6, (0, 0, 255), -1)

    # Show output
    cv2.imshow("Mirrored Pose Tracking - Right Arm", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Flip frame for mirror view
        #frame = cv2.flip(frame, 1)