import cv2
import time
import random

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Simulated match
    matched = random.choice([True, False])

    if matched:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)

    # Draw 90° arm
    shoulder = (320, 150)
    elbow = (320, 250)
    wrist = (420, 250)

    cv2.line(frame, shoulder, elbow, color, 4)
    cv2.line(frame, elbow, wrist, color, 4)

    cv2.putText(frame,
                f"FPS: {int(fps)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2)

    cv2.imshow("Pose Guide", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
