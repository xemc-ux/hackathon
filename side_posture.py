import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def calculate_angle(p1, p2):
    radians = math.atan2(p2[0] - p1[0], p2[1] - p1[1])
    angle = math.degrees(radians)
    return abs(angle)

cap = cv2.VideoCapture(1)

print("Starting Posture Tracker... Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR]
        shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

        ear_pos = (int(ear.x * w), int(ear.y * h))
        sho_pos = (int(shoulder.x * w), int(shoulder.y * h))
        hip_pos = (int(hip.x * w), int(hip.y * h))

        neck_angle = calculate_angle(ear_pos, sho_pos)
        torso_angle = calculate_angle(sho_pos, hip_pos)

        color = (0, 255, 0)
        status = "Good Posture"

        if neck_angle > 20 or torso_angle > 15:
            color = (0, 0, 255)
            status = "BAD POSTURE!"

        cv2.line(frame, ear_pos, sho_pos, color, 3)
        cv2.line(frame, sho_pos, hip_pos, color, 3)

        for pos in [ear_pos, sho_pos, hip_pos]:
            cv2.circle(frame, pos, 7, (255, 255, 255), -1)
            cv2.circle(frame, pos, 8, color, 2)

        cv2.putText(frame, f"Neck Angle: {int(neck_angle)} deg", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Torso Angle: {int(torso_angle)} deg", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, status, (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow('MediaPipe Posture Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
