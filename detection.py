import os
os.environ["QT_QPA_PLATFORM"] = "cocoa"
 
import cv2
import mediapipe as mp
import numpy as np
import time
 
# mediaPipe setup
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
 
# eye landmark indices
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
 
# thresholds
EAR_THRESH = 0.22 
BLINK_THRESH = 0.18  
STRAIN_TIME_LIMIT = 10  
BLINK_WINDOW = 60  
LOW_BLINK_RATE = 8 
 
 
def calc_ear(lm, indices, w, h):
    pts = [np.array([lm[i].x * w, lm[i].y * h]) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)
 
 
def draw_eye(frame, lm, indices, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
    for p in pts:
        cv2.circle(frame, p, 2, (0, 255, 200), -1)
    cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 200), 1)
 
 
def text(frame, msg, pos, color=(255, 255, 255), scale=0.6):
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3)
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
 
 
# state
low_ear_since = None
blink_times = []
in_blink = False
 
cap = cv2.VideoCapture(1)
print("running — press Q to quit")
 
with face_mesh:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
 
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
 
        status, color = "no face detected", (180, 180, 180)
 
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
 
            # draw eyes
            draw_eye(frame, lm, LEFT_EYE,  w, h)
            draw_eye(frame, lm, RIGHT_EYE, w, h)
 
            # EAR
            ear = (calc_ear(lm, LEFT_EYE, w, h) + calc_ear(lm, RIGHT_EYE, w, h)) / 2
 
            # blink detection
            if ear < BLINK_THRESH:
                in_blink = True
            elif in_blink:
                blink_times.append(now)
                in_blink = False
 
            # rolling blink window
            blink_times  = [t for t in blink_times if now - t <= BLINK_WINDOW]
            blink_rate   = len(blink_times) * (60 / BLINK_WINDOW)
 
            # sustained low EAR
            if ear < EAR_THRESH:
                low_ear_since = low_ear_since or now
                strain_secs   = now - low_ear_since
            else:
                low_ear_since = None
                strain_secs   = 0
 
            # status
            if strain_secs >= STRAIN_TIME_LIMIT or (blink_rate < LOW_BLINK_RATE and len(blink_times) > 2):
                status, color = "eye strain — take a break", (0, 0, 255)
            elif ear < EAR_THRESH:
                status, color = "eyes narrowing...", (0, 165, 255)
            else:
                status, color = "eyes are good!", (0, 220, 100)
 
            # HUD
            cv2.rectangle(frame, (0, 0), (w, 100), (20, 20, 20), -1)
            text(frame, status,                              (15, 30),  color, 0.7)
            text(frame, f"EAR: {ear:.3f}",                  (15, 60))
            text(frame, f"blinks/min: {blink_rate:.1f}",    (15, 85))
 
        else:
            cv2.rectangle(frame, (0, 0), (w, 50), (20, 20, 20), -1)
            text(frame, status, (15, 30), color)
 
        cv2.imshow("Eye Strain Monitor", frame)
 
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
 
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)