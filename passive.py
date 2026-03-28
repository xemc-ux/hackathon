import os
os.environ["QT_QPA_PLATFORM"] = "cocoa"
 
import cv2
import mediapipe as mp
import numpy as np
import time
 
# MediaPipe setup
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
 
# eye-landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
 
# head-pose landmark indices
NOSE_TIP = 4
CHIN = 152
LEFT_EYE_L = 263
RIGHT_EYE_R = 33
LEFT_MOUTH = 287
RIGHT_MOUTH = 57
 
# thresholds
EAR_THRESH = 0.22   # below = narrowing
BLINK_THRESH = 0.18   # below = blink
PERCLOS_THRESH = 0.22   # eye "closed" if EAR below this
PERCLOS_WINDOW = 60     # seconds to measure PERCLOS
PERCLOS_ALERT = 0.15   # alert if closed >15% of window
STRAIN_TIME_LIMIT = 10     # seconds before fatigue alert
NARROWING_TIME_LIMIT = 7      # seconds before narrowing shows
LOW_BLINK_RATE = 8      # blinks/min below this = strain
MIN_BLINK_WINDOW = 10     # wait this long before showing blink rate
HEAD_PITCH_THRESH = 15     # degrees nodding
HEAD_ROLL_THRESH = 15     # degrees tilting
 
 
def calc_ear(lm, indices, w, h):
    pts = [np.array([lm[i].x * w, lm[i].y * h]) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)
 
 
def calc_head_pose(lm, w, h):
    model_pts = np.array([
        [0.0,    0.0,    0.0],
        [0.0,  -63.6,  -12.5],
        [-43.3,  32.7, -26.0],
        [43.3,   32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9,  -28.9, -24.1],
    ], dtype=np.float64)
 
    image_pts = np.array([
        [lm[NOSE_TIP].x    * w, lm[NOSE_TIP].y    * h],
        [lm[CHIN].x        * w, lm[CHIN].y        * h],
        [lm[LEFT_EYE_L].x  * w, lm[LEFT_EYE_L].y  * h],
        [lm[RIGHT_EYE_R].x * w, lm[RIGHT_EYE_R].y * h],
        [lm[LEFT_MOUTH].x  * w, lm[LEFT_MOUTH].y  * h],
        [lm[RIGHT_MOUTH].x * w, lm[RIGHT_MOUTH].y * h],
    ], dtype=np.float64)
 
    focal   = w
    cam_mat = np.array([[focal, 0, w/2],
                        [0, focal, h/2],
                        [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1))
 
    ok, rvec, _ = cv2.solvePnP(model_pts, image_pts, cam_mat, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0, 0, 0
 
    rmat, _    = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]  # pitch, yaw, roll
 
 
def draw_eye(frame, lm, indices, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
    for p in pts:
        cv2.circle(frame, p, 2, (0, 255, 200), -1)
    cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 200), 1)
 
 
def put_text(frame, msg, pos, color=(255, 255, 255), scale=0.55, bold=False):
    t = 2 if bold else 1
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t + 2)
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, t)
 
 
# state
low_ear_since = None
blink_times   = []
in_blink      = False
perclos_log   = []
start_time    = time.time()
 
cap = cv2.VideoCapture(0)
print("Running — press Q to quit")
 
with face_mesh:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
 
        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        now     = time.time()
        elapsed = now - start_time
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result  = face_mesh.process(rgb)
 
        status, s_color = "No face detected", (180, 180, 180)
        ear_val = pitch = yaw = roll = 0
        blink_rate = perclos = strain_secs = 0
 
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
 
            draw_eye(frame, lm, LEFT_EYE,  w, h)
            draw_eye(frame, lm, RIGHT_EYE, w, h)
 
            # EAR
            ear_val = (calc_ear(lm, LEFT_EYE,  w, h) +
                       calc_ear(lm, RIGHT_EYE, w, h)) / 2
 
            # blink detection
            if ear_val < BLINK_THRESH:
                in_blink = True
            elif in_blink:
                blink_times.append(now)
                in_blink = False
 
            # blink rate — through elapsed time
            window      = min(elapsed, PERCLOS_WINDOW)
            blink_times = [t for t in blink_times if now - t <= window]
            blink_rate  = (len(blink_times) * 60 / window) if elapsed >= MIN_BLINK_WINDOW else None
 
            # PERCLOS — fraction of frames where eye was closed
            perclos_log.append((now, ear_val < PERCLOS_THRESH))
            perclos_log = [(t, c) for t, c in perclos_log if now - t <= PERCLOS_WINDOW]
            perclos     = sum(c for _, c in perclos_log) / len(perclos_log) if perclos_log else 0
 
            # sustained low EAR timer
            if ear_val < EAR_THRESH:
                low_ear_since = low_ear_since or now
                strain_secs   = now - low_ear_since
            else:
                low_ear_since = None
                strain_secs   = 0
 
            # head pose
            pitch, yaw, roll = calc_head_pose(lm, w, h)
            head_fatigue     = abs(pitch) > HEAD_PITCH_THRESH or abs(roll) > HEAD_ROLL_THRESH
 
            # count fatigue signals
            signals = sum([
                strain_secs >= STRAIN_TIME_LIMIT,
                blink_rate is not None and blink_rate < LOW_BLINK_RATE,
                perclos > PERCLOS_ALERT,
                head_fatigue,
            ])
 
            if signals >= 2:
                status, s_color = "fatigue detected - take a break!", (0, 0, 255)
            elif strain_secs >= NARROWING_TIME_LIMIT:
                status, s_color = "eyes narrowing...", (0, 165, 255)
            else:
                status, s_color = "eyes ok", (0, 220, 100)
 
        # HUD
        cv2.rectangle(frame, (0, 0), (w, 165), (20, 20, 20), -1)
        put_text(frame, status, (15, 28), s_color, scale=0.7, bold=True)
        put_text(frame, f"EAR: {ear_val:.3f}", (15, 58))
 
        if blink_rate is not None:
            put_text(frame, f"blink rate: {blink_rate:.1f} /min  (healthy >={LOW_BLINK_RATE})", (15, 83))
        else:
            put_text(frame, f"blink rate: calibrating ({elapsed:.0f}s / {MIN_BLINK_WINDOW}s)...", (15, 83), (150, 150, 150))
 
        put_text(frame, f"PERCLOS: {perclos*100:.1f}%  (alert >{PERCLOS_ALERT*100:.0f}%)", (15, 108))
        put_text(frame, f"head pitch: {pitch:.1f}°  roll: {roll:.1f}°", (15, 133))
        put_text(frame, f"low-EAR duration: {strain_secs:.1f}s", (15, 158))
 
        cv2.imshow("Eye Strain Monitor", frame)
 
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
 
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)