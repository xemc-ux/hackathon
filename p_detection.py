import os
os.environ["QT_QPA_PLATFORM"] = "cocoa"
 
import cv2
import mediapipe as mp
import numpy as np
import time
 
# mediapipe setup
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose_model = mp.solutions.pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
 
# eye-landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
 
# head-pose landmark indices
NOSE_TIP = 4;   CHIN = 152
LEFT_EYE_L = 263; RIGHT_EYE_R = 33
LEFT_MOUTH = 287; RIGHT_MOUTH = 57
 
# thresholds
BLINK_THRESH = 0.18
EAR_THRESH = 0.22
MIN_BLINK_WINDOW = 10
LOW_BLINK_RATE = 15
HEAD_PITCH_THRESH = 25
HEAD_ROLL_THRESH = 25
SHRUG_RATIO = 0.90
 
# durations
NARROWING_DURATION = 10
LOW_EAR_DURATION   = 10
HEAD_DURATION      = 10
LOW_BLINK_DURATION = 10
SHRUG_DURATION     = 10
 
# notification cooldown per signal
NOTIF_COOLDOWN = 60
 
BAR_W = 220
 
 
# helper functions
def calc_ear(lm, indices, w, h):
    pts = [np.array([lm[i].x * w, lm[i].y * h]) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)
 
 
def calc_head_pose(lm, w, h):
    model_pts = np.array([
        [0.0, 0.0, 0.0],  [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0], [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1], [28.9, -28.9, -24.1],
    ], dtype=np.float64)
    image_pts = np.array([
        [lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h],
        [lm[CHIN].x * w, lm[CHIN].y * h],
        [lm[LEFT_EYE_L].x * w, lm[LEFT_EYE_L].y * h],
        [lm[RIGHT_EYE_R].x * w, lm[RIGHT_EYE_R].y * h],
        [lm[LEFT_MOUTH].x * w, lm[LEFT_MOUTH].y * h],
        [lm[RIGHT_MOUTH].x * w, lm[RIGHT_MOUTH].y * h],
    ], dtype=np.float64)
    cam = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(model_pts, image_pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0, 0, 0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]
 
 
def draw_eye(frame, lm, indices, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
    for p in pts:
        cv2.circle(frame, p, 2, (0, 255, 200), -1)
    cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 200), 1)
 
 
def txt(frame, msg, pos, color=(255, 255, 255), scale=0.52, bold=False):
    t = 2 if bold else 1
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t + 2)
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, t)
 
 
def draw_bar(frame, label, secs, threshold, x, y):
    ratio = min(secs / threshold, 1.0)
    filled = int(ratio * BAR_W)
    full = ratio >= 1.0
    color = (0, 0, 255) if full else (0, 165, 255) if ratio >= 0.6 else (0, 200, 100)
    txt(frame, f"{label}   {secs:.0f}s / {threshold}s", (x, y), scale=0.44)
    cv2.rectangle(frame, (x, y + 5), (x + BAR_W, y + 16), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y + 5), (x + filled, y + 16), color, -1)
    cv2.rectangle(frame, (x, y + 5), (x + BAR_W, y + 16), (100, 100, 100), 1)
    if full:
        txt(frame, "!", (x + BAR_W + 6, y + 15), (0, 0, 255), scale=0.5, bold=True)
    return full
 
 
def divider(frame, title, y):
    cv2.line(frame, (8, y), (632, y), (80, 80, 80), 1)
    txt(frame, title, (8, y + 14), (180, 180, 180), scale=0.5, bold=True)
 
 
def check_notif(signal_full, last_notif, now):
    """returns updated last_notif and whether to fire notification."""
    if signal_full and now - last_notif >= NOTIF_COOLDOWN:
        return now, True
    return last_notif, False
 
 
# state
low_ear_since   = None
head_bad_since  = None
low_blink_since = None
shrug_since     = None
blink_times     = []
in_blink        = False
baseline_l_gap  = None
baseline_r_gap  = None
start_time      = time.time()
 
# per-signal notification timestamps
last_notif_ear   = 0
last_notif_head  = 0
last_notif_blink = 0
last_notif_shrug = 0
 
cap = cv2.VideoCapture(1)
print("sit up straight to calibrate. press r to recalibrate, q to quit.")
 
with face_mesh, pose_model:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
 
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        elapsed = now - start_time
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
        face_res = face_mesh.process(rgb)
        pose_res = pose_model.process(rgb)
 
        # reset per-frame values
        ear_val = pitch = roll = 0
        blink_rate = None
        strain_secs = head_secs = blink_low_secs = shrug_secs = 0
        fatigue_status, fatigue_color = "no fatigue apparent", (0, 220, 100)
        posture_status, posture_color = "calibrating...", (150, 150, 150)
 
        # fatigue (face mesh)
        if face_res.multi_face_landmarks:
            lm = face_res.multi_face_landmarks[0].landmark
 
            draw_eye(frame, lm, LEFT_EYE,  w, h)
            draw_eye(frame, lm, RIGHT_EYE, w, h)
 
            ear_val = (calc_ear(lm, LEFT_EYE,  w, h) +
                       calc_ear(lm, RIGHT_EYE, w, h)) / 2
 
            # blink-detection
            if ear_val < BLINK_THRESH:
                in_blink = True
            elif in_blink:
                blink_times.append(now)
                in_blink = False
 
            window = min(elapsed, 60)
            blink_times = [t for t in blink_times if now - t <= window]
            if elapsed >= MIN_BLINK_WINDOW:
                blink_rate = len(blink_times) * 60 / window
 
            # head-pose
            pitch, _, roll = calc_head_pose(lm, w, h)
            head_bad = abs(pitch) > HEAD_PITCH_THRESH or abs(roll) > HEAD_ROLL_THRESH
 
            # timers for fatigue bars
            if ear_val < EAR_THRESH:
                low_ear_since = low_ear_since or now
                strain_secs = now - low_ear_since
            else:
                low_ear_since = None
 
            if head_bad:
                head_bad_since = head_bad_since or now
                head_secs = now - head_bad_since
            else:
                head_bad_since = None
 
            if blink_rate is not None and blink_rate < LOW_BLINK_RATE:
                low_blink_since = low_blink_since or now
                blink_low_secs = now - low_blink_since
            else:
                low_blink_since = None
 
        # posture
        if pose_res.pose_landmarks:
            plm = pose_res.pose_landmarks.landmark
 
            curr_l_gap = plm[11].y - plm[7].y
            curr_r_gap = plm[12].y - plm[8].y
 
            if baseline_l_gap is None:
                baseline_l_gap = curr_l_gap
                baseline_r_gap = curr_r_gap
                print("posture calibrated.")
 
            is_shrug = (curr_l_gap < baseline_l_gap * SHRUG_RATIO or
                        curr_r_gap < baseline_r_gap * SHRUG_RATIO)
 
            if is_shrug:
                shrug_since = shrug_since or now
                shrug_secs  = now - shrug_since
            else:
                shrug_since = None
                shrug_secs  = 0
 
            mp_drawing.draw_landmarks(frame, pose_res.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)
 
        # HUD
        cv2.rectangle(frame, (0, 0), (w, 290), (20, 20, 20), -1)
 
        # fatigue section
        divider(frame, "FATIGUE", 6)
        b1 = draw_bar(frame, "low EAR  ", strain_secs, LOW_EAR_DURATION, 15, 28)
        b2 = draw_bar(frame, "head tilt", head_secs, HEAD_DURATION, 15, 58)
        b3 = draw_bar(frame, "low blink", blink_low_secs, LOW_BLINK_DURATION, 15, 88)
 
        any_fatigue = any([b1, b2, b3])
        if any_fatigue:
            fatigue_status, fatigue_color = "fatigue detected. take a break.", (0, 0, 255)
        elif strain_secs >= NARROWING_DURATION:
            fatigue_status, fatigue_color = "eyes narrowing...monitoring", (0, 165, 255)
 
        txt(frame, fatigue_status, (15, 118), fatigue_color, scale=0.6, bold=True)
 
        br_str = f"{blink_rate:.1f}/min" if blink_rate is not None else f"calibrating ({elapsed:.0f}s)..."
        txt(frame, f"EAR: {ear_val:.3f}   Blink: {br_str}   Pitch: {pitch:.1f}   Roll: {roll:.1f}",
            (15, 140), scale=0.44)
 
        # posture section
        divider(frame, "POSTURE", 158)
        p1 = draw_bar(frame, "shrugging", shrug_secs, SHRUG_DURATION, 15, 180)
 
        any_posture = p1
        if shrug_secs >= SHRUG_DURATION:
            posture_status, posture_color = "shoulders too high. relax them.", (0, 0, 255)
        elif baseline_l_gap is not None:
            posture_status, posture_color = "good posture", (0, 220, 100)
 
        txt(frame, posture_status, (15, 215), posture_color, scale=0.6, bold=True)
        txt(frame, "r = recalibrate   q = quit", (15, 240), (100, 100, 100), scale=0.42)
 
        # per-signal notification logic
        last_notif_ear, fire_ear   = check_notif(b1, last_notif_ear,   now)
        last_notif_head, fire_head  = check_notif(b2, last_notif_head,  now)
        last_notif_blink, fire_blink = check_notif(b3, last_notif_blink, now)
        last_notif_shrug, fire_shrug = check_notif(p1, last_notif_shrug, now)
 
        if fire_ear:
            print("notif: low EAR. your eyes are narrowing. you may be fatigued.")
        if fire_head:
            print("notif: head tilt detected. adjust your head position.")
        if fire_blink:
            print("notif: low blink rate. remember to blink more often.")
        if fire_shrug:
            print("notif: shoulders too high. relax your shoulders.")
 
        cv2.imshow("Fatigue & Posture Monitor", frame)
 
        # key handling
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            baseline_l_gap = None
            baseline_r_gap = None
            print("recalibrating posture...")
 
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
 