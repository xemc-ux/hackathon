import os
os.environ["QT_QPA_PLATFORM"] = "cocoa"

import cv2
import mediapipe as mp
import numpy as np
import time

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_pose      = mp.solutions.pose
mp_drawing   = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,       # needed for iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Eye landmark indices ──────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Iris centre landmarks (refine_landmarks=True required)
LEFT_IRIS  = 468
RIGHT_IRIS = 473

# Head pose landmarks
NOSE_TIP    = 4
CHIN        = 152
LEFT_EYE_L  = 263
RIGHT_EYE_R = 33
LEFT_MOUTH  = 287
RIGHT_MOUTH = 57

# ── Fatigue thresholds ────────────────────────────────────────────────────────
BLINK_THRESH       = 0.18
EAR_THRESH         = 0.22
MIN_BLINK_WINDOW   = 10
LOW_BLINK_RATE     = 8
HEAD_PITCH_THRESH  = 15
HEAD_ROLL_THRESH   = 15

# Sustained durations before bar fills (fatigue)
NARROWING_DURATION = 7
LOW_EAR_DURATION   = 20
HEAD_DURATION      = 15
LOW_BLINK_DURATION = 30

# ── Posture thresholds ────────────────────────────────────────────────────────
SHRUG_RATIO    = 0.75   # gap 25% smaller than baseline = shrug
UNEVEN_THRESH  = 0.03   # shoulder height diff

# Sustained durations before posture bar fills
SHRUG_DURATION  = 8
UNEVEN_DURATION = 8

# ── Attention / cognitive load (iris) ─────────────────────────────────────────
# Iris position ratio: how far iris is from eye centre (0=centre, 1=edge)
# High variance over time = distracted gaze
ATTENTION_WINDOW   = 10   # seconds of iris variance to average
DISTRACTED_THRESH  = 0.04 # iris displacement variance above this = distracted
DISTRACTED_DURATION = 12  # seconds before bar fills


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    cam_mat = np.array([[focal, 0, w/2], [0, focal, h/2], [0, 0, 1]], dtype=np.float64)
    dist    = np.zeros((4, 1))
    ok, rvec, _ = cv2.solvePnP(model_pts, image_pts, cam_mat, dist,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0, 0, 0
    rmat, _               = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]


def iris_displacement(lm, eye_indices, iris_idx, w, h):
    """How far the iris centre is from the eye centre, normalised by eye width."""
    eye_pts  = [np.array([lm[i].x * w, lm[i].y * h]) for i in eye_indices]
    eye_cx   = np.mean([p[0] for p in eye_pts])
    eye_cy   = np.mean([p[1] for p in eye_pts])
    eye_w    = np.linalg.norm(eye_pts[0] - eye_pts[3])
    iris_x   = lm[iris_idx].x * w
    iris_y   = lm[iris_idx].y * h
    dist     = np.sqrt((iris_x - eye_cx)**2 + (iris_y - eye_cy)**2)
    return dist / (eye_w + 1e-6)


def draw_eye(frame, lm, indices, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
    for p in pts:
        cv2.circle(frame, p, 2, (0, 255, 200), -1)
    cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 200), 1)


def put_text(frame, msg, pos, color=(255, 255, 255), scale=0.52, bold=False):
    t = 2 if bold else 1
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t + 2)
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, t)


def signal_bar(frame, label, seconds, threshold, pos):
    bar_w  = 160
    ratio  = min(seconds / threshold, 1.0)
    filled = int(ratio * bar_w)
    full   = ratio >= 1.0
    color  = (0, 0, 255) if full else (0, 165, 255) if ratio >= 0.6 else (0, 200, 100)
    x, y   = pos
    put_text(frame, label, (x, y), scale=0.42)
    cv2.rectangle(frame, (x, y + 4), (x + bar_w, y + 13), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y + 4), (x + filled, y + 13), color, -1)
    if full:
        put_text(frame, "!", (x + bar_w + 4, y + 13), (0, 0, 255), scale=0.45, bold=True)
    return full


def section_header(frame, text, y, color=(200, 200, 200)):
    cv2.line(frame, (10, y - 4), (630, y - 4), (60, 60, 60), 1)
    put_text(frame, text, (10, y + 10), color, scale=0.48, bold=True)


# ── State ─────────────────────────────────────────────────────────────────────
# Fatigue
low_ear_since   = None
head_bad_since  = None
low_blink_since = None
blink_times     = []
in_blink        = False
start_time      = time.time()

# Posture
baseline_l_gap  = None
baseline_r_gap  = None
shrug_since     = None
uneven_since    = None

# Attention
iris_log            = []   # list of (timestamp, avg_displacement)
distracted_since    = None

cap = cv2.VideoCapture(1)
print("Starting — sit up straight for calibration. Press R to recalibrate, Q to quit.")

with face_mesh, pose_model:
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        now     = time.time()
        elapsed = now - start_time
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_result = face_mesh.process(rgb)
        pose_result = pose_model.process(rgb)

        # Per-frame values
        ear_val = pitch = roll = 0
        blink_rate = 0
        strain_secs = head_secs = blink_low_secs = 0
        shrug_secs = uneven_secs = distracted_secs = 0
        posture_status, posture_color = "Calibrating...", (150, 150, 150)
        fatigue_status, fatigue_color = "No fatigue apparent", (0, 220, 100)
        attention_status, attention_color = "Focused", (0, 220, 100)
        avg_disp = 0

        # ── FACE / FATIGUE ────────────────────────────────────────────────────
        if face_result.multi_face_landmarks:
            lm = face_result.multi_face_landmarks[0].landmark

            draw_eye(frame, lm, LEFT_EYE,  w, h)
            draw_eye(frame, lm, RIGHT_EYE, w, h)

            # Draw iris points
            for idx in [LEFT_IRIS, RIGHT_IRIS]:
                ix = int(lm[idx].x * w)
                iy = int(lm[idx].y * h)
                cv2.circle(frame, (ix, iy), 3, (255, 200, 0), -1)

            # EAR
            ear_val = (calc_ear(lm, LEFT_EYE,  w, h) +
                       calc_ear(lm, RIGHT_EYE, w, h)) / 2

            # Blink
            if ear_val < BLINK_THRESH:
                in_blink = True
            elif in_blink:
                blink_times.append(now)
                in_blink = False
            window      = min(elapsed, 60)
            blink_times = [t for t in blink_times if now - t <= window]
            blink_rate  = (len(blink_times) * 60 / window) if elapsed >= MIN_BLINK_WINDOW else None

            # Head pose
            pitch, _, roll = calc_head_pose(lm, w, h)
            head_bad       = abs(pitch) > HEAD_PITCH_THRESH or abs(roll) > HEAD_ROLL_THRESH

            # Sustained timers — fatigue
            if ear_val < EAR_THRESH:
                low_ear_since = low_ear_since or now
                strain_secs   = now - low_ear_since
            else:
                low_ear_since = None
                strain_secs   = 0

            if head_bad:
                head_bad_since = head_bad_since or now
                head_secs      = now - head_bad_since
            else:
                head_bad_since = None
                head_secs      = 0

            if blink_rate is not None and blink_rate < LOW_BLINK_RATE:
                low_blink_since = low_blink_since or now
                blink_low_secs  = now - low_blink_since
            else:
                low_blink_since = None
                blink_low_secs  = 0

            # ── ATTENTION — iris displacement variance ─────────────────────
            l_disp   = iris_displacement(lm, LEFT_EYE,  LEFT_IRIS,  w, h)
            r_disp   = iris_displacement(lm, RIGHT_EYE, RIGHT_IRIS, w, h)
            avg_disp = (l_disp + r_disp) / 2
            iris_log.append((now, avg_disp))
            iris_log = [(t, d) for t, d in iris_log if now - t <= ATTENTION_WINDOW]

            if len(iris_log) >= 5:
                variance = float(np.var([d for _, d in iris_log]))
                distracted = variance > DISTRACTED_THRESH
                if distracted:
                    distracted_since = distracted_since or now
                    distracted_secs  = now - distracted_since
                else:
                    distracted_since = None
                    distracted_secs  = 0

                if distracted_secs >= DISTRACTED_DURATION:
                    attention_status, attention_color = "Gaze wandering — refocus", (0, 100, 255)
                else:
                    attention_status, attention_color = "Focused", (0, 220, 100)

        # ── POSTURE ───────────────────────────────────────────────────────────
        if pose_result.pose_landmarks:
            plm = pose_result.pose_landmarks.landmark

            curr_l_gap = plm[11].y - plm[7].y
            curr_r_gap = plm[12].y - plm[8].y

            # Calibrate on first frame
            if baseline_l_gap is None:
                baseline_l_gap = curr_l_gap
                baseline_r_gap = curr_r_gap

            l_shrug    = curr_l_gap < (baseline_l_gap * SHRUG_RATIO)
            r_shrug    = curr_r_gap < (baseline_r_gap * SHRUG_RATIO)
            is_shrug   = l_shrug or r_shrug
            sh_diff    = abs(plm[11].y - plm[12].y)
            is_uneven  = sh_diff > UNEVEN_THRESH

            # Sustained posture timers
            if is_shrug:
                shrug_since = shrug_since or now
                shrug_secs  = now - shrug_since
            else:
                shrug_since = None
                shrug_secs  = 0

            if is_uneven and not is_shrug:
                uneven_since = uneven_since or now
                uneven_secs  = now - uneven_since
            else:
                uneven_since = None
                uneven_secs  = 0

            # Posture status — independent of fatigue
            if shrug_secs >= SHRUG_DURATION:
                posture_status, posture_color = "Shoulders too high — relax them", (0, 0, 255)
            elif uneven_secs >= UNEVEN_DURATION:
                posture_status, posture_color = "Uneven shoulders — straighten up", (0, 165, 255)
            elif baseline_l_gap is not None:
                posture_status, posture_color = "Good posture", (0, 220, 100)

            mp_drawing.draw_landmarks(frame, pose_result.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)

        # ── FATIGUE STATUS — driven by bars ───────────────────────────────────
        # Draw HUD background
        cv2.rectangle(frame, (0, 0), (w, 310), (20, 20, 20), -1)

        # Section: Fatigue
        section_header(frame, "FATIGUE", 18)
        bar1 = signal_bar(frame, f"Low EAR    {strain_secs:.0f}s / {LOW_EAR_DURATION}s",
                          strain_secs,    LOW_EAR_DURATION,   (15, 35))
        bar2 = signal_bar(frame, f"Head tilt  {head_secs:.0f}s / {HEAD_DURATION}s",
                          head_secs,      HEAD_DURATION,      (15, 58))
        bar3 = signal_bar(frame, f"Low blink  {blink_low_secs:.0f}s / {LOW_BLINK_DURATION}s",
                          blink_low_secs, LOW_BLINK_DURATION, (15, 81))

        if any([bar1, bar2, bar3]):
            fatigue_status, fatigue_color = "Fatigue is apparent — take a break", (0, 0, 255)
        elif strain_secs >= NARROWING_DURATION:
            fatigue_status, fatigue_color = "Eyes narrowing — monitoring", (0, 165, 255)

        put_text(frame, fatigue_status, (15, 108), fatigue_color, scale=0.6, bold=True)

        if blink_rate is not None:
            put_text(frame, f"EAR: {ear_val:.3f}   Blink: {blink_rate:.1f}/min   "
                            f"Pitch: {pitch:.1f}°  Roll: {roll:.1f}°", (15, 128))
        else:
            put_text(frame, f"EAR: {ear_val:.3f}   Blink: calibrating ({elapsed:.0f}s)...", (15, 128))

        # Section: Posture
        section_header(frame, "POSTURE", 148)
        signal_bar(frame, f"Shrugging  {shrug_secs:.0f}s / {SHRUG_DURATION}s",
                   shrug_secs,  SHRUG_DURATION,  (15, 165))
        signal_bar(frame, f"Uneven     {uneven_secs:.0f}s / {UNEVEN_DURATION}s",
                   uneven_secs, UNEVEN_DURATION, (15, 188))
        put_text(frame, posture_status, (15, 212), posture_color, scale=0.6, bold=True)

        # Section: Attention
        section_header(frame, "ATTENTION", 232)
        signal_bar(frame, f"Gaze wander {distracted_secs:.0f}s / {DISTRACTED_DURATION}s",
                   distracted_secs, DISTRACTED_DURATION, (15, 249))
        put_text(frame, f"Iris displacement: {avg_disp:.3f}", (15, 272))
        put_text(frame, attention_status, (15, 295), attention_color, scale=0.6, bold=True)

        cv2.imshow("Focus & Fatigue Monitor", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            baseline_l_gap = None
            baseline_r_gap = None
            print("Recalibrating posture...")

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)