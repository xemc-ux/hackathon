import os
os.environ["QT_QPA_PLATFORM"] = "cocoa"

import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess

# MediaPipe setup
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Eye landmark indices
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Head pose landmark indices
NOSE_TIP    = 4
CHIN        = 152
LEFT_EYE_L  = 263
RIGHT_EYE_R = 33
LEFT_MOUTH  = 287
RIGHT_MOUTH = 57

# --- Sustained time thresholds before flagging ---
BLINK_THRESH         = 0.18   # EAR below = blink
EAR_THRESH           = 0.22   # EAR below = narrowing
PERCLOS_THRESH       = 0.22   # EAR below = "closed" for PERCLOS
PERCLOS_WINDOW       = 60     # seconds to compute PERCLOS over
PERCLOS_ALERT        = 0.15   # >15% closed = signal
MIN_BLINK_WINDOW     = 10     # wait before showing blink rate
LOW_BLINK_RATE       = 8      # blinks/min below this = signal

# How long each signal must persist before it counts
NARROWING_DURATION   = 7      # seconds of low EAR before "narrowing"
LOW_EAR_DURATION     = 20     # seconds before low EAR counts as fatigue signal
HEAD_DURATION        = 15     # seconds of bad head pose before it counts
LOW_BLINK_DURATION   = 30     # seconds of low blink rate before it counts
PERCLOS_DURATION     = 20     # seconds of high PERCLOS before it counts

HEAD_PITCH_THRESH    = 15     # degrees
HEAD_ROLL_THRESH     = 15     # degrees

# How many sustained signals needed for FATIGUE DETECTED
SIGNALS_NEEDED       = 2

# Notification cooldown — don't spam notifications
NOTIF_COOLDOWN       = 60     # seconds between notifications


def send_notification(title, message):
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script])


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

    rmat, _              = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]


def draw_eye(frame, lm, indices, w, h):
    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in indices]
    for p in pts:
        cv2.circle(frame, p, 2, (0, 255, 200), -1)
    cv2.polylines(frame, [np.array(pts, np.int32)], True, (0, 255, 200), 1)


def put_text(frame, msg, pos, color=(255, 255, 255), scale=0.55, bold=False):
    t = 2 if bold else 1
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t + 2)
    cv2.putText(frame, msg, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, t)


def signal_bar(frame, label, seconds, threshold, pos, w):
    """Draws a progress bar showing how long a signal has been active."""
    bar_w    = 180
    filled   = int(min(seconds / threshold, 1.0) * bar_w)
    color    = (0, 200, 100) if seconds < threshold * 0.6 else \
               (0, 165, 255) if seconds < threshold else (0, 0, 255)
    x, y     = pos
    put_text(frame, label, (x, y), scale=0.45)
    cv2.rectangle(frame, (x, y + 4),  (x + bar_w, y + 14), (60, 60, 60), -1)
    cv2.rectangle(frame, (x, y + 4),  (x + filled, y + 14), color, -1)


# ── State ─────────────────────────────────────────────────────────────────────
low_ear_since      = None
head_bad_since     = None
low_blink_since    = None
perclos_bad_since  = None
blink_times        = []
in_blink           = False
perclos_log        = []
start_time         = time.time()
last_notif_time    = 0

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
        ear_val = pitch = roll = 0
        blink_rate = perclos = 0
        strain_secs = head_secs = blink_low_secs = perclos_secs = 0

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            draw_eye(frame, lm, LEFT_EYE,  w, h)
            draw_eye(frame, lm, RIGHT_EYE, w, h)

            # EAR
            ear_val = (calc_ear(lm, LEFT_EYE,  w, h) +
                       calc_ear(lm, RIGHT_EYE, w, h)) / 2

            # Blink detection
            if ear_val < BLINK_THRESH:
                in_blink = True
            elif in_blink:
                blink_times.append(now)
                in_blink = False

            window      = min(elapsed, PERCLOS_WINDOW)
            blink_times = [t for t in blink_times if now - t <= window]
            blink_rate  = (len(blink_times) * 60 / window) if elapsed >= MIN_BLINK_WINDOW else None

            # PERCLOS
            perclos_log.append((now, ear_val < PERCLOS_THRESH))
            perclos_log = [(t, c) for t, c in perclos_log if now - t <= PERCLOS_WINDOW]
            perclos     = sum(c for _, c in perclos_log) / len(perclos_log) if perclos_log else 0

            # Head pose
            pitch, _, roll   = calc_head_pose(lm, w, h)
            head_bad         = abs(pitch) > HEAD_PITCH_THRESH or abs(roll) > HEAD_ROLL_THRESH

            # ── Sustained timers ──────────────────────────────────────────────

            # Low EAR timer
            if ear_val < EAR_THRESH:
                low_ear_since = low_ear_since or now
                strain_secs   = now - low_ear_since
            else:
                low_ear_since = None
                strain_secs   = 0

            # Head pose timer
            if head_bad:
                head_bad_since = head_bad_since or now
                head_secs      = now - head_bad_since
            else:
                head_bad_since = None
                head_secs      = 0

            # Low blink rate timer
            if blink_rate is not None and blink_rate < LOW_BLINK_RATE:
                low_blink_since = low_blink_since or now
                blink_low_secs  = now - low_blink_since
            else:
                low_blink_since = None
                blink_low_secs  = 0

            # High PERCLOS timer
            if perclos > PERCLOS_ALERT:
                perclos_bad_since = perclos_bad_since or now
                perclos_secs      = now - perclos_bad_since
            else:
                perclos_bad_since = None
                perclos_secs      = 0

            # ── Count signals that have been sustained long enough ────────────
            signals = sum([
                strain_secs    >= LOW_EAR_DURATION,
                head_secs      >= HEAD_DURATION,
                blink_low_secs >= LOW_BLINK_DURATION,
                perclos_secs   >= PERCLOS_DURATION,
            ])

            # ── Status ────────────────────────────────────────────────────────
            if signals >= SIGNALS_NEEDED:
                status, s_color = "FATIGUE DETECTED — Take a break!", (0, 0, 255)
                if now - last_notif_time > NOTIF_COOLDOWN:
                    send_notification("Eye Strain Monitor", "Fatigue detected — take a break!")
                    last_notif_time = now
            elif strain_secs >= NARROWING_DURATION:
                status, s_color = "Eyes narrowing — monitor closely", (0, 165, 255)
            else:
                status, s_color = "Eyes OK", (0, 220, 100)

        # ── HUD ───────────────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 210), (20, 20, 20), -1)

        put_text(frame, status, (15, 28), s_color, scale=0.7, bold=True)
        put_text(frame, f"EAR: {ear_val:.3f}   PERCLOS: {perclos*100:.1f}%   "
                        f"Pitch: {pitch:.1f}°  Roll: {roll:.1f}°", (15, 55))

        if blink_rate is not None:
            put_text(frame, f"Blink rate: {blink_rate:.1f} /min", (15, 78))
        else:
            put_text(frame, f"Blink rate: calibrating ({elapsed:.0f}s)...", (15, 78), (150, 150, 150))

        # Signal progress bars
        signal_bar(frame, f"Low EAR   {strain_secs:.0f}s / {LOW_EAR_DURATION}s",
                   strain_secs,    LOW_EAR_DURATION,    (15, 100),  w)
        signal_bar(frame, f"Head pose {head_secs:.0f}s / {HEAD_DURATION}s",
                   head_secs,      HEAD_DURATION,       (15, 128),  w)
        signal_bar(frame, f"Low blink {blink_low_secs:.0f}s / {LOW_BLINK_DURATION}s",
                   blink_low_secs, LOW_BLINK_DURATION,  (15, 156),  w)
        signal_bar(frame, f"PERCLOS   {perclos_secs:.0f}s / {PERCLOS_DURATION}s",
                   perclos_secs,   PERCLOS_DURATION,    (15, 184),  w)

        cv2.imshow("Eye Strain Monitor", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)