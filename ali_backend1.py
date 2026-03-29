"""
AL-I backend — dual-camera fatigue & posture monitor
1. Camera 0 (iPhone/continuity, side view): side posture (neck/torso angles)
2. Camera 1 (MacBook, front view): fatigue (EAR, blink, head tilt) + shrug
landmarks drawn by MediaPipe's own drawing utils on the frame, streamed as MJPEG.
run: python ali_backend.py
"""
import os, cv2, mediapipe as mp, numpy as np, math, time, json, threading, subprocess
from flask import Flask, Response, jsonify, request

app = Flask(__name__)
app.config["SECRET_KEY"] = "ali-secret"

mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh      = mp.solutions.face_mesh
mp_pose           = mp.solutions.pose

_fm_front = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)
_pose_front = mp_pose.Pose(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)
_pose_side = mp_pose.Pose(
    static_image_mode=False, model_complexity=1,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

NOSE_TIP = 4; CHIN = 152
LEFT_EYE_L = 263; RIGHT_EYE_R = 33
LEFT_MOUTH = 287; RIGHT_MOUTH = 57

BLINK_THRESH       = 0.18
EAR_THRESH         = 0.22
MIN_BLINK_WINDOW   = 10
LOW_BLINK_RATE     = 15
HEAD_PITCH_THRESH  = 25
HEAD_ROLL_THRESH   = 25
SHRUG_RATIO        = 0.90
NECK_ANGLE_THRESH  = 20
TORSO_ANGLE_THRESH = 15

LOW_EAR_DURATION   = 10
HEAD_DURATION      = 10
LOW_BLINK_DURATION = 10
SHRUG_DURATION     = 10
BAD_SIDE_DURATION  = 30
NOTIF_COOLDOWN     = 60

state_lock       = threading.Lock()
frame_lock_front = threading.Lock()
frame_lock_side  = threading.Lock()

shared_state = {
    "ear": 0.0, "blink_rate": 0.0, "pitch": 0.0, "roll": 0.0,
    "low_ear_secs": 0.0, "head_secs": 0.0, "blink_low_secs": 0.0,
    "shrug_secs": 0.0,
    "neck_angle": 0.0, "torso_angle": 0.0, "bad_side_secs": 0.0,
    "fatigue_alert": False, "posture_alert": False, "side_posture_alert": False,
    "alerts": [], "notifications": [],
    "calibrated_front": False, "calibrated_side": False,
}

latest_front_frame = None
latest_side_frame  = None

last_notif  = {"ear": 0, "head": 0, "blink": 0, "shrug": 0, "side": 0}
recal_front = threading.Event()
recal_side  = threading.Event()

def calc_ear(lm, indices, w, h):
    pts = [np.array([lm[i].x * w, lm[i].y * h]) for i in indices]
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C) if C > 0 else 0.0

def calc_head_pose(lm, w, h):
    model_pts = np.array([
        [0.0, 0.0, 0.0],      [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0], [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],[28.9, -28.9, -24.1],
    ], dtype=np.float64)
    image_pts = np.array([
        [lm[NOSE_TIP].x*w,    lm[NOSE_TIP].y*h],
        [lm[CHIN].x*w,        lm[CHIN].y*h],
        [lm[LEFT_EYE_L].x*w,  lm[LEFT_EYE_L].y*h],
        [lm[RIGHT_EYE_R].x*w, lm[RIGHT_EYE_R].y*h],
        [lm[LEFT_MOUTH].x*w,  lm[LEFT_MOUTH].y*h],
        [lm[RIGHT_MOUTH].x*w, lm[RIGHT_MOUTH].y*h],
    ], dtype=np.float64)
    cam_matrix = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(model_pts, image_pts, cam_matrix,
                                np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0], angles[1], angles[2]

def calc_angle(p1, p2):
    return abs(math.degrees(math.atan2(p2[0]-p1[0], p2[1]-p1[1])))

def check_notif(key, condition, now):
    if condition and now - last_notif[key] >= NOTIF_COOLDOWN:
        last_notif[key] = now; return True
    return False

def mac_notify(title, msg):
    try:
        subprocess.run(["osascript", "-e",
            f'display notification "{msg}" with title "{title}" sound name "Ping"'],
            check=False, timeout=5)
    except Exception as e:
        print(f"[notify error] {e}")

def open_cap(index):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def encode_jpeg(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return buf.tobytes()

def put_text(frame, text, pos, color, scale=0.45, thickness=1):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,   thickness, cv2.LINE_AA)

def front_capture_thread():
    global latest_front_frame
    low_ear_since = head_bad_since = low_blink_since = shrug_since = None
    blink_times = []; in_blink = False
    baseline_l_gap = baseline_r_gap = None
    start_time = time.time()
    cap = open_cap(1)

    with _fm_front, _pose_front:
        while True:
            if recal_front.is_set():
                baseline_l_gap = baseline_r_gap = None
                recal_front.clear()

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05); continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            now   = time.time()
            elapsed = now - start_time
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            face_res = _fm_front.process(rgb)
            pose_res = _pose_front.process(rgb)

            rgb.flags.writeable = True
            annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            ear_val = pitch = roll = 0.0
            blink_rate = low_ear_secs = head_secs = blink_low_secs = shrug_secs = 0.0
            calibrated_front = False

            if face_res.multi_face_landmarks:
                lm = face_res.multi_face_landmarks[0].landmark
                ear_val = (calc_ear(lm, LEFT_EYE, w, h) +
                           calc_ear(lm, RIGHT_EYE, w, h)) / 2

                if ear_val < BLINK_THRESH:
                    in_blink = True
                elif in_blink:
                    blink_times.append(now); in_blink = False
                window = min(elapsed, 60)
                blink_times = [t for t in blink_times if now-t <= window]
                blink_rate = (len(blink_times)*60/window) if window > 0 else 0.0

                pitch, _, roll = calc_head_pose(lm, w, h)
                head_bad = abs(pitch) > HEAD_PITCH_THRESH or abs(roll) > HEAD_ROLL_THRESH

                low_ear_since   = low_ear_since   or now if ear_val < EAR_THRESH else None
                head_bad_since  = head_bad_since  or now if head_bad              else None
                low_blink_since = low_blink_since or now if (
                    elapsed >= MIN_BLINK_WINDOW and blink_rate < LOW_BLINK_RATE) else None

                low_ear_secs   = (now - low_ear_since)   if low_ear_since   else 0.0
                head_secs      = (now - head_bad_since)  if head_bad_since  else 0.0
                blink_low_secs = (now - low_blink_since) if low_blink_since else 0.0

                # MediaPipe native draws — tessellation + contours + irises
                mp_drawing.draw_landmarks(
                    annotated,
                    face_res.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
                mp_drawing.draw_landmarks(
                    annotated,
                    face_res.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                )
                mp_drawing.draw_landmarks(
                    annotated,
                    face_res.multi_face_landmarks[0],
                    mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

                eye_color = (
                    (0, 220, 100) if ear_val >= EAR_THRESH  else
                    (0, 200, 245) if ear_val >= BLINK_THRESH else
                    (30,  30, 255)
                )
                for eye_idx in [LEFT_EYE, RIGHT_EYE]:
                    pts = np.array([[int(lm[i].x*w), int(lm[i].y*h)] for i in eye_idx])
                    cx, cy = pts.mean(axis=0).astype(int)
                    rx = int((pts[:,0].max() - pts[:,0].min()) / 2) + 5
                    ry = int((pts[:,1].max() - pts[:,1].min()) / 2) + 7
                    cv2.ellipse(annotated, (cx,cy), (rx,ry), 0, 0, 360, eye_color, 2, cv2.LINE_AA)

            if pose_res.pose_landmarks:
                plm = pose_res.pose_landmarks.landmark
                curr_l = plm[11].y - plm[7].y
                curr_r = plm[12].y - plm[8].y
                if baseline_l_gap is None:
                    baseline_l_gap = curr_l
                    baseline_r_gap = curr_r
                calibrated_front = True

                is_shrug = (curr_l < baseline_l_gap * SHRUG_RATIO or
                            curr_r < baseline_r_gap * SHRUG_RATIO)
                shrug_since = shrug_since or now if is_shrug else None
                shrug_secs  = (now - shrug_since) if shrug_since else 0.0

                # MediaPipe native pose draw
                mp_drawing.draw_landmarks(
                    annotated,
                    pose_res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Shoulder-to-shoulder connector, colored by shrug state
                l11 = pose_res.pose_landmarks.landmark[11]
                r12 = pose_res.pose_landmarks.landmark[12]
                sx1, sy1 = int(l11.x*w), int(l11.y*h)
                sx2, sy2 = int(r12.x*w), int(r12.y*h)
                shrug_col = (30, 30, 255) if is_shrug else (0, 220, 100)
                cv2.line(annotated, (sx1,sy1), (sx2,sy2), shrug_col, 2, cv2.LINE_AA)
                lx = (sx1+sx2)//2 - 30
                ly = min(sy1,sy2) - 12
                put_text(annotated, "SHRUG" if is_shrug else "SHOULDERS OK",
                         (lx, ly), shrug_col, 0.5, 1)

            # ── HUD
            ear_col  = (0,220,100) if ear_val>=EAR_THRESH  else (0,200,245) if ear_val>=BLINK_THRESH else (30,30,255)
            blnk_col = (0,220,100) if blink_rate>=15       else (0,200,245) if blink_rate>=10        else (30,30,255)
            pit_col  = (0,220,100) if abs(pitch)<=25       else (30,30,255)
            rol_col  = (0,220,100) if abs(roll)<=25        else (30,30,255)
            y = 20
            for text, col in [
                (f"EAR {ear_val:.3f}", ear_col),
                (f"BLINK {blink_rate:.0f}/min", blnk_col),
                (f"PITCH {abs(pitch):.1f}", pit_col),
                (f"ROLL  {abs(roll):.1f}",  rol_col),
            ]:
                put_text(annotated, text, (8, y), col); y += 18

            with frame_lock_front:
                latest_front_frame = encode_jpeg(annotated)

            with state_lock:
                s = shared_state
                s["ear"]              = round(ear_val, 3)
                s["blink_rate"]       = round(blink_rate, 1)
                s["pitch"]            = round(pitch, 1)
                s["roll"]             = round(roll, 1)
                s["low_ear_secs"]     = round(low_ear_secs, 1)
                s["head_secs"]        = round(head_secs, 1)
                s["blink_low_secs"]   = round(blink_low_secs, 1)
                s["shrug_secs"]       = round(shrug_secs, 1)
                s["calibrated_front"] = calibrated_front

            time.sleep(0.033)

def side_capture_thread():
    global latest_side_frame
    bad_side_since = None
    cap = open_cap(0)

    with _pose_side:
        while True:
            if recal_side.is_set():
                recal_side.clear()

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05); continue

            h, w = frame.shape[:2]
            now  = time.time()
            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            pose_res = _pose_side.process(rgb)
            rgb.flags.writeable = True
            annotated = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            neck_angle = torso_angle = bad_side_secs = 0.0
            calibrated_side = False

            if pose_res.pose_landmarks:
                plm = pose_res.pose_landmarks.landmark
                calibrated_side = True

                ear_lm = plm[mp_pose.PoseLandmark.LEFT_EAR]
                sho_lm = plm[mp_pose.PoseLandmark.LEFT_SHOULDER]
                hip_lm = plm[mp_pose.PoseLandmark.LEFT_HIP]

                ear_pos = (int(ear_lm.x*w), int(ear_lm.y*h))
                sho_pos = (int(sho_lm.x*w), int(sho_lm.y*h))
                hip_pos = (int(hip_lm.x*w), int(hip_lm.y*h))

                neck_angle  = calc_angle(ear_pos, sho_pos)
                torso_angle = calc_angle(sho_pos, hip_pos)
                bad_side    = (neck_angle > NECK_ANGLE_THRESH or
                               torso_angle > TORSO_ANGLE_THRESH)

                bad_side_since = bad_side_since or now if bad_side else None
                bad_side_secs  = (now - bad_side_since) if bad_side_since else 0.0

                # MediaPipe native pose skeleton
                mp_drawing.draw_landmarks(
                    annotated,
                    pose_res.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                spine_col = (30, 30, 255) if bad_side else (0, 220, 100)
                cv2.line(annotated, ear_pos, sho_pos, spine_col, 3, cv2.LINE_AA)
                cv2.line(annotated, sho_pos, hip_pos, spine_col, 3, cv2.LINE_AA)

                for pt, label in [(ear_pos,"EAR"),(sho_pos,"SHO"),(hip_pos,"HIP")]:
                    cv2.circle(annotated, pt, 9, spine_col, -1, cv2.LINE_AA)
                    cv2.circle(annotated, pt, 9, (255,255,255), 1, cv2.LINE_AA)
                    put_text(annotated, label, (pt[0]+12, pt[1]+5), spine_col, 0.45)

                # vertical reference from hip upward
                cv2.line(annotated,
                         (hip_pos[0], hip_pos[1]),
                         (hip_pos[0], max(0, ear_pos[1]-10)),
                         (180,180,180), 1, cv2.LINE_AA)

                # angle readout near shoulder
                put_text(annotated,
                         f"N:{neck_angle:.0f}  T:{torso_angle:.0f}",
                         (max(0, sho_pos[0]-60), sho_pos[1]-18),
                         spine_col, 0.48)

                # status banner when bad posture
                if bad_side:
                    msg = ("NECK FORWARD"    if neck_angle>NECK_ANGLE_THRESH and torso_angle<=TORSO_ANGLE_THRESH else
                           "TORSO TILTED"    if torso_angle>TORSO_ANGLE_THRESH and neck_angle<=NECK_ANGLE_THRESH else
                           "HUNCHING \u2014 SIT UP")
                    tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0][0]
                    bx = w//2 - tw//2
                    cv2.rectangle(annotated, (bx-8, 6), (bx+tw+8, 30), (20,20,20), -1)
                    cv2.rectangle(annotated, (bx-8, 6), (bx+tw+8, 30), spine_col, 1)
                    cv2.putText(annotated, msg, (bx, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, spine_col, 2, cv2.LINE_AA)

            # HUD
            nk_col = (0,220,100) if neck_angle<=NECK_ANGLE_THRESH   else (30,30,255)
            to_col = (0,220,100) if torso_angle<=TORSO_ANGLE_THRESH else (30,30,255)
            put_text(annotated, f"NECK  {neck_angle:.1f}", (8, 20), nk_col)
            put_text(annotated, f"TORSO {torso_angle:.1f}", (8, 38), to_col)

            with frame_lock_side:
                latest_side_frame = encode_jpeg(annotated)

            with state_lock:
                s = shared_state
                s["neck_angle"]      = round(neck_angle, 1)
                s["torso_angle"]     = round(torso_angle, 1)
                s["bad_side_secs"]   = round(bad_side_secs, 1)
                s["calibrated_side"] = calibrated_side

            time.sleep(0.033)

# alert 
def alert_thread():
    while True:
        now = time.time()
        with state_lock:
            s = shared_state
            les  = s["low_ear_secs"];  hs = s["head_secs"]
            bls  = s["blink_low_secs"]; ss = s["shrug_secs"]
            bds  = s["bad_side_secs"]

        fa = les>=LOW_EAR_DURATION or hs>=HEAD_DURATION or bls>=LOW_BLINK_DURATION
        pa = ss  >= SHRUG_DURATION
        sa = bds >= BAD_SIDE_DURATION

        alerts = []
        notifs = []
        if fa: alerts.append("Fatigue detected — take a break.")
        if pa: alerts.append("Shoulders too high — relax them.")
        if sa: alerts.append("Hunching detected — sit up straight.")

        if check_notif("ear",   les>=LOW_EAR_DURATION,  now): mac_notify("AL-I · Eye Strain","Your eyes are narrowing."); notifs.append("low_ear")
        if check_notif("head",  hs>=HEAD_DURATION,       now): mac_notify("AL-I · Head Tilt","Head tilt detected.");       notifs.append("head_tilt")
        if check_notif("blink", bls>=LOW_BLINK_DURATION, now): mac_notify("AL-I · Blink Rate","Blink more often.");        notifs.append("low_blink")
        if check_notif("shrug", ss>=SHRUG_DURATION,      now): mac_notify("AL-I · Posture","Relax your shoulders.");      notifs.append("shrug")
        if check_notif("side",  bds>=BAD_SIDE_DURATION,  now): mac_notify("AL-I · Posture","Sit up straight.");           notifs.append("hunching")

        with state_lock:
            shared_state.update({"fatigue_alert":fa,"posture_alert":pa,
                                  "side_posture_alert":sa,"alerts":alerts,"notifications":notifs})
        time.sleep(0.5)

# flask routes
@app.route("/")
def index_route():
    return "AL-I backend running. Open index.html in your browser.", 200

@app.route("/stream")
def stream():
    def gen():
        while True:
            with state_lock:
                payload = {k: shared_state[k] for k in [
                    "ear","blink_rate","pitch","roll",
                    "neck_angle","torso_angle","shrug_secs",
                    "low_ear_secs","head_secs","blink_low_secs","bad_side_secs",
                    "fatigue_alert","posture_alert","side_posture_alert",
                    "alerts","notifications","calibrated_front","calibrated_side",
                ]}
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.1)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no",
                             "Access-Control-Allow-Origin":"*"})

def mjpeg_gen(lock, get_frame):
    while True:
        with lock:
            frame = get_frame()
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        time.sleep(0.033)

@app.route("/feed/front")
def feed_front():
    return Response(mjpeg_gen(frame_lock_front, lambda: latest_front_frame),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Access-Control-Allow-Origin":"*"})

@app.route("/feed/side")
def feed_side():
    return Response(mjpeg_gen(frame_lock_side, lambda: latest_side_frame),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Access-Control-Allow-Origin":"*"})

@app.route("/recalibrate",       methods=["POST"])
def recalibrate():       recal_front.set(); recal_side.set();  return jsonify({"ok":True})
@app.route("/recalibrate_front", methods=["POST"])
def recalibrate_front(): recal_front.set();                    return jsonify({"ok":True})
@app.route("/recalibrate_side",  methods=["POST"])
def recalibrate_side():  recal_side.set();                     return jsonify({"ok":True})

# entry point
if __name__ == "__main__":
    for t in [front_capture_thread, side_capture_thread, alert_thread]:
        threading.Thread(target=t, daemon=True).start()
    print("AL-I backend  →  http://127.0.0.1:5050")
    print("  cam 0  iPhone/continuity  →  /feed/side")
    print("  cam 1  MacBook            →  /feed/front")
    app.run(host="127.0.0.1", port=5050, threaded=True)
