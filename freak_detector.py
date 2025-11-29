 # freak_detector.py (IMPROVED: QUICK NOD→BLINK + HAND MESH + STABLE GIFS)

import cv2
import mediapipe as mp
import imageio
import numpy as np
import time
from pathlib import Path
from collections import deque, Counter
import math

# ---------------------------
# Config
# ---------------------------
CAM_W, CAM_H = 640, 480
GIF_W, GIF_H = CAM_W, CAM_H
COMBINED_W = CAM_W + GIF_W
COMBINED_H = max(CAM_H, GIF_H)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

GIF_FILENAMES = [
    "tuff.gif",            # 0 - weighing hands gesture (tuff)
    "thinking.gif",        # 1
    "idea.gif",            # 2
    "blink.gif",           # 3 (quick nod)
    "tongue.gif",          # 4
    "hands_up.gif",        # 5
    "rubbing_hands.gif",   # 6
]

GESTURE_TO_GIF = {
    "tuff": 0,
    "thinking": 1,
    "idea": 2,
    "blink": 3,
    "nod": 3,
    "tongue": 4,
    "hands_up": 5,
    "rubbing": 6,
    "neutral": -1,
}

PROCESS_EVERY_N_FRAMES = 1

BLINK_EAR_THRESH = 0.18
MOUTH_OPEN_THRESH = 0.045
FINGER_MOUTH_DIST_THRESH = 0.06
HANDS_JOINED_DIST = 0.08
HANDS_UP_Y_DIFF = -0.10
INDEX_UP_Y_MARGIN = -0.02
WEIGHING_DELTA = 0.06
NOD_HISTORY = 12
NOD_SPEED_THRESH = 0.018
GESTURE_HISTORY_LEN = 5

# ---------------------------
# Helpers
# ---------------------------
def load_gif_frames(path: Path, size=(GIF_W, GIF_H)):
    if not path.exists():
        raise FileNotFoundError(f"GIF not found: {path}")
    reader = imageio.get_reader(str(path))
    frames = []
    for fr in reader:
        fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        fr = cv2.resize(fr, size)
        frames.append(fr)
    return frames

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def eye_aspect_ratio(lm, e):
    try:
        l = lm[e['outer']]
        r = lm[e['inner']]
        t = lm[e['top']]
        b = lm[e['bottom']]
    except:
        return 1.0
    h = dist((l.x,l.y),(r.x,r.y))
    v = dist((t.x,t.y),(b.x,b.y))
    return v/h if h else 1.0

def mouth_gap(face):
    try:
        top = face[13]
        bot = face[14]
        left = face[234]
        right = face[454]
    except:
        return 0
    gap = dist((top.x,top.y),(bot.x,bot.y))
    width = dist((left.x,left.y),(right.x,right.y))
    return gap/width if width else 0

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
L_EYE = {"outer":33,"inner":133,"top":159,"bottom":145}
R_EYE = {"outer":362,"inner":263,"top":386,"bottom":374}
NOSE = 1

INDEX_TIP = 8
INDEX_PIP = 6
WRIST = 0

# ---------------------------
# Main
# ---------------------------
def main():
    # Load GIFs
    gif_frames_list = []
    gif_lengths = []
    for filename in GIF_FILENAMES:
        try:
            frames = load_gif_frames(ASSETS_DIR / filename)
        except:
            frames = [np.zeros((GIF_H,GIF_W,3),dtype=np.uint8)]
        gif_frames_list.append(frames)
        gif_lengths.append(len(frames))

    BLACK_SCREEN = np.zeros((GIF_H, GIF_W, 3), dtype=np.uint8)

    gif_idx = 0
    gif_frame_idx = 0
    gif_fps = 12
    gif_interval = 1.0/gif_fps
    last_gif_time = time.time()
    displayed_gif_frame = gif_frames_list[gif_idx][gif_frame_idx].copy()

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    cap = cv2.VideoCapture(0)
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)

    frame_count=0
    nose_history = deque(maxlen=NOD_HISTORY)
    gesture_history = deque(maxlen=GESTURE_HISTORY_LEN)

    print("Running Freak Detector | Press q to quit")

    while True:
        ret,frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame,1)
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            results = holistic.process(rgb)
        frame_count += 1

        gesture="neutral"
        face=None
        now_time = time.time()

        # --------------------------------------------
        # FACE
        # --------------------------------------------
        if results.face_landmarks:
            face = results.face_landmarks.landmark
            nose_history.append(face[NOSE].y)

            # MOUTH/TONGUE
            if mouth_gap(face) > MOUTH_OPEN_THRESH:
                gesture="tongue"

        # --------------------------------------------
        # QUICK NOD → BLINK
        # --------------------------------------------
        blink_detected = False
        nod_detected = False
        if len(nose_history) >= 3:
            dy = nose_history[-1] - nose_history[-3]  # small frame interval
            speed = dy / 3.0
            if speed > 0.02:  # quick downward movement
                blink_detected = True
            elif abs(nose_history[-1]-nose_history[0]) > NOD_SPEED_THRESH:
                nod_detected = True

        if blink_detected:
            gesture = "blink"
        elif nod_detected:
            gesture = "nod"

        # HANDS
        lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
        rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None

        # Draw hand mesh
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # IDEA
        if lh or rh:
            for hand in (lh,rh):
                if not hand: continue
                tip = hand[INDEX_TIP]
                pip = hand[INDEX_PIP]
                if (tip.y - pip.y) < INDEX_UP_Y_MARGIN:
                    gesture="idea"

        # THINKING
        if face and (lh or rh):
            mouth_center=((face[13].x+face[14].x)/2, (face[13].y+face[14].y)/2)
            for hand in (lh,rh):
                if not hand: continue
                tip = hand[INDEX_TIP]
                if dist((tip.x,tip.y), mouth_center) < FINGER_MOUTH_DIST_THRESH:
                    gesture="thinking"

        # RUBBING
        if lh and rh:
            d = dist((lh[INDEX_TIP].x,lh[INDEX_TIP].y),(rh[INDEX_TIP].x,rh[INDEX_TIP].y))
            if d < HANDS_JOINED_DIST:
                gesture="rubbing"

        # HANDS_UP (using hand mesh)
        if face and lh and rh:
            nose_y = face[NOSE].y
            lh_y = np.mean([lh[i].y for i in [0,5,9,13,17]])
            rh_y = np.mean([rh[i].y for i in [0,5,9,13,17]])
            avg_y = (lh_y + rh_y)/2
            if avg_y < nose_y + HANDS_UP_Y_DIFF:
                gesture="hands_up"

        # TUFF (using hand mesh)
        if lh and rh:
            lh_y = np.mean([lh[i].y for i in [0,5,9,13,17]])
            rh_y = np.mean([rh[i].y for i in [0,5,9,13,17]])
            hands_distance = dist((lh[WRIST].x, lh[WRIST].y), (rh[WRIST].x, rh[WRIST].y))
            if abs(lh_y - rh_y) > WEIGHING_DELTA and hands_distance > 0.05:
                gesture="tuff"

        # ---------------------------
        # Gesture stabilization
        # ---------------------------
        gesture_history.append(gesture)
        chosen = Counter(gesture_history).most_common(1)[0][0]

        target_gif = GESTURE_TO_GIF.get(chosen,-1)

        # ---------------------------
        # GIF PANEL
        # ---------------------------
        if target_gif == -1:
            gif_panel = np.zeros((GIF_H,GIF_W,3),dtype=np.uint8)
        else:
            if target_gif != gif_idx:
                gif_idx = target_gif
                gif_frame_idx = 0
                displayed_gif_frame = gif_frames_list[gif_idx][gif_frame_idx].copy()
                last_gif_time = time.time()

            now = time.time()
            if now - last_gif_time >= 1.0/gif_fps:
                gif_frame_idx = (gif_frame_idx + 1) % gif_lengths[gif_idx]
                next_f = gif_frames_list[gif_idx][gif_frame_idx]
                displayed_gif_frame = cv2.addWeighted(next_f,0.6,displayed_gif_frame,0.4,0)
                last_gif_time = now

            gif_panel = displayed_gif_frame

        # ---------------------------
        # DRAW PANELS
        # ---------------------------
        vis = frame.copy()
        cv2.putText(vis,f"Gesture: {chosen}",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

        cam_panel = cv2.resize(vis,(CAM_W,CAM_H))

        combined = np.zeros((COMBINED_H,COMBINED_W,3),dtype=np.uint8)
        combined[0:CAM_H,0:CAM_W] = cam_panel
        combined[0:GIF_H,CAM_W:CAM_W+GIF_W] = gif_panel

        cv2.line(combined,(CAM_W,0),(CAM_W,COMBINED_H),(255,255,255),2)
        cv2.imshow("Freak Detector — Camera | GIF",combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()

if __name__=="__main__":
    main()

# import cv2
# import mediapipe as mp
# import imageio
# import numpy as np
# import time
# from pathlib import Path
# from collections import deque, Counter
# import math

# # ---------------------------
# # Config
# # ---------------------------
# CAM_W, CAM_H = 640, 480
# GIF_W, GIF_H = CAM_W, CAM_H
# COMBINED_W = CAM_W + GIF_W
# COMBINED_H = max(CAM_H, GIF_H)

# ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# GIF_FILENAMES = [
#     "tuff.gif",              # index 0
#     "thinking.gif",          # index 1
#     "idea.gif",              # index 2
#     "blink.gif",             # index 3 (one-shot)
#     "tongue.gif",            # index 4
#     "hands_up.gif",          # index 5
#     "rubbing_hands.gif",     # index 6
# ]

# # (gesture → gif index)
# GESTURE_TO_GIF = {
#     "tuff": 0,
#     "thinking": 1,
#     "idea": 2,
#     "blink": 3,
#     "nod": 3,
#     "tongue": 4,
#     "hands_up": 5,
#     "rubbing": 6,
#     "neutral": -1,
# }

# # 🔥 NEW: one-shot GIFs
# ONE_SHOT_GIFS = {0, 3, 4}     # tuff, blink, tongue

# PROCESS_EVERY_N_FRAMES = 1
# BLINK_EAR_THRESH = 0.18
# MOUTH_OPEN_THRESH = 0.045
# FINGER_MOUTH_DIST_THRESH = 0.06
# HANDS_JOINED_DIST = 0.08
# HANDS_UP_Y_DIFF = -0.10
# INDEX_UP_Y_MARGIN = -0.02
# WEIGHING_DELTA = 0.06
# NOD_HISTORY = 12
# NOD_SPEED_THRESH = 0.018
# GESTURE_HISTORY_LEN = 5

# # ---------------------------
# # Helpers
# # ---------------------------
# def load_gif_frames(path: Path, size=(GIF_W, GIF_H)):
#     reader = imageio.get_reader(str(path))
#     frames = []
#     for fr in reader:
#         fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
#         fr = cv2.resize(fr, size)
#         frames.append(fr)
#     return frames

# def dist(a, b):
#     return math.hypot(a[0]-b[0], a[1]-b[1])

# def mouth_gap(face):
#     try:
#         top = face[13]
#         bot = face[14]
#         left = face[234]
#         right = face[454]
#     except:
#         return 0
#     gap = dist((top.x, top.y), (bot.x, bot.y))
#     width = dist((left.x, left.y), (right.x, right.y))
#     return gap / width if width else 0

# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# NOSE = 1
# INDEX_TIP = 8
# INDEX_PIP = 6
# WRIST = 0

# # ---------------------------
# # Main
# # ---------------------------

# def main():

#     gif_frames_list = []
#     gif_lengths = []
#     for f in GIF_FILENAMES:
#         try:
#             frames = load_gif_frames(ASSETS_DIR / f)
#         except:
#             frames = [np.zeros((GIF_H, GIF_W, 3), dtype=np.uint8)]
#         gif_frames_list.append(frames)
#         gif_lengths.append(len(frames))

#     BLACK_SCREEN = np.zeros((GIF_H, GIF_W, 3), dtype=np.uint8)

#     gif_idx = 0
#     gif_frame_idx = 0
#     gif_interval = 1/12
#     last_gif_time = time.time()

#     displayed_gif_frame = gif_frames_list[gif_idx][gif_frame_idx].copy()

#     # 🔒 one-shot lock
#     gif_locked = False
#     gif_lock_end_time = 0

#     holistic = mp_holistic.Holistic(
#         min_detection_confidence=0.6,
#         min_tracking_confidence=0.6
#     )

#     cap = cv2.VideoCapture(0)
#     cap.set(3, CAM_W)
#     cap.set(4, CAM_H)

#     frame_count = 0
#     nose_history = deque(maxlen=NOD_HISTORY)
#     gesture_history = deque(maxlen=GESTURE_HISTORY_LEN)

#     print("Running Freak Detector | Press q to quit")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.flip(frame, 1)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         if frame_count % PROCESS_EVERY_N_FRAMES == 0:
#             results = holistic.process(rgb)
#         frame_count += 1

#         gesture = "neutral"
#         face = None
#         now_time = time.time()

#         # ---------------- FACE ----------------
#         if results.face_landmarks:
#             face = results.face_landmarks.landmark
#             nose_history.append(face[NOSE].y)

#             if mouth_gap(face) > MOUTH_OPEN_THRESH:
#                 gesture = "tongue"

#         # quick nod → blink
#         blink = False
#         nod = False
#         if len(nose_history) >= 3:
#             dy = nose_history[-1] - nose_history[-3]
#             speed = dy / 3
#             if speed > 0.02:
#                 blink = True
#             elif abs(nose_history[-1] - nose_history[0]) > NOD_SPEED_THRESH:
#                 nod = True

#         if blink:
#             gesture = "blink"
#         elif nod:
#             gesture = "nod"

#         # ---------------- HANDS ----------------
#         lh = results.left_hand_landmarks.landmark if results.left_hand_landmarks else None
#         rh = results.right_hand_landmarks.landmark if results.right_hand_landmarks else None

#         # draw mesh
#         if results.left_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, results.left_hand_landmarks,
#                                       mp.solutions.hands.HAND_CONNECTIONS)
#         if results.right_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, results.right_hand_landmarks,
#                                       mp.solutions.hands.HAND_CONNECTIONS)

#         # idea gesture
#         if lh or rh:
#             for hand in (lh, rh):
#                 if not hand: continue
#                 tip = hand[INDEX_TIP]
#                 pip = hand[INDEX_PIP]
#                 if (tip.y - pip.y) < INDEX_UP_Y_MARGIN:
#                     gesture = "idea"

#         # thinking: finger near mouth
#         if face and (lh or rh):
#             mouth_center = ((face[13].x + face[14].x) / 2,
#                             (face[13].y + face[14].y) / 2)
#             for hand in (lh, rh):
#                 if not hand: continue
#                 tip = hand[INDEX_TIP]
#                 if dist((tip.x, tip.y), mouth_center) < FINGER_MOUTH_DIST_THRESH:
#                     gesture = "thinking"

#         # rubbing hands
#         if lh and rh:
#             d = dist((lh[INDEX_TIP].x, lh[INDEX_TIP].y),
#                      (rh[INDEX_TIP].x, rh[INDEX_TIP].y))
#             if d < HANDS_JOINED_DIST:
#                 gesture = "rubbing"

#         # hands up
#         if face and lh and rh:
#             nose_y = face[NOSE].y
#             lh_y = np.mean([lh[i].y for i in [0,5,9,13,17]])
#             rh_y = np.mean([rh[i].y for i in [0,5,9,13,17]])
#             if (lh_y + rh_y)/2 < nose_y + HANDS_UP_Y_DIFF:
#                 gesture = "hands_up"

#         # tuff gesture
#         if lh and rh:
#             lh_y = np.mean([lh[i].y for i in [0,5,9,13,17]])
#             rh_y = np.mean([rh[i].y for i in [0,5,9,13,17]])
#             dis = dist((lh[WRIST].x, lh[WRIST].y),
#                        (rh[WRIST].x, rh[WRIST].y))
#             if abs(lh_y - rh_y) > WEIGHING_DELTA and dis > 0.05:
#                 gesture = "tuff"

#         # ---------------- Stabilize ----------------
#         gesture_history.append(gesture)
#         chosen = Counter(gesture_history).most_common(1)[0][0]
#         target_gif = GESTURE_TO_GIF.get(chosen, -1)

#         # ---------------- GIF LOGIC (one-shot included) ----------------

#         # one-shot trigger
#         if target_gif in ONE_SHOT_GIFS and not gif_locked:
#             gif_locked = True
#             gif_idx = target_gif
#             gif_frame_idx = 0
#             displayed_gif_frame = gif_frames_list[gif_idx][gif_frame_idx].copy()
#             last_gif_time = time.time()
#             gif_lock_end_time = last_gif_time + gif_lengths[gif_idx] * gif_interval

#         # while locked → play full
#         if gif_locked:
#             now = time.time()

#             if now - last_gif_time >= gif_interval:
#                 gif_frame_idx += 1

#                 if gif_frame_idx >= gif_lengths[gif_idx]:
#                     gif_locked = False
#                     chosen = "neutral"
#                     gif_panel = BLACK_SCREEN

#                     cam_panel = cv2.resize(frame, (CAM_W, CAM_H))
#                     combined = np.zeros((COMBINED_H, COMBINED_W, 3), dtype=np.uint8)
#                     combined[0:CAM_H, 0:CAM_W] = cam_panel
#                     combined[0:GIF_H, CAM_W:CAM_W+GIF_W] = gif_panel
#                     cv2.imshow("Freak Detector — Camera | GIF", combined)
#                     if cv2.waitKey(1) & 0xFF == ord('q'):
#                         break
#                     continue

#                 displayed_gif_frame = gif_frames_list[gif_idx][gif_frame_idx]
#                 last_gif_time = now

#             gif_panel = displayed_gif_frame

#         else:
#             if target_gif == -1:
#                 gif_panel = BLACK_SCREEN
#             else:
#                 if target_gif != gif_idx:
#                     gif_idx = target_gif
#                     gif_frame_idx = 0
#                     displayed_gif_frame = gif_frames_list[target_gif][gif_frame_idx].copy()
#                     last_gif_time = time.time()

#                 now = time.time()
#                 if now - last_gif_time >= gif_interval:
#                     gif_frame_idx = (gif_frame_idx + 1) % gif_lengths[gif_idx]
#                     displayed_gif_frame = gif_frames_list[gif_idx][gif_frame_idx]
#                     last_gif_time = now

#                 gif_panel = displayed_gif_frame

#         # ---------------- Display ----------------
#         vis = frame.copy()
#         cv2.putText(vis, f"Gesture: {chosen}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

#         cam_panel = cv2.resize(vis, (CAM_W, CAM_H))

#         combined = np.zeros((COMBINED_H, COMBINED_W, 3), dtype=np.uint8)
#         combined[0:CAM_H, 0:CAM_W] = cam_panel
#         combined[0:GIF_H, CAM_W:CAM_W+GIF_W] = gif_panel

#         cv2.line(combined, (CAM_W,0), (CAM_W,COMBINED_H), (255,255,255), 2)
#         cv2.imshow("Freak Detector — Camera | GIF", combined)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     holistic.close()

# if __name__ == "__main__":
#     main()
