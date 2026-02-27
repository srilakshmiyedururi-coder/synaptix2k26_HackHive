import tkinter as tk
from tkinter import messagebox
import threading
import cv2
import pyautogui
from PIL import Image
import numpy as np
import os
import urllib.request

# urls for task models
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _ensure_model(path: str, url: str) -> str:
    """Download model from `url` if it doesn't already exist and return local path."""
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:
            print(f"failed to download model {url}: {e}")
            raise
    return path

# Import mediapipe normally and derive submodules from mp.solutions
# Attempt to import mediapipe; later code will check availability.
try:
    import mediapipe as mp
    mp_available = True
    # note: newer versions of mediapipe (0.10+) expose tasks instead of solutions
    mp_has_solutions = hasattr(mp, "solutions")
    mp_has_tasks = hasattr(mp, "tasks")
except ImportError:
    mp_available = False
    mp_has_solutions = False
    mp_has_tasks = False

# report availability
print(f"mediapipe available={mp_available} has_solutions={mp_has_solutions} has_tasks={mp_has_tasks}")
# also log to disk so we can inspect later
try:
    with open(r"d:\dataset\mp_log.txt", "a") as _f:
        _f.write(f"availability={mp_available}, solutions={mp_has_solutions}, tasks={mp_has_tasks}\n")
except Exception:
    pass

# only define aliases if we can access solutions
if mp_available and mp_has_solutions:
    mp_hands = mp.solutions.hands
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
else:
    # placeholders to avoid NameErrors
    mp_hands = None
    mp_face_mesh = None
    mp_drawing = None

# 1. Hand Volume Control

def _hand_volume_control():
    cap = cv2.VideoCapture(0)
    if not mp_available or not (mp_has_solutions or mp_has_tasks):
        print("cannot start hand volume control; mediapipe is unavailable")
        cap.release()
        return

    # prepare for tasks-based detection if needed
    landmarker = None
    if mp_has_tasks and not mp_has_solutions:
        model_path = _ensure_model("hand_landmarker.task", HAND_MODEL_URL)
        landmarker = mp.tasks.vision.HandLandmarker.create_from_model_path(model_path)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if mp_has_solutions:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = mp_hands.Hands(  # pylint: disable=not-callable
                    max_num_hands=1, min_detection_confidence=0.7
                ).process(rgb)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        thumb = hand_landmarks.landmark[4]
                        index = hand_landmarks.landmark[8]
                        x1, y1 = int(index.x * w), int(index.y * h)
                        x2, y2 = int(thumb.x * w), int(thumb.y * h)
                        cv2.circle(frame, (x1, y1), 10, (0, 255, 255), -1)
                        cv2.circle(frame, (x2, y2), 10, (0, 0, 255), -1)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        distance = np.hypot(x2 - x1, y2 - y1)
                        if distance > 50:
                            pyautogui.press("volumeup")
                        else:
                            pyautogui.press("volumedown")
            else:
                # tasks API branch
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect(mp_img)
                if result.hand_landmarks:
                    for hand in result.hand_landmarks:
                        x1 = int(hand[8].x * w)
                        y1 = int(hand[8].y * h)
                        x2 = int(hand[4].x * w)
                        y2 = int(hand[4].y * h)
                        cv2.circle(frame, (x1, y1), 10, (0, 255, 255), -1)
                        cv2.circle(frame, (x2, y2), 10, (0, 0, 255), -1)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        distance = np.hypot(x2 - x1, y2 - y1)
                        if distance > 50:
                            pyautogui.press("volumeup")
                        else:
                            pyautogui.press("volumedown")

            cv2.imshow("Hand Volume Control - Press ESC to Exit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if landmarker:
            landmarker.close()

def button1_click():
    if not mp_available or not (mp_has_solutions or mp_has_tasks):
        messagebox.showerror("Dependency Error", "Mediapipe is not installed or unavailable.")
        return
    print('Hand Volume Control Activated')
    threading.Thread(target=_hand_volume_control, daemon=True).start()

# 2. Virtual Mouse

def _virtual_mouse():
    cap = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()
    if not mp_available or not (mp_has_solutions or mp_has_tasks):
        print("cannot start virtual mouse; mediapipe is unavailable")
        cap.release()
        return

    landmarker = None
    if mp_has_tasks and not mp_has_solutions:
        model_path = _ensure_model("hand_landmarker.task", HAND_MODEL_URL)
        landmarker = mp.tasks.vision.HandLandmarker.create_from_model_path(model_path)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if mp_has_solutions:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7).process(rgb)
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        index = hand_landmarks.landmark[8]
                        thumb = hand_landmarks.landmark[4]
                        ix, iy = int(index.x * w), int(index.y * h)
                        tx, ty = int(thumb.x * w), int(thumb.y * h)
                        screen_x = screen_w / w * ix
                        screen_y = screen_h / h * iy
                        pyautogui.moveTo(screen_x, screen_y)
                        if np.hypot(ix - tx, iy - ty) < 40:
                            pyautogui.click()
                            pyautogui.sleep(0.5)
            else:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect(mp_img)
                if result.hand_landmarks:
                    for hand in result.hand_landmarks:
                        ix = int(hand[8].x * w)
                        iy = int(hand[8].y * h)
                        tx = int(hand[4].x * w)
                        ty = int(hand[4].y * h)
                        screen_x = screen_w / w * ix
                        screen_y = screen_h / h * iy
                        pyautogui.moveTo(screen_x, screen_y)
                        if np.hypot(ix - tx, iy - ty) < 40:
                            pyautogui.click()
                            pyautogui.sleep(0.5)

            cv2.imshow("Virtual Mouse - Press ESC to Exit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if landmarker:
            landmarker.close()

def button2_click():
    if not mp_available or not (mp_has_solutions or mp_has_tasks):
        messagebox.showerror("Dependency Error", "Mediapipe is not installed or unavailable.")
        return
    print('Virtual Mouse Activated')
    threading.Thread(target=_virtual_mouse, daemon=True).start()

# 3. Eye Controlled Mouse

def _eye_controlled_mouse():
    cap = cv2.VideoCapture(0)
    screen_w, screen_h = pyautogui.size()
    if not mp_available or not (mp_has_solutions or mp_has_tasks):
        print("cannot start eye mouse; mediapipe is unavailable")
        cap.release()
        return

    face_landmarker = None
    if mp_has_tasks and not mp_has_solutions:
        model_path = _ensure_model("face_landmarker.task", FACE_MODEL_URL)
        face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_model_path(model_path)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            if mp_has_solutions:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = mp_face_mesh.FaceMesh(refine_landmarks=True).process(rgb)
                if result.multi_face_landmarks:
                    landmarks = result.multi_face_landmarks[0].landmark
                    for id, lm in enumerate(landmarks[474:478]):
                        x, y = int(lm.x * w), int(lm.y * h)
                        if id == 1:
                            screen_x = screen_w / w * x
                            screen_y = screen_h / h * y
                            pyautogui.moveTo(screen_x, screen_y)
                    left = [landmarks[145], landmarks[159]]
                    if (left[0].y - left[1].y) < 0.004:
                        pyautogui.click()
                        pyautogui.sleep(1)
            else:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                                  data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = face_landmarker.detect(mp_img)
                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    # same indices as earlier face_mesh
                    for id, idx in enumerate(range(474, 478)):
                        lm = landmarks[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        if id == 1:
                            screen_x = screen_w / w * x
                            screen_y = screen_h / h * y
                            pyautogui.moveTo(screen_x, screen_y)
                    left = [landmarks[145], landmarks[159]]
                    if (left[0].y - left[1].y) < 0.004:
                        pyautogui.click()
                        pyautogui.sleep(1)

            cv2.imshow("Eye Mouse - Press ESC to Exit", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if face_landmarker:
            face_landmarker.close()

def button3_click():
    if not mp_available or not (mp_has_solutions or mp_has_tasks):
        messagebox.showerror("Dependency Error", "Mediapipe is not installed or unavailable.")
        return
    print('Eye Controlled Mouse Activated')
    threading.Thread(target=_eye_controlled_mouse, daemon=True).start()

# 4. Capture Photo
def _capture_photo():
    cap = cv2.VideoCapture(0)
    count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        cv2.putText(frame, f"Wait: {50-count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Photo Capture", frame)
        count += 1
        if count >= 50:
            cv2.imwrite("captured_photo.jpg", frame)
            break
        if cv2.waitKey(1) & 0xFF == 27: break
    cap.release()
    cv2.destroyAllWindows()
    Image.open("captured_photo.jpg").show()

def button4_click():
    print('Photo Capture Activated')
    threading.Thread(target=_capture_photo, daemon=True).start()

# GUI Design
root = tk.Tk()
root.title("AI Gesture Controller")
root.geometry("300x400")

tk.Label(root, text="Select a Feature", font=("Arial", 12, "bold")).pack(pady=20)
btn_params = {"width": 20, "height": 2, "font": ("Arial", 10)}

tk.Button(root, text="Hand Volume Control", command=button1_click, bg="#FFADAD", **btn_params).pack(pady=5)
tk.Button(root, text="Virtual Mouse", command=button2_click, bg="#CAFFBF", **btn_params).pack(pady=5)
tk.Button(root, text="Eye Controlled Mouse", command=button3_click, bg="#9BF6FF", **btn_params).pack(pady=5)
tk.Button(root, text="Capture Photo", command=button4_click, bg="#FFD6A5", **btn_params).pack(pady=5)
root.mainloop()