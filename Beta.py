import cv2
import os
import numpy as np
import mediapipe as mp
import vlc
from keras.models import load_model # type: ignore
import ctypes
import time
import win32gui
import win32con

# ==== Cấu hình ==== 
VLC_PATH = r"C:\\Program Files\\VideoLAN\\VLC"
MODEL_PATH = r"D:\\VideoGestureControl\\gesture_model_v7.h5"
VIDEO_FOLDER = r"D:\\VideoGestureControl\\Video"
GESTURE_LABELS = ['Next', 'Pause', 'Play', 'Start']

# ==== Load libvlc.dll ==== 
os.environ['PATH'] += os.pathsep + VLC_PATH
ctypes.CDLL(os.path.join(VLC_PATH, "libvlc.dll"))

# ==== Load model ==== 
model = load_model(MODEL_PATH)

# ==== MediaPipe ==== 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

def landmarks_to_vector(hand_landmarks):
    vector = []
    for lm in hand_landmarks.landmark[:21]:
        vector.extend([lm.x, lm.y])
    return np.array(vector, dtype=np.float32)

def predict_gesture(hand_landmarks):
    if hand_landmarks is None:
        return None

    input_vector = landmarks_to_vector(hand_landmarks).reshape(1, 42)
    predictions = model.predict(input_vector, verbose=0)
    predicted_index = np.argmax(predictions)
    gesture = GESTURE_LABELS[predicted_index]
    print(f"✅ Gesture: {gesture} (Index: {predicted_index})")

    with open("gesture_log.txt", "a") as log:
        log.write(f"{time.ctime()}: {gesture}\n")
    return gesture

# ==== VLC Setup ==== 
vlc_instance = vlc.Instance()
player = vlc_instance.media_player_new()

video_list = [os.path.join(VIDEO_FOLDER, f) for f in os.listdir(VIDEO_FOLDER) if f.endswith('.mp4')]
video_index = 0
gesture_state = "Pause"
previous_gesture = None

def set_vlc_window_position():
    hwnd = player.get_hwnd()
    if hwnd:
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 800, 600, win32con.SWP_NOZORDER | win32con.SWP_SHOWWINDOW)

# ==== Phát video bằng VLC ==== 
def play_video_vlc(index):
    media = vlc_instance.media_new(video_list[index])
    player.set_media(media)
    player.play()
    time.sleep(1)

def play_video_vlc_with_camera(index):
    global cap, gesture_state, video_index, previous_gesture

    media = vlc_instance.media_new(video_list[index])
    player.set_media(media)
    player.set_fullscreen(False)  # Tắt chế độ toàn màn hình
    player.play()
    player.video_set_scale(1.5)  # Tăng kích thước video lên lần

    time.sleep(1)  # Wait for the VLC window to initialize

    while True:
        # Đọc khung hình từ camera
        ret_cam, frame_cam = cap.read()
        if not ret_cam:
            print("❌ Cannot read webcam.")
            break

        # Hiển thị camera ở góc phải dưới
        frame_cam = cv2.flip(frame_cam, 1)
        cam_resized = cv2.resize(frame_cam, (320, 240))
        cv2.imshow("Camera Feed", cam_resized)

        # Xử lý cử chỉ
        rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_cam, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = predict_gesture(hand_landmarks)
                if gesture != previous_gesture:
                    previous_gesture = gesture
                    if gesture == "Start" or gesture == "Play":
                        gesture_state = "Play"
                        player.play()
                    elif gesture == "Pause":
                        gesture_state = "Pause"
                        player.pause()
                    elif gesture == "Next":
                        video_index = (video_index + 1) % len(video_list)
                        player.stop()
                        play_video_vlc_with_camera(video_index)
                        return

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC để thoát
            cap.release()
            player.stop()
            cv2.destroyAllWindows()
            break

cap = cv2.VideoCapture(0)

def on_exit(event=None):
    player.stop()
    cap.release()

# ==== Khởi động ==== 
play_video_vlc_with_camera(video_index)
