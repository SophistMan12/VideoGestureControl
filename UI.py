import tkinter as tk
from tkinter import Label
import cv2
import os
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

# === Khởi tạo MediaPipe ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# === Hàm nhận diện cử chỉ tay ===
def detect_gesture(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    if (thumb_tip.x < index_tip.x and abs(thumb_tip.y - index_tip.y) < 0.05 and
        middle_tip.y < wrist.y and ring_tip.y < wrist.y and pinky_tip.y < wrist.y):
        return "Start"
    elif (index_tip.y < wrist.y and middle_tip.y < wrist.y and ring_tip.y < wrist.y and
          pinky_tip.y < wrist.y and thumb_tip.y < wrist.y and
          middle_tip.y < index_tip.y and middle_tip.y < ring_tip.y and middle_tip.y < pinky_tip.y):
        return "Pause"
    elif (thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and
          thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y):
        return "Play"
    elif abs(thumb_tip.x - index_tip.x) > 0.1 and abs(thumb_tip.y - index_tip.y) < 0.05:
        return "Next"
    else:
        return "Unknown"

# === Cập nhật khung hình ===
def update_frame():
    global gesture_state, video_index, video, cap

    # Đọc frame camera
    ret_cam, frame_cam = cap.read()
    if ret_cam:
        frame_cam = cv2.flip(frame_cam, 1)
        rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesture = "Unknown"
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_cam, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = detect_gesture(hand_landmarks)

                if gesture == "Start":
                    gesture_state = "Play"
                elif gesture == "Pause":
                    gesture_state = "Pause"
                elif gesture == "Play":
                    gesture_state = "Play"
                elif gesture == "Next":
                    video_index = (video_index + 1) % len(video_list)
                    video = cv2.VideoCapture(video_list[video_index])
                    gesture_state = "Play"

        # Hiển thị camera
        cam_resized = cv2.resize(frame_cam, (camera_width, camera_height))
        img_cam = ImageTk.PhotoImage(Image.fromarray(cam_resized))
        camera_label.config(image=img_cam)
        camera_label.image = img_cam

    # Đọc frame video nếu ở trạng thái Play
    if gesture_state == "Play" and video is not None and video.isOpened():
        ret_vid, frame_vid = video.read()
        if ret_vid:
            frame_vid = cv2.resize(frame_vid, (screen_width, screen_height))
            frame_vid = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
            img_vid = ImageTk.PhotoImage(Image.fromarray(frame_vid))
            video_canvas.create_image(0, 0, image=img_vid, anchor="nw")
            video_canvas.image = img_vid
        else:
            # Nếu hết video thì dừng
            gesture_state = "Pause"

    root.after(30, update_frame)

# === Thoát ứng dụng ===
def exit_app(event):
    if cap: cap.release()
    if video: video.release()
    root.destroy()

# === Khởi tạo giao diện ===
cap = cv2.VideoCapture(0)
camera_width, camera_height = 320, 240

video_folder = './Video'  # Đảm bảo thư mục này tồn tại
video_list = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]
video_index = 0
video = cv2.VideoCapture(video_list[video_index]) if video_list else None
gesture_state = "Pause"

# Tạo cửa sổ Tkinter
root = tk.Tk()
root.attributes('-fullscreen', True)
root.configure(bg='black')

# Kích thước màn hình
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Canvas để hiển thị video nền
video_canvas = tk.Canvas(root, width=screen_width, height=screen_height, highlightthickness=0)
video_canvas.pack(fill="both", expand=True)

# Camera ở góc phải dưới
camera_label = Label(root)
camera_label.place(relx=1.0, rely=1.0, anchor='se', x=-10, y=-10)

# Bắt đầu cập nhật
update_frame()

# Bấm ESC để thoát
root.bind('<Escape>', exit_app)

# Chạy vòng lặp giao diện
root.mainloop()
