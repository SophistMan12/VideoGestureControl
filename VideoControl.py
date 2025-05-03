import cv2
import mediapipe as mp
import numpy as np
import os
import tensorflow as tf # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import pickle
from sklearn.preprocessing import LabelEncoder


# Load model đã huấn luyện
model = load_model('D:\VideoGestureControl\Model.h5') 

# Load lại LabelEncoder nếu cần
# Nếu không có file encoder, bạn cần khớp thứ tự class giống lúc train
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Play', 'Pause', 'Next'])  # hoặc theo thứ tự bạn đã train

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

def predict_gesture(hand_landmarks):
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.append(lm.x)
        keypoints.append(lm.y)
    keypoints = np.array(keypoints)

    # Normalize input nếu cần (phải giống lúc train!)
    keypoints = np.nan_to_num(keypoints)
    keypoints = keypoints / np.max(keypoints)

    # Dự đoán
    keypoints = keypoints.reshape(1, -1)  # reshape (1,42)
    prediction = model.predict(keypoints)
    predicted_class = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label

def play_video(path):
    global gesture_state, video_index
    video = cv2.VideoCapture(path)

    while video.isOpened():
        if gesture_state == "Pause":
            cv2.waitKey(100)
        else:
            ret, frame = video.read()
            if not ret:
                break
            cv2.imshow("Video Player", frame)

        # Webcam input
        ret_cam, frame_cam = cap.read()
        if not ret_cam or frame_cam is None:
            print("❌ Cannot read webcam.")
            continue

        frame_cam = cv2.flip(frame_cam, 1)
        rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_cam, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = predict_gesture(hand_landmarks)

                if gesture == "Start":
                    gesture_state = "Play"
                elif gesture == "Pause":
                    gesture_state = "Pause"
                elif gesture == "Play":
                    gesture_state = "Play"
                elif gesture == "Next":
                    video_index = (video_index + 1) % len(video_list)
                    video.release()
                    return  # Switch to next video

        cv2.putText(frame_cam, f"Gesture: {gesture_state}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Gesture Camera", frame_cam)

        if cv2.waitKey(10) & 0xFF == 27:  # ESC
            video.release()
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Load video list
video_folder = 'D:\VideoGestureControl\Video'  # 🔥 chỉnh đúng folder chứa video của bạn
video_list = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]

# Initialize webcam
cap = cv2.VideoCapture(0)
gesture_state = "Pause"
video_index = 0

# Main loop
while True:
    print(f"Playing video: {video_list[video_index]}")
    play_video(video_list[video_index])
