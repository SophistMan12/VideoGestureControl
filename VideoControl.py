import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
import os
from keras.models import load_model  # type: ignore

# Load model từ file
model = load_model(r'D:\VideoGestureControl\Model.h5')  # Đảm bảo path đúng

# Các nhãn tương ứng với output của model
gesture_labels = ["Start", "Pause", "Play"]

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ✅ Hàm chuyển landmarks thành vector 420 phần tử
def landmarks_to_vector(hand_landmarks):
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])  # 21 x 3 = 63

    # Tăng số chiều lên 672 bằng cách nhân lên
    landmarks = landmarks * 10  # 63 x 10 = 630
    while len(landmarks) < 672:
        landmarks.append(0.0)

    return np.array(landmarks[:672], dtype=np.float32)


# Biến đếm toàn cục
gesture_count = 0

def predict_gesture(hand_landmarks):
    global gesture_count
    gesture_count += 1  # Tăng biến đếm mỗi lần nhận diện
    input_vector = landmarks_to_vector(hand_landmarks).reshape(1, 672)
    predictions = model.predict(input_vector, verbose=0)
    predicted_index = np.argmax(predictions)
    gesture = gesture_labels[predicted_index]
    
    # Thông báo ra terminal với thứ tự
    print(f"✅ Gesture #{gesture_count}: {gesture}")
    
    return gesture

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
                    gesture_state = "Start"
                elif gesture == "Pause":
                    gesture_state = "Pause"
                elif gesture == "Play":
                    gesture_state = "Play"

        cv2.putText(frame_cam, f"Gesture: {gesture_state}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Gesture Camera", frame_cam)

        if cv2.waitKey(10) & 0xFF == 27:
            video.release()
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Load video list
video_folder = r'D:\VideoGestureControl\Video'
video_list = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]

# Initialize webcam
cap = cv2.VideoCapture(0)
gesture_state = "Pause"
video_index = 0

# Play loop
while True:
    print(f"▶ Playing video: {video_list[video_index]}")
    play_video(video_list[video_index])
