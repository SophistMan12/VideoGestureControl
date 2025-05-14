import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
import os
import time
from keras.models import load_model  # type: ignore

# Load mô hình đã huấn luyện
model = load_model(r'D:\VideoGestureControl\gesture_model_v7.h5')

# Nhãn tương ứng với output của mô hình
gesture_labels = [ 'Next', 'Pause', 'Play', 'Start']

# Khởi tạo MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Chuyển landmarks thành vector 42 chiều (x, y)
def landmarks_to_vector(hand_landmarks):
    vector = []
    for lm in hand_landmarks.landmark[:21]:  # Chỉ lấy 21 điểm đầu tiên
        vector.extend([lm.x, lm.y])
    return np.array(vector, dtype=np.float32)

# Dự đoán cử chỉ từ landmark
# Thêm debug để kiểm tra dự đoán

def predict_gesture(hand_landmarks):
    if hand_landmarks is None:
        return None  # Trả về None nếu không nhận diện được

    input_vector = landmarks_to_vector(hand_landmarks).reshape(1, 42)  # Đảm bảo reshape thành (1, 42)
    predictions = model.predict(input_vector, verbose=0)
    #print(f"🔍 Raw predictions: {predictions}")  # Debug: In ra dự đoán thô

    predicted_index = np.argmax(predictions)
    gesture = gesture_labels[predicted_index]
    print(f"✅ Gesture: {gesture} (Index: {predicted_index})")

    with open("gesture_log.txt", "a") as log:
        log.write(f"{time.ctime()}: {gesture}\n")
    return gesture

# Phát video với điều khiển bằng cử chỉ
def play_video(path):
    global gesture_state, video_index
    video = cv2.VideoCapture(path)

    while True:
        if gesture_state == "Pause":
            cv2.waitKey(100)
        else:
            ret, frame = video.read()
            if not ret:
                print("🔁 Replaying video")
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            cv2.imshow("Video Player", frame)

        ret_cam, frame_cam = cap.read()
        if not ret_cam:
            print("❌ Cannot read webcam.")
            break

        frame_cam = cv2.flip(frame_cam, 1)
        rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame_cam, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = predict_gesture(hand_landmarks)
                if gesture == "Start":
                    gesture_state = "Play"
                elif gesture == "Play":
                    gesture_state = "Play"
                elif gesture == "Pause":
                    gesture_state = "Pause"
                elif gesture == "Next":
                    video_index = (video_index + 1) % len(video_list)
                    video.release()
                    return  # chuyển video mới

        cv2.putText(frame_cam, f"Gesture: {gesture_state}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Gesture Camera", frame_cam)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            cap.release()
            video.release()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('n'):
            video_index = (video_index + 1) % len(video_list)
            video.release()
            return  # chuyển video mới

# Tải danh sách video
video_folder = r'D:\VideoGestureControl\Video'
video_list = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
gesture_state = "Pause"
video_index = 0

# Vòng lặp chính
while True:
    print(f"▶ Playing video: {video_list[video_index]}")
    play_video(video_list[video_index])