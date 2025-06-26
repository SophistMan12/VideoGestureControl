import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
import os
import time
from keras.models import load_model  # type: ignore
import ctypes
import vlc   # type: ignore

# Add VLC installation directory to PATH to ensure libvlc.dll and dependencies are found
vlc_installation_path = r"C:\\Program Files\\VideoLAN\\VLC"
os.environ["PATH"] += os.pathsep + vlc_installation_path

# Ensure libvlc.dll is loaded correctly
try:
    ctypes.CDLL(os.path.join(vlc_installation_path, "libvlc.dll"))
except OSError as e:
    print(f"❌ Failed to load libvlc.dll: {e}")
    exit(1)

# Load mô hình đã huấn luyện
model = load_model(r'D:\VideoGestureControl\gesture_model_v8.h5')

# Nhãn tương ứng với output của mô hình
gesture_labels = ['Next', 'Pause', 'Play', 'Start']
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

# Thêm biến lưu trữ gesture trước đó
previous_gesture = None

# Phát video với điều khiển bằng cử chỉ
# Sử dụng VLC để phát video có âm thanh
def play_video(path):
    global gesture_state, video_index, previous_gesture

    # Tạo đối tượng VLC
    player = vlc.MediaPlayer(path)
    player.play()
    player.video_set_scale(1.5)  # Tăng kích thước video lên lần
    
    while True:
        if gesture_state == "Pause":
            player.set_pause(1)  # Tạm dừng video và âm thanh
        else:
            player.set_pause(0)  # Tiếp tục phát video

        # Xử lý camera ngay cả khi video đang tạm dừng
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

                # Chỉ xử lý khi gesture khác với gesture trước đó
                if gesture != previous_gesture:
                    previous_gesture = gesture
                    if gesture == "Start":
                        gesture_state = "Play"
                    elif gesture == "Play":
                        gesture_state = "Play"
                    elif gesture == "Pause":
                        gesture_state = "Pause"
                    elif gesture == "Next":
                        video_index = (video_index + 1) % len(video_list)
                        player.stop()
                        return  # chuyển video mới
                        
        # Giảm kích thước khung hình camera trước khi hiển thị
        frame_cam = cv2.resize(frame_cam, (320, 240))
        cv2.putText(frame_cam, f"Gesture: {gesture_state}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Gesture Camera", frame_cam)
        # Đưa cửa sổ webcam xuống góc phải dưới màn hình
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
        cam_width, cam_height = 480, 360
        cv2.moveWindow("Gesture Camera", screen_width - cam_width, screen_height - cam_height)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:  # ESC
            cap.release()
            player.stop()
            cv2.destroyAllWindows()
            exit()
        elif key == ord('n'):
            video_index = (video_index + 1) % len(video_list)
            player.stop()
            return  # chuyển video mới

# Tải danh sách video
video_folder = r'D:\VideoGestureControl\Video'
video_list = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

# Thử mở camera với các ID khác nhau nếu ID 0 không hoạt động
camera_id = 0
while not cap.isOpened() and camera_id < 5:
    print(f"⚠️ Camera ID {camera_id} không hoạt động. Thử ID tiếp theo...")
    camera_id += 1
    cap = cv2.VideoCapture(camera_id)

if not cap.isOpened():
    print("❌ Không thể mở bất kỳ camera nào. Vui lòng kiểm tra kết nối hoặc quyền truy cập.")
    exit(1)

print(f"✅ Camera đã được mở thành công với ID {camera_id}.")

# Xóa logic kiểm tra khung hình từ camera

gesture_state = "Pause"
video_index = 0

# Vòng lặp chính
while True:
    print(f"▶ Playing video: {video_list[video_index]}\n")
    play_video(video_list[video_index])