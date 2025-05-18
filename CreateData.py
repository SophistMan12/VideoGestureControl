import cv2
import numpy as np # type: ignore
import pandas as pd # type: ignore
import mediapipe as mp
import os  # Thêm thư viện os để kiểm tra file tồn tại

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Đây là nơi lưu dữ liệu
collected_data = []
labels = []  # nhãn tương ứng

# Chỉ lấy 42 cột (21 điểm x 2 tọa độ x và y)
def extract_keypoints(hand_landmarks):
    keypoints = []
    for lm in hand_landmarks.landmark[:21]:  # Chỉ lấy 21 điểm đầu tiên
        keypoints.append(lm.x)
        keypoints.append(lm.y)
    return keypoints

gesture_label = input("Nhập tên cử chỉ hiện tại (Next, Pause, Play, Start): ")

print("Nhấn SPACE để lưu keypoints, ESC để thoát")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Keypoint Collection", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC để thoát
        break
    elif key == 32 and result.multi_hand_landmarks:  # SPACE để lưu dữ liệu
        # Chỉ lấy landmark bàn tay đầu tiên
        keypoints = extract_keypoints(result.multi_hand_landmarks[0])
        collected_data.append(keypoints)
        labels.append(gesture_label)
        print(f"Đã có {len(collected_data)} mẫu cho {gesture_label}")
cap.release()
cv2.destroyAllWindows()

# Đường dẫn file CSV
csv_file = 'hand_gesture_dataset.csv'

# Nếu file đã tồn tại, đọc dữ liệu cũ
if os.path.exists(csv_file):
    existing_data = pd.read_csv(csv_file)
    df = pd.DataFrame(collected_data, columns=[f'keypoint_{i}' for i in range(1, 43)])
    df['label'] = labels
    # Loại bỏ các mục trống hoặc toàn bộ giá trị NA trước khi nối
    existing_data = existing_data.dropna(how='all')
    df = pd.concat([existing_data, df], ignore_index=True)  # Nối dữ liệu mới vào dữ liệu cũ
else:
    # Nếu file chưa tồn tại, chỉ tạo DataFrame từ dữ liệu mới
    df = pd.DataFrame(collected_data, columns=[f'keypoint_{i}' for i in range(1, 43)])
    df['label'] = labels

# Ghi dữ liệu vào file CSV
df.to_csv(csv_file, index=False)

print("🎯 Đã lưu xong dataset hand_gesture_dataset.csv!")

import pandas as pd  # type: ignore

# Đọc file CSV đã upload
df = pd.read_csv('D:\VideoGestureControl\hand_gesture_dataset.csv')

# Xem tổng số dòng (số mẫu)
print("Tổng số mẫu:", len(df))

# Xem thống kê từng label (từng loại cử chỉ)
print("\nSố mẫu theo từng cử chỉ:")
print(df['label'].value_counts())