import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
import os

# Initialize Mediapipe Hands and Drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
def detect_gesture(hand_landmarks):
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    # OK gesture (Start video)
    if (thumb_tip.x < index_tip.x and
        abs(thumb_tip.y - index_tip.y) < 0.05 and
        middle_tip.y < wrist.y and
        ring_tip.y < wrist.y and
        pinky_tip.y < wrist.y):
        return "Start"
    # Open hand (Pause)
    elif (index_tip.y < wrist.y and
          middle_tip.y < wrist.y and
          ring_tip.y < wrist.y and
          pinky_tip.y < wrist.y and
          thumb_tip.y < wrist.y and
          middle_tip.y < index_tip.y and
          middle_tip.y < ring_tip.y and
          middle_tip.y < pinky_tip.y):
        return "Pause"
    # Fist (Play after pause)
    elif (thumb_tip.y < index_tip.y and
          thumb_tip.y < middle_tip.y and
          thumb_tip.y < ring_tip.y and
          thumb_tip.y < pinky_tip.y):
        return "Play"
    # Thumb and index finger apart (Next video)
    elif abs(thumb_tip.x - index_tip.x) > 0.1 and abs(thumb_tip.y - index_tip.y) < 0.05:
        return "Next"
    else:
        return "Unknown"
def play_video(path):
    global gesture_state, video_index
    video = cv2.VideoCapture(path)

    while video.isOpened():
        if gesture_state == "Pause":
            cv2.waitKey(100)  # Wait 100ms when paused
        else:
            ret, frame = video.read()
            if not ret:
                break
            cv2.imshow("Video Player", frame)

        # Process webcam input
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
                gesture = detect_gesture(hand_landmarks)

                if gesture == "Start":
                    gesture_state = "Play"
                elif gesture == "Pause":
                    gesture_state = "Pause"
                elif gesture == "Play":
                    gesture_state = "Play"
                elif gesture == "Next":
                    video_index = (video_index + 1) % len(video_list)
                    video.release()
                    return  # Exit to switch to the next video

        cv2.putText(frame_cam, f"Gesture: {gesture_state}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv2.imshow("Gesture Camera", frame_cam)

        if cv2.waitKey(10) & 0xFF == 27:  # ESC to exit
            video.release()
            cap.release()
            cv2.destroyAllWindows()
            exit()

# Load video list from a folder
video_folder = '.\\Video'  # Change this to your video folder path
video_list = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.mp4')]

# Initialize webcam
cap = cv2.VideoCapture(0)
gesture_state = "Pause"
video_index = 0

# Main loop to play videos
while True:
    print(f"Playing video: {video_list[video_index]}")
    play_video(video_list[video_index])