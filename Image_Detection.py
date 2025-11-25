import cv2
import requests
import numpy as np
import mediapipe as mp
from collections import deque
import pickle
import os
import time

# === ESP32 CAM SETUP ===
ESP32_IP = "172.20.10.3"
stream_url = f"http://{ESP32_IP}:81/stream"
adc_url = f"http://{ESP32_IP}/adc"

stream = requests.get(stream_url, stream=True)
bytes_buffer = b""

# === Load trained model ===
model_file = 'models/hand_sign_model.pkl'

try:
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)

    if isinstance(model_data, dict):
        model = model_data['model']
        GESTURES = model_data['gestures']
        print(f"Loaded model: {model_file}")
        print(f"Loaded gesture mapping: {GESTURES}")
    else:
        model = model_data
        gesture_mapping_file = "data/gesture_mapping.pkl"
        if os.path.exists(gesture_mapping_file):
            with open(gesture_mapping_file, 'rb') as f:
                GESTURES = pickle.load(f)
            print(f"Loaded model: {model_file}")
            print(f"Loaded gesture mapping from data directory: {GESTURES}")
        else:
            from config import GESTURES

            print(f"Loaded model: {model_file}")
            print(f"WARNING: No gesture mapping found. Using default from config: {GESTURES}")

except Exception as e:
    print("Error loading model:", e)
    exit()

# === Mediapipe setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

text_buffer = deque(maxlen=30)

# ADC checking variables
last_adc_check = 0
adc_check_interval = 2.0  # Check ADC every 2 seconds
low_light = False

# Window size
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 960


def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()


def get_adc_value():
    """Fetch ADC value from ESP32"""
    try:
        response = requests.get(adc_url, timeout=0.3)
        data = response.json()
        return data['adc_raw']
    except:
        return None


print("Starting ESP32 Hand Sign Recognition...")
print("Press 'q' to quit.\n")

# Create named window and resize it
cv2.namedWindow("ESP32 Hand Sign Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ESP32 Hand Sign Recognition", DISPLAY_WIDTH, DISPLAY_HEIGHT)

# === MAIN LOOP: READ ESP32 STREAM → PROCESS FRAME ===
for chunk in stream.iter_content(chunk_size=4096):

    if not chunk:
        continue

    bytes_buffer += chunk

    # Find header ending
    header_end = bytes_buffer.find(b'\r\n\r\n')
    if header_end == -1:
        continue

    # JPEG start/end markers
    jpg_start = bytes_buffer.find(b'\xff\xd8', header_end)
    jpg_end = bytes_buffer.find(b'\xff\xd9', jpg_start)

    if jpg_start != -1 and jpg_end != -1:
        jpg = bytes_buffer[jpg_start:jpg_end + 2]
        bytes_buffer = bytes_buffer[jpg_end + 2:]

        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # === CHECK LIGHT LEVEL (only every 2 seconds) ===
        current_time = time.time()
        if current_time - last_adc_check >= adc_check_interval:
            adc_value = get_adc_value()
            if adc_value is not None:
                print(f"ADC Value: {adc_value}")
                low_light = adc_value < 2500
            last_adc_check = current_time

        # === PROCESS FRAME WITH MEDIAPIPE & MODEL ===
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Resize frame for display
        frame_display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
        display_h, display_w, _ = frame_display.shape

        # Show low light warning if light is insufficient
        if low_light:
            cv2.putText(frame_display, "Low confidence due to low light", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

        # Always process hand detection regardless of light level
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the resized display frame
                mp_drawing.draw_landmarks(
                    frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                landmarks = normalize_landmarks(landmarks)

                pred_idx = model.predict([landmarks])[0]
                prediction = GESTURES[int(pred_idx)]
                text_buffer.append(prediction)

                if len(text_buffer) == text_buffer.maxlen and not low_light:
                    most_common = max(set(text_buffer), key=text_buffer.count)
                    cv2.putText(frame_display, f"Sign: {most_common}", (20, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 4)
        else:
            if not low_light:
                cv2.putText(frame_display, "No hand detected", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)

        cv2.imshow("ESP32 Hand Sign Recognition", frame_display)

        # ---- Keyboard controls ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

cv2.destroyAllWindows()