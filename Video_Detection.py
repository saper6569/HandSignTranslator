import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pickle
import os
import requests

# ================================
# ESP32 CAM STREAM SETUP
# ================================
ESP32_URL = "http://172.20.10.3:81/stream"
stream = requests.get(ESP32_URL, stream=True)
bytes_buffer = b""

# ============================================
# STATIC MODEL LOADING
# ============================================
static_model_file = 'models/hand_sign_model.pkl'
static_model = None
STATIC_GESTURES = None

try:
    with open(static_model_file, 'rb') as f:
        model_data = pickle.load(f)

    if isinstance(model_data, dict):
        static_model = model_data['model']
        STATIC_GESTURES = model_data['gestures']
        print(f"✓ Loaded static model: {static_model_file}")
    else:
        static_model = model_data
        gesture_mapping_file = "data/gesture_mapping.pkl"
        if os.path.exists(gesture_mapping_file):
            with open(gesture_mapping_file, 'rb') as f:
                STATIC_GESTURES = pickle.load(f)
            print(f"✓ Loaded static model + gesture mapping (legacy)")
        else:
            from config import GESTURES

            STATIC_GESTURES = GESTURES
            print("⚠ Using default gesture mapping (no mapping file found).")
except Exception as e:
    print(f"⚠ Static model load failed: {e}")

# ============================================
# MOVEMENT (SEQUENCE) MODEL LOADING
# ============================================
sequence_model = None
MOVEMENT_GESTURES = None
SEQUENCE_LENGTH = None
sequence_model_type = None
sequence_scaler = None

# Try TensorFlow/Keras first
try:
    from tensorflow import keras

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

if TENSORFLOW_AVAILABLE:
    try:
        metadata_file = 'models/sequence_model_metadata.pkl'
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)

            MOVEMENT_GESTURES = metadata['gestures']
            SEQUENCE_LENGTH = metadata['sequence_length']
            sequence_model_type = metadata['model_type']

            sequence_model = keras.models.load_model('models/sequence_model.h5')
            print(f"✓ Loaded Keras sequence model (LSTM)")
    except Exception as e:
        print(f"⚠ Keras model failed: {e}")

# Fall back to sklearn sequence model
if sequence_model is None:
    try:
        with open('models/sequence_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        sequence_model = model_data['model']
        MOVEMENT_GESTURES = model_data['gestures']
        SEQUENCE_LENGTH = model_data['sequence_length']
        sequence_model_type = model_data['model_type']
        sequence_scaler = model_data.get('scaler', None)
        print(f"✓ Loaded sklearn sequence model")
    except:
        print("⚠ No sequence model found.")

# If no models at all, exit
if static_model is None and sequence_model is None:
    print("❌ No models available. Exiting.")
    exit()

# ============================================
# Mediapipe Setup
# ============================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ============================================
# Buffers & Settings
# ============================================
text_buffer = deque(maxlen=30)
output_text = ""

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH if SEQUENCE_LENGTH else 30)
frame_counter = 0
FRAME_SKIP = 1

detection_mode = "auto"


def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()


print("\n=== Hand Sign Recognition System (ESP32 Stream) ===")
print("Modes: a=auto, s=static, m=movement, space=add word, c=clear, q=quit\n")

# ============================================
# MAIN LOOP — READ FROM ESP32 STREAM
# ============================================
for chunk in stream.iter_content(chunk_size=4096):

    if not chunk:
        continue

    bytes_buffer += chunk

    # Find JPEG markers
    jpg_start = bytes_buffer.find(b'\xff\xd8')
    jpg_end = bytes_buffer.find(b'\xff\xd9', jpg_start)

    if jpg_start == -1 or jpg_end == -1:
        continue

    jpg = bytes_buffer[jpg_start:jpg_end + 2]
    bytes_buffer = bytes_buffer[jpg_end + 2:]

    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    # =============================
    # PROCESS FRAME
    # =============================
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_prediction = None
    prediction_type = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = [[p.x, p.y, p.z] for p in hand_landmarks.landmark]
            lm_norm = normalize_landmarks(lm)

            # ------------------------------
            # STATIC MODEL PREDICTION
            # ------------------------------
            if (detection_mode in ["auto", "static"]) and static_model is not None:
                try:
                    idx = static_model.predict([lm_norm])[0]
                    idx = int(idx[0]) if hasattr(idx, '__len__') else int(idx)
                    static_pred = STATIC_GESTURES[idx]
                    current_prediction = static_pred
                    prediction_type = "Static"
                except:
                    pass

            # ------------------------------
            # MOVEMENT MODEL PREDICTION
            # ------------------------------
            if (detection_mode in ["auto", "movement"]) and sequence_model:
                if frame_counter % FRAME_SKIP == 0:
                    sequence_buffer.append(lm_norm)
                frame_counter += 1

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    seq = np.array(sequence_buffer).reshape(1, SEQUENCE_LENGTH, -1)

                    try:
                        if sequence_model_type == "lstm" and TENSORFLOW_AVAILABLE:
                            probs = sequence_model.predict(seq, verbose=0)[0]
                            idx = int(np.argmax(probs))
                        else:
                            flat = seq.reshape(1, -1)
                            if sequence_scaler:
                                flat = sequence_scaler.transform(flat)
                            idx = int(sequence_model.predict(flat)[0])

                        movement_pred = MOVEMENT_GESTURES[idx]
                        current_prediction = movement_pred
                        prediction_type = "Movement"
                    except:
                        pass

    # =============================
    # SMOOTHING & DISPLAY
    # =============================
    if current_prediction:
        text_buffer.append(current_prediction)

        if len(text_buffer) >= 10:
            final_pred = max(set(text_buffer), key=text_buffer.count)
            color = (0, 255, 0) if prediction_type == "Static" else (0, 255, 255)

            cv2.putText(frame, f"{prediction_type}: {final_pred}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.putText(frame, f"Mode: {detection_mode.upper()}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(frame, f"Text: {output_text}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("ESP32 Hand Sign Recognition", frame)

    # =============================
    # KEYBOARD CONTROLS
    # =============================
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord(' '):
        if len(text_buffer):
            output_text += max(set(text_buffer), key=text_buffer.count) + " "
            text_buffer.clear()
            sequence_buffer.clear()
    elif key == ord('c'):
        output_text = ""
        text_buffer.clear()
        sequence_buffer.clear()
    elif key == ord('a'):
        detection_mode = "auto"
        print("Mode → AUTO")
    elif key == ord('s'):
        detection_mode = "static"
        sequence_buffer.clear()
        print("Mode → STATIC")
    elif key == ord('m'):
        detection_mode = "movement"
        text_buffer.clear()
        print("Mode → MOVEMENT")

cv2.destroyAllWindows()
