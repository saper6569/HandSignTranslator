import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pickle
import os

# === Load static hand sign model ===
static_model_file = 'models/hand_sign_model.pkl'
static_model = None
STATIC_GESTURES = None

try:
    with open(static_model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    # Handle both old format (just model) and new format (dict with model and gestures)
    if isinstance(model_data, dict):
        static_model = model_data['model']
        STATIC_GESTURES = model_data['gestures']
        print(f"✓ Loaded static model: {static_model_file}")
        print(f"  Static gestures: {STATIC_GESTURES}")
    else:
        # Old format - try to load gesture mapping from data directory
        static_model = model_data
        gesture_mapping_file = "data/gesture_mapping.pkl"
        if os.path.exists(gesture_mapping_file):
            with open(gesture_mapping_file, 'rb') as f:
                STATIC_GESTURES = pickle.load(f)
            print(f"✓ Loaded static model: {static_model_file}")
            print(f"  Static gestures: {STATIC_GESTURES}")
        else:
            from config import GESTURES
            STATIC_GESTURES = GESTURES
            print(f"✓ Loaded static model: {static_model_file}")
            print(f"  WARNING: Using default static gestures from config: {STATIC_GESTURES}")
except FileNotFoundError:
    print("⚠ Static model file not found. Static detection will be disabled.")
except Exception as e:
    print(f"⚠ Error loading static model: {e}. Static detection will be disabled.")

# === Load sequence (movement) model ===
sequence_model = None
MOVEMENT_GESTURES = None
SEQUENCE_LENGTH = None
sequence_model_type = None
sequence_scaler = None  # For sklearn models that need scaling

# Try to load TensorFlow/Keras model
try:
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import keras
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
            
            model_file = 'models/sequence_model.h5'
            sequence_model = keras.models.load_model(model_file)
            print(f"✓ Loaded sequence model: {model_file}")
            print(f"  Movement gestures: {MOVEMENT_GESTURES}")
            print(f"  Sequence length: {SEQUENCE_LENGTH}")
    except Exception as e:
        print(f"⚠ Error loading Keras sequence model: {e}")

# Try to load sklearn fallback model if Keras model not available
if sequence_model is None:
    try:
        model_file = 'models/sequence_model.pkl'
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        sequence_model = model_data['model']
        MOVEMENT_GESTURES = model_data['gestures']
        SEQUENCE_LENGTH = model_data['sequence_length']
        sequence_model_type = model_data['model_type']
        
        # Load scaler if it exists (for RandomForest)
        if 'scaler' in model_data:
            sequence_scaler = model_data['scaler']
        else:
            sequence_scaler = None
        
        print(f"✓ Loaded sequence model: {model_file}")
        print(f"  Movement gestures: {MOVEMENT_GESTURES}")
        print(f"  Sequence length: {SEQUENCE_LENGTH}")
        print(f"  Model type: {sequence_model_type}")
    except FileNotFoundError:
        print("⚠ Sequence model file not found. Movement detection will be disabled.")
    except Exception as e:
        print(f"⚠ Error loading sequence model: {e}. Movement detection will be disabled.")

# Check if we have at least one model
if static_model is None and sequence_model is None:
    print("❌ Error: No models found. Please train at least one model.")
    exit()

# === Mediapipe setup ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# === Webcam setup ===
cap = cv2.VideoCapture(0)
text_buffer = deque(maxlen=30)  # For static gesture smoothing
output_text = ""

# === Sequence buffer for movement detection ===
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH if SEQUENCE_LENGTH else 30)
frame_counter = 0
FRAME_SKIP = 1  # Match config

def normalize_landmarks(landmarks):
    """Normalize so model sees consistent positions"""
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()

# Detection mode: 'auto', 'static', or 'movement'
detection_mode = 'auto'

print("\n" + "="*60)
print("Hand Sign Recognition System")
print("="*60)
if static_model:
    print(f"✓ Static detection enabled: {len(STATIC_GESTURES)} gestures")
if sequence_model:
    print(f"✓ Movement detection enabled: {len(MOVEMENT_GESTURES)} gestures")
print("\nControls:")
print("  'q' - Quit")
print("  ' ' (space) - Add current detection to text")
print("  'c' - Clear output text")
print("  'a' - Auto mode (detect both static and movement)")
print("  's' - Static mode only")
print("  'm' - Movement mode only")
print("="*60 + "\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_prediction = None
    prediction_type = None

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and normalize landmarks
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            landmarks = normalize_landmarks(landmarks)

            # === Static Detection ===
            if (detection_mode in ['auto', 'static']) and static_model is not None:
                try:
                    pred_idx = static_model.predict([landmarks])[0]
                    if hasattr(pred_idx, '__iter__') and len(pred_idx) > 0:
                        pred_idx = int(pred_idx[0])
                    else:
                        pred_idx = int(pred_idx)
                    
                    static_prediction = STATIC_GESTURES[pred_idx]
                    
                    if detection_mode == 'static':
                        current_prediction = static_prediction
                        prediction_type = "Static"
                    elif detection_mode == 'auto':
                        # In auto mode, prefer static detection
                        current_prediction = static_prediction
                        prediction_type = "Static"
                except Exception as e:
                    pass

            # === Movement Detection ===
            if (detection_mode in ['auto', 'movement']) and sequence_model is not None and SEQUENCE_LENGTH is not None:
                # Add to sequence buffer
                if frame_counter % FRAME_SKIP == 0:
                    sequence_buffer.append(landmarks)
                frame_counter += 1

                # When buffer is full, try to detect movement
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    try:
                        sequence = np.array(list(sequence_buffer))
                        sequence = sequence.reshape(1, SEQUENCE_LENGTH, -1)  # (1, seq_len, features)

                        if sequence_model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                            # Keras LSTM model
                            predictions = sequence_model.predict(sequence, verbose=0)
                            pred_idx = np.argmax(predictions[0])
                        else:
                            # sklearn model (RandomForest) - flatten and scale if needed
                            sequence_flat = sequence.reshape(1, -1)
                            if sequence_scaler is not None:
                                sequence_flat = sequence_scaler.transform(sequence_flat)
                            pred_idx = sequence_model.predict(sequence_flat)[0]
                        
                        movement_prediction = MOVEMENT_GESTURES[int(pred_idx)]
                        
                        # Only use movement detection if:
                        # 1. We're in movement mode, OR
                        # 2. We're in auto mode
                        # PRIORITIZE MOVEMENT in auto mode
                        if detection_mode == 'movement' or detection_mode == 'auto':
                            current_prediction = movement_prediction
                            prediction_type = "Movement"

                    except Exception as e:
                        pass

            # Display current prediction
            if current_prediction:
                text_buffer.append(current_prediction)
                
                # Smoothed prediction
                if len(text_buffer) >= 10:
                    most_common = max(set(text_buffer), key=text_buffer.count)
                    color = (0, 255, 0) if prediction_type == "Static" else (0, 255, 255)
                    cv2.putText(frame, f"{prediction_type}: {most_common}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    cv2.putText(frame, f"Mode: {detection_mode.upper()}", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Show sequence buffer progress for movement mode
                    if sequence_model and SEQUENCE_LENGTH and detection_mode in ['auto', 'movement']:
                        buffer_progress = len(sequence_buffer) / SEQUENCE_LENGTH
                        cv2.rectangle(frame, (10, h - 60), (210, h - 30), (100, 100, 100), 2)
                        cv2.rectangle(frame, (10, h - 60), (10 + int(200 * buffer_progress), h - 30), (0, 255, 255), -1)
                        cv2.putText(frame, f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", 
                                    (220, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"Mode: {detection_mode.upper()}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display recognized text
    cv2.putText(frame, f"Text: {output_text}", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Hand Sign to Text", frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):  # Space = append current word
        if len(text_buffer) > 0:
            most_common = max(set(text_buffer), key=text_buffer.count)
            output_text += most_common + " "
            text_buffer.clear()
            sequence_buffer.clear()  # Clear sequence buffer after adding text
    elif key == ord('c'):  # Clear output text
        output_text = ""
        text_buffer.clear()
        sequence_buffer.clear()
    elif key == ord('a'):  # Auto mode
        detection_mode = 'auto'
        print("Mode: AUTO (detecting both static and movement)")
    elif key == ord('s'):  # Static mode
        detection_mode = 'static'
        print("Mode: STATIC (static gestures only)")
        sequence_buffer.clear()
    elif key == ord('m'):  # Movement mode
        detection_mode = 'movement'
        print("Mode: MOVEMENT (movement gestures only)")
        text_buffer.clear()

cap.release()
cv2.destroyAllWindows()

