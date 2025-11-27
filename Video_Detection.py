import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pickle
import os
import requests
import time

from config import LOW_LIGHT

# ================================
# CAMERA SOURCE SETUP
# ================================
use_webcam = False  # Start with ESP32 stream
ESP32_IP = "172.20.10.3"
ESP32_URL = f"http://{ESP32_IP}:81/stream"
adc_url = f"http://{ESP32_IP}/adc"

# Initialize ESP32 stream
stream = None
stream_iterator = None
bytes_buffer = b""
webcam = None

def init_esp32_stream():
    """Initialize ESP32 stream connection"""
    global stream, stream_iterator, bytes_buffer
    try:
        stream = requests.get(ESP32_URL, stream=True, timeout=5)
        # Use larger chunk size for better performance with ESP32
        stream_iterator = stream.iter_content(chunk_size=8192)
        bytes_buffer = b""
        return True
    except Exception as e:
        print(f"Error connecting to ESP32 stream: {e}")
        return False

def init_webcam():
    """Initialize webcam capture"""
    global webcam
    try:
        webcam = cv2.VideoCapture(0)
        if webcam.isOpened():
            webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            return True
        return False
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        return False

def cleanup_esp32_stream():
    """Clean up ESP32 stream connection"""
    global stream, stream_iterator, bytes_buffer
    if stream:
        try:
            stream.close()
        except:
            pass
    stream = None
    stream_iterator = None
    bytes_buffer = b""

def cleanup_webcam():
    """Clean up webcam capture"""
    global webcam
    if webcam:
        try:
            webcam.release()
        except:
            pass
    webcam = None

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

# Exit if no movement model available
if sequence_model is None:
    print("❌ No movement model available. Exiting.")
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
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH if SEQUENCE_LENGTH else 30)
frame_counter = 0
FRAME_SKIP = 1

# ADC checking variables
last_adc_check = 0
adc_check_interval = 2.0  # Check ADC every 2 seconds
low_light = False

# Performance optimization settings
PROCESS_WIDTH = 640   # Process MediaPipe on smaller frame for better performance
PROCESS_HEIGHT = 480
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 960
FRAME_SKIP_WEBCAM = 1  # Process every Nth frame for webcam (1 = all frames)
FRAME_SKIP_ESP32 = 3   # Process every Nth frame for ESP32 (2 = every other frame for better performance)

# Cache last detection result to prevent flickering when frames are skipped
cached_result = None
cached_hand_landmarks = None
cached_prediction_text = None  # Cache the displayed prediction text
no_hand_frame_count = 0  # Track consecutive frames with no hand


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


print("\n=== Hand Sign Recognition System (Movement Only) ===")
print("Controls: q=quit, w=switch camera source\n")

# Initialize starting camera source
if not use_webcam:
    if not init_esp32_stream():
        print("Failed to connect to ESP32 stream. Switching to webcam...")
        use_webcam = True
        if not init_webcam():
            print("Error: Could not initialize webcam. Exiting...")
            exit()
else:
    if not init_webcam():
        print("Error: Could not initialize webcam. Exiting...")
        exit()

# Create named window and resize it
window_name = "Hand Sign Recognition (Movement)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

# ============================================
# MAIN LOOP
# ============================================
running = True

while running:
    # Handle camera source switching
    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        # Switch camera source
        if use_webcam:
            cleanup_webcam()
            use_webcam = False
            frame_counter = 0  # Reset frame counter when switching
            cached_result = None  # Clear cache when switching cameras
            cached_hand_landmarks = None
            cached_prediction_text = None  # Clear cached prediction when switching
            no_hand_frame_count = 0  # Reset no hand frame count
            sequence_buffer.clear()  # Clear sequence buffer when switching
            if init_esp32_stream():
                print("Switched to ESP32 stream")
            else:
                print("Failed to connect to ESP32 stream. Staying on webcam...")
                init_webcam()
                use_webcam = True
        else:
            cleanup_esp32_stream()
            use_webcam = True
            frame_counter = 0  # Reset frame counter when switching
            cached_result = None  # Clear cache when switching cameras
            cached_hand_landmarks = None
            cached_prediction_text = None  # Clear cached prediction when switching
            no_hand_frame_count = 0  # Reset no hand frame count
            sequence_buffer.clear()  # Clear sequence buffer when switching
            if init_webcam():
                print("Switched to webcam")
            else:
                print("Failed to initialize webcam. Staying on ESP32 stream...")
                init_esp32_stream()
                use_webcam = False
        continue
    elif key == ord('q'):
        break

    frame = None

    # Get frame from current source
    if use_webcam:
        # Read from webcam
        if webcam and webcam.isOpened():
            ret, frame = webcam.read()
            if not ret:
                continue
        else:
            continue
    else:
        # Read from ESP32 stream
        if not stream or stream_iterator is None:
            continue
        
        try:
            # Read multiple chunks at once for better performance
            # Read until we have a complete JPEG frame
            max_chunks = 20  # Limit to prevent infinite loop
            chunks_read = 0
            while chunks_read < max_chunks:
                chunk = next(stream_iterator, None)
                if chunk is None or len(chunk) == 0:
                    # Stream ended, try to reconnect
                    raise StopIteration
                    
                bytes_buffer += chunk
                chunks_read += 1

                # Find JPEG start/end markers (skip header if present)
                jpg_start = bytes_buffer.find(b'\xff\xd8')
                if jpg_start == -1:
                    continue
                    
                jpg_end = bytes_buffer.find(b'\xff\xd9', jpg_start)
                if jpg_end == -1:
                    continue

                # Extract JPEG frame
                jpg = bytes_buffer[jpg_start:jpg_end + 2]
                bytes_buffer = bytes_buffer[jpg_end + 2:]
                frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    break
            
        except (StopIteration, requests.RequestException, Exception) as e:
            # Stream ended or error, try to reconnect
            print(f"ESP32 stream disconnected: {e}. Attempting to reconnect...")
            cleanup_esp32_stream()
            time.sleep(1)
            if init_esp32_stream():
                print("Reconnected to ESP32 stream")
            else:
                print("Reconnection failed. Switching to webcam...")
                use_webcam = True
                if not init_webcam():
                    print("Failed to initialize webcam. Exiting...")
                    running = False
            continue

    if frame is None:
        continue

    # =============================
    # PROCESS FRAME
    # =============================
    # Only flip if using ESP32 (not webcam)
    if not use_webcam:
        frame = cv2.flip(frame, 1)
        # Rotate frame 90 degrees clockwise (camera is lying on its side)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h, w, _ = frame.shape
    
    # Increment frame counter for frame skipping
    frame_counter += 1
    
    # Resize frame for processing (smaller = faster MediaPipe processing)
    # This is especially important for ESP32 streams which may be higher resolution
    frame_process = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))

    # === CHECK LIGHT LEVEL (only for ESP32, every 2 seconds) ===
    if not use_webcam:
        current_time = time.time()
        if current_time - last_adc_check >= adc_check_interval:
            adc_value = get_adc_value()
            if adc_value is not None:
                print(f"ADC Value: {adc_value}")
                low_light = adc_value < LOW_LIGHT
            last_adc_check = current_time
    else:
        # Reset low_light when using webcam (no ADC available)
        low_light = False

    # Resize frame for display
    frame_display = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
    display_h, display_w, _ = frame_display.shape

    # Display current camera source
    source_text = "Webcam" if use_webcam else "ESP32 Stream"
    cv2.putText(frame_display, f"Source: {source_text}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show low light warning if light is insufficient (ESP32 only)
    if low_light and not use_webcam:
        cv2.putText(frame_display, "Low confidence due to low light", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    # Process hand detection only on frames we want to process (frame skipping)
    # Use different skip rates for ESP32 vs webcam (ESP32 needs more skipping for performance)
    current_frame_skip = FRAME_SKIP_ESP32 if not use_webcam else FRAME_SKIP_WEBCAM
    
    result = None
    hand_landmarks_to_draw = None
    
    if frame_counter % current_frame_skip == 0:
        # Process on smaller frame for better performance
        # MediaPipe landmarks are normalized (0-1), so they'll work correctly on any frame size
        rgb = cv2.cvtColor(frame_process, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        
        # Update cache with latest result
        if result.multi_hand_landmarks:
            cached_result = result
            cached_hand_landmarks = result.multi_hand_landmarks[0]  # Cache first hand
            no_hand_frame_count = 0  # Reset counter when hand is detected
        else:
            # Clear cache if no hand detected
            cached_result = result
            cached_hand_landmarks = None
            no_hand_frame_count += 1
            # Clear cached prediction after multiple consecutive "no hand" detections
            if no_hand_frame_count >= 5:  # Clear after 5 processed frames with no hand
                cached_prediction_text = None
    else:
        # Use cached result when skipping frames to prevent flickering
        if cached_result is not None:
            result = cached_result
            hand_landmarks_to_draw = cached_hand_landmarks

    # Determine which landmarks to draw
    if result is not None and result.multi_hand_landmarks:
        hand_landmarks_to_draw = result.multi_hand_landmarks[0]
    elif hand_landmarks_to_draw is None and cached_hand_landmarks is not None:
        hand_landmarks_to_draw = cached_hand_landmarks

    current_prediction = None

    if hand_landmarks_to_draw is not None:
        # Hand landmarks drawing removed - only processing for prediction
        
        # Only process landmarks on frames we're processing (not skipped ones)
        if frame_counter % current_frame_skip == 0 and result is not None and result.multi_hand_landmarks:
            lm = [[p.x, p.y, p.z] for p in hand_landmarks_to_draw.landmark]
            lm_norm = normalize_landmarks(lm)

            # ------------------------------
            # MOVEMENT MODEL PREDICTION
            # ------------------------------
            # Append to sequence buffer (only on processed frames)
            sequence_buffer.append(lm_norm)

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

                    current_prediction = MOVEMENT_GESTURES[idx]
                    # Update cached prediction when we get a new one
                    if current_prediction:
                        cached_prediction_text = current_prediction
                except:
                    pass

    # =============================
    # DISPLAY
    # =============================
    # Use cached prediction to prevent flickering on skipped frames
    display_prediction = current_prediction if current_prediction else cached_prediction_text
    
    if display_prediction and not low_light:
        text_y = 140 if low_light and not use_webcam else 100
        cv2.putText(frame_display, f" {display_prediction}", (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
    elif not display_prediction and cached_hand_landmarks is None and not low_light:
        # Only show "No hand detected" if we don't have a cached result or prediction
        text_y = 140 if use_webcam else 100
        cv2.putText(frame_display, "No hand detected", (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)

    cv2.imshow(window_name, frame_display)

# Cleanup
cleanup_esp32_stream()
cleanup_webcam()
cv2.destroyAllWindows()