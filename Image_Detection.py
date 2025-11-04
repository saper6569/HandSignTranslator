import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from joblib import load 

# === Load trained model ===
model_file = 'hand_sign_model.pkl'

try:
    model = load(model_file)
    print(f"Loaded model: {model_file}")
except FileNotFoundError:
    print("Model file not found. Train it first with train_model.py.")
    exit()

# === Gesture mapping (update to match your GESTURES list) ===
GESTURES = ["B","L","C"]

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
text_buffer = deque(maxlen=30)
output_text = ""

def normalize_landmarks(landmarks):
    """Normalize so model sees consistent positions"""
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]
    landmarks -= wrist
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()

print("🚀 Starting Hand Sign Recognition...")
print("Press 'q' to quit, space to add to text, or 'c' to clear.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract and normalize landmarks
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            landmarks = normalize_landmarks(landmarks)

            # Predict gesture (model outputs index)
            pred_idx = model.predict([landmarks])[0]
            prediction = GESTURES[int(pred_idx)]
            text_buffer.append(prediction)

            # Smoothed prediction
            if len(text_buffer) == text_buffer.maxlen:
                most_common = max(set(text_buffer), key=text_buffer.count)
                cv2.putText(frame, f"Sign: {most_common}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

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
    elif key == ord('c'):  # Clear output text
        output_text = ""

cap.release()
cv2.destroyAllWindows()