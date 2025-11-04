import cv2
import mediapipe as mp
import numpy as np
import os

# === CONFIG ===
GESTURES = ["B","L","C"]
SAMPLES_PER_GESTURE = 300

# === Mediapipe setup ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]
    landmarks -= wrist  # translate so wrist is at origin
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()

def collect_data_incremental():
    # Try loading existing data
    if os.path.exists("data/landmarks.npy"):
        X_old = np.load("data/landmarks.npy")
        y_old = np.load("data/labels.npy")
        print(f"Loaded existing dataset: {X_old.shape[0]} samples")
    else:
        X_old = np.empty((0, 63))  # 21 landmarks × 3 coords
        y_old = np.empty((0,))
        print("No existing dataset found, starting fresh")

    X_new = []
    y_new = []

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        for label_idx, gesture_name in enumerate(GESTURES):
            print(f"\nCollecting data for gesture: {gesture_name}")
            print("Press 's' to start recording...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    break

            count = 0
            while count < SAMPLES_PER_GESTURE:
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Ensure rgb is a contiguous numpy array with uint8 dtype
                rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        landmarks = normalize_landmarks(landmarks)

                        X_new.append(landmarks)
                        y_new.append(label_idx)
                        count += 1

                cv2.putText(frame, f"Samples: {count}/{SAMPLES_PER_GESTURE}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Data Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    cv2.destroyAllWindows()

    # Combine old and new data
    X_new = np.array(X_new)
    y_new = np.array(y_new)

    X_combined = np.concatenate((X_old, X_new))
    y_combined = np.concatenate((y_old, y_new))

    np.save("data/landmarks.npy", X_combined)
    np.save("data/labels.npy", y_combined)

    print(f"\nAdded {X_new.shape[0]} new samples.")
    print(f"Total dataset size: {X_combined.shape[0]} samples.")

if __name__ == "__main__":
    collect_data_incremental()
