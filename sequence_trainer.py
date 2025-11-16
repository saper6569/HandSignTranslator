import cv2
import mediapipe as mp
import numpy as np
import os
import pickle
from config import MOVEMENT_GESTURES, SEQUENCES_PER_GESTURE, SEQUENCE_LENGTH, FRAME_SKIP

# === Mediapipe setup ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def normalize_landmarks(landmarks):
    """Normalize landmarks to be position-invariant"""
    landmarks = np.array(landmarks).reshape(-1, 3)
    wrist = landmarks[0]
    landmarks -= wrist  # translate so wrist is at origin
    max_val = np.max(np.abs(landmarks))
    if max_val > 0:
        landmarks /= max_val
    return landmarks.flatten()

def collect_sequence_data():
    """Collect sequence data for movement-based hand signs"""
    # Load existing gesture mapping if it exists
    movement_gesture_mapping_file = "data/movement_gesture_mapping.pkl"
    if os.path.exists(movement_gesture_mapping_file):
        with open(movement_gesture_mapping_file, 'rb') as f:
            existing_gestures = pickle.load(f)
        print(f"Loaded existing movement gesture mapping: {existing_gestures}")
    else:
        existing_gestures = []
        print("No existing movement gesture mapping found, starting fresh")
    
    # Try loading existing sequence data
    if os.path.exists("data/sequences.npy"):
        X_old = np.load("data/sequences.npy")
        y_old = np.load("data/sequence_labels.npy")
        print(f"Loaded existing sequence dataset: {X_old.shape[0]} sequences")
    else:
        X_old = np.empty((0, SEQUENCE_LENGTH, 63))  # (sequences, frames, features)
        y_old = np.empty((0,))
        print("No existing sequence dataset found, starting fresh")

    # Find which gestures are new (not in existing data)
    new_gestures = [g for g in MOVEMENT_GESTURES if g not in existing_gestures]
    
    if not new_gestures:
        print(f"\nAll gestures in MOVEMENT_GESTURES ({MOVEMENT_GESTURES}) already have data.")
        print("To collect data for new movement gestures, add them to MOVEMENT_GESTURES in config.py")
        return
    
    print(f"\nNew movement gestures to collect: {new_gestures}")
    print(f"Existing movement gestures (skipped): {[g for g in MOVEMENT_GESTURES if g in existing_gestures]}")

    X_new = []
    y_new = []

    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        for gesture_name in new_gestures:
            # Assign label index: use existing length as starting point for new gestures
            label_idx = len(existing_gestures) + new_gestures.index(gesture_name)
            
            print(f"\nCollecting sequences for gesture: {gesture_name} (label index: {label_idx})")
            print(f"Need to collect {SEQUENCES_PER_GESTURE} sequences")
            print("Press 's' to start recording a sequence...")
            print("Press 'q' to quit early")
            
            sequence_count = 0
            while sequence_count < SEQUENCES_PER_GESTURE:
                # Wait for 's' to start recording
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Sequence: {sequence_count}/{SEQUENCES_PER_GESTURE}", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Press 's' to start recording", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.imshow("Sequence Data Collection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key != ord('s'):
                    continue
                
                # Collect sequence
                sequence = []
                frame_counter = 0
                collecting = True
                
                print(f"  Recording sequence {sequence_count + 1}...")
                
                while collecting and len(sequence) < SEQUENCE_LENGTH:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                            landmarks = normalize_landmarks(landmarks)
                            
                            # Add frame based on FRAME_SKIP setting
                            if frame_counter % FRAME_SKIP == 0:
                                sequence.append(landmarks)
                            
                            frame_counter += 1
                    else:
                        # If no hand detected, we can either pad with zeros or wait for hand
                        # For now, we'll pad with zeros to maintain sequence length
                        if frame_counter % FRAME_SKIP == 0:
                            sequence.append(np.zeros(63))  # 21 landmarks * 3 coords
                        frame_counter += 1

                    # Display progress
                    progress = len(sequence) / SEQUENCE_LENGTH * 100
                    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Sequence: {sequence_count + 1}/{SEQUENCES_PER_GESTURE}", (10, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(frame, f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.imshow("Sequence Data Collection", frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        collecting = False
                        break
                
                # If we collected a valid sequence, add it
                if len(sequence) == SEQUENCE_LENGTH:
                    X_new.append(np.array(sequence))
                    y_new.append(label_idx)
                    sequence_count += 1
                    print(f"  ✓ Collected sequence {sequence_count}/{SEQUENCES_PER_GESTURE}")
                else:
                    print(f"  ✗ Sequence incomplete, discarding...")

    cap.release()
    cv2.destroyAllWindows()

    # Combine old and new data
    if len(X_new) > 0:
        X_new = np.array(X_new)
        y_new = np.array(y_new)

        if len(X_old) > 0:
            X_combined = np.concatenate((X_old, X_new))
            y_combined = np.concatenate((y_old, y_new))
        else:
            X_combined = X_new
            y_combined = y_new

        # Update gesture mapping with new gestures
        updated_gestures = existing_gestures + new_gestures
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        np.save("data/sequences.npy", X_combined)
        np.save("data/sequence_labels.npy", y_combined)
        
        # Save updated gesture mapping
        with open(movement_gesture_mapping_file, 'wb') as f:
            pickle.dump(updated_gestures, f)

        print(f"\n✓ Added {X_new.shape[0]} new sequences.")
        print(f"Total sequence dataset size: {X_combined.shape[0]} sequences.")
        print(f"Updated movement gesture mapping: {updated_gestures}")
    else:
        print("\nNo sequences collected.")

if __name__ == "__main__":
    collect_sequence_data()

