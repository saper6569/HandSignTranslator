import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Load your saved training data
X = np.load('data/landmarks.npy')  # landmark features
y = np.load('data/labels.npy')     # gesture labels

# Load gesture mapping
gesture_mapping_file = "data/gesture_mapping.pkl"
if os.path.exists(gesture_mapping_file):
    with open(gesture_mapping_file, 'rb') as f:
        gestures = pickle.load(f)
    print(f"Loaded gesture mapping: {gestures}")
else:
    print("WARNING: No gesture mapping found. Using default from config.")
    from config import GESTURES
    gestures = GESTURES

# Create and train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Save the trained model and gesture mapping together
model_data = {
    'model': model,
    'gestures': gestures
}

with open('models/hand_sign_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"Model trained and saved as hand_sign_model.pkl")
print(f"Gesture mapping: {gestures}")
print(f"Number of classes: {len(gestures)}")
