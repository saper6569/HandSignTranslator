import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load your saved training data
X = np.load('data/landmarks.npy')  # landmark features
y = np.load('data/labels.npy')     # gesture labels

# Create and train the model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Save the trained model
with open('hand_sign_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as hand_sign_model.pkl")
