import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

# Check if TensorFlow/Keras is available
try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import keras
        from keras import layers
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        print("Warning: TensorFlow/Keras not found. Using sklearn as fallback.")

if not TENSORFLOW_AVAILABLE:
    # Fallback to a simpler approach using sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

# Load your saved sequence training data
sequences_file = 'data/sequences.npy'
labels_file = 'data/sequence_labels.npy'

if not os.path.exists(sequences_file) or not os.path.exists(labels_file):
    print("Error: Sequence data files not found.")
    print("Please run sequence_trainer.py first to collect sequence data.")
    exit()

X = np.load(sequences_file)  # Shape: (n_sequences, sequence_length, features)
y = np.load(labels_file)     # Shape: (n_sequences,)

# Load gesture mapping
movement_gesture_mapping_file = "data/movement_gesture_mapping.pkl"
if os.path.exists(movement_gesture_mapping_file):
    with open(movement_gesture_mapping_file, 'rb') as f:
        gestures = pickle.load(f)
    print(f"Loaded movement gesture mapping: {gestures}")
else:
    print("WARNING: No movement gesture mapping found. Using default from config.")
    from config import MOVEMENT_GESTURES
    gestures = MOVEMENT_GESTURES

print(f"\nDataset shape: {X.shape}")
print(f"Number of sequences: {X.shape[0]}")
print(f"Sequence length: {X.shape[1]}")
print(f"Features per frame: {X.shape[2]}")
print(f"Number of classes: {len(gestures)}")
print(f"Gesture mapping: {gestures}")

# Split into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape[0]} sequences")
print(f"Validation set: {X_val.shape[0]} sequences")

if TENSORFLOW_AVAILABLE:
    # Build LSTM model for sequence classification
    model = keras.Sequential([
        layers.LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        layers.Dropout(0.3),
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(len(gestures), activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Train the model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save the trained model and gesture mapping together
    model_data = {
        'model': model,
        'gestures': gestures,
        'sequence_length': X.shape[1],
        'model_type': 'lstm'
    }
    
    # Save Keras model separately and metadata
    model.save('models/sequence_model.h5')
    with open('models/sequence_model_metadata.pkl', 'wb') as f:
        pickle.dump({
            'gestures': gestures,
            'sequence_length': X.shape[1],
            'model_type': 'lstm'
        }, f)
    
    print(f"\n✓ Model trained and saved as sequence_model.h5")
    print(f"✓ Metadata saved as sequence_model_metadata.pkl")
    print(f"Gesture mapping: {gestures}")
    print(f"Number of classes: {len(gestures)}")
    
else:
    # Fallback: Flatten sequences and use RandomForest
    print("\nUsing sklearn RandomForest (fallback - less optimal for sequences)...")
    print("For better results, install TensorFlow: pip install tensorflow")
    
    # Flatten sequences: (n_sequences, sequence_length, features) -> (n_sequences, sequence_length * features)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    
    # Train RandomForest
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    val_accuracy = model.score(X_val_scaled, y_val)
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Save model and metadata
    model_data = {
        'model': model,
        'scaler': scaler,
        'gestures': gestures,
        'sequence_length': X.shape[1],
        'model_type': 'randomforest'
    }
    
    with open('models/sequence_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✓ Model trained and saved as sequence_model.pkl")
    print(f"Gesture mapping: {gestures}")
    print(f"Number of classes: {len(gestures)}")

