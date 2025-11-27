# Hand Sign Translator

A real-time hand sign recognition system that can recognize both static hand gestures (letters) and dynamic movement-based gestures using MediaPipe and machine learning. The system supports both webcam and ESP32 camera streaming, with automatic light level detection for optimal performance.

## Table of Contents

- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Hardware Setup](#hardware-setup)
- [Configuration](#configuration)
- [Training Instructions](#training-instructions)
  - [Training Static Gesture Model](#training-static-gesture-model)
  - [Training Sequence/Movement Model](#training-sequencemovement-model)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Technical Details](#technical-details)

## Features

- **Dual Model Support**:
  - Static gesture recognition (letters/characters) using Random Forest
  - Dynamic sequence-based gesture recognition (movements) using LSTM or Random Forest
  
- **Multiple Camera Sources**:
  - Webcam support
  - ESP32 camera streaming via WiFi
  
- **Real-time Processing**:
  - Optimized frame skipping for performance
  - Cached predictions to prevent flickering
  - Low-latency hand tracking using MediaPipe
  
- **Smart Features**:
  - Automatic light level detection (ESP32 only)
  - Incremental training (add new gestures without retraining existing ones)
  - Position-invariant landmark normalization
  
- **User-Friendly Interface**:
  - Live video feed with hand landmark visualization
  - Real-time gesture prediction display
  - Easy camera source switching

## Hardware Requirements

### For Webcam-Only Setup:
- Computer with webcam
- Python 3.7+ installed

### For ESP32 Camera Setup:
- **ESP32 Wrover Module** (or compatible ESP32 board with PSRAM)
  - Recommended: ESP32-WROVER-KIT
  - Alternative: AI Thinker ESP32-CAM (with PSRAM)
  
- **Camera Module**:
  - OV2640 camera module (compatible with ESP32-CAM boards)
  
- **Additional Components**:
  - Light sensor connected to ADC pin (GPIO 32) - optional but recommended
  - USB cable for programming ESP32
  - Power supply (USB or external)
  - WiFi network for streaming

### ESP32 Camera Pin Configuration

The project uses the standard ESP32-CAM pin configuration:
- Camera pins are automatically configured based on your camera model
- ADC pin for light sensor: **GPIO 32**
- Ensure your ESP32 board has PSRAM for best performance

## Software Requirements

### Python Packages

The following Python packages are required:

```
opencv-python>=4.5.0
mediapipe>=0.8.0
numpy>=1.19.0
scikit-learn>=0.24.0
requests>=2.25.0
tensorflow>=2.5.0 (optional, for LSTM sequence model)
```

**Note**: TensorFlow is optional. If not installed, the sequence model will use scikit-learn's Random Forest as a fallback (less optimal but functional).

### ESP32 Firmware

- Arduino IDE 1.8.13 or later
- ESP32 Board Support Package
- Required libraries (installed via Arduino Library Manager):
  - `WiFi` (included with ESP32 core)
  - `esp_camera` (included with ESP32 core)
  - `esp_http_server` (included with ESP32 core)

## Project Structure

```
HandSignTranslator/
│
├── README.md                      # This file
├── config.py                      # Configuration file for gestures and settings
│
├── Image_Detection.py             # Static gesture recognition (main script)
├── Video_Detection.py             # Sequence/movement gesture recognition
│
├── medaipipe_Trainer.py           # Data collection for static gestures
├── train_model.py                 # Train static gesture model
│
├── sequence_trainer.py            # Data collection for movement gestures
├── train_sequence_model.py        # Train sequence/movement model
│
├── streamer.py                    # Simple ESP32 stream viewer (testing)
├── veiwer.py                      # Alternative ESP32 stream viewer
│
├── CameraWebServer/               # ESP32 firmware
│   ├── CameraWebServer.ino        # Main Arduino sketch
│   ├── camera_pins.h              # Camera pin definitions
│   ├── camera_index.h             # Web interface HTML
│   └── app_httpd.cpp              # HTTP server implementation
│
├── data/                          # Training data storage
│   ├── landmarks.npy              # Static gesture landmarks
│   ├── labels.npy                 # Static gesture labels
│   ├── gesture_mapping.pkl        # Static gesture mapping
│   ├── sequences.npy              # Sequence gesture data
│   ├── sequence_labels.npy        # Sequence gesture labels
│   └── movement_gesture_mapping.pkl  # Sequence gesture mapping
│
└── models/                        # Trained models
    ├── hand_sign_model.pkl        # Static gesture model
    ├── sequence_model.h5          # Sequence model (TensorFlow)
    ├── sequence_model.pkl         # Sequence model (sklearn fallback)
    └── sequence_model_metadata.pkl # Sequence model metadata
```

## Installation

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd HandSignTranslator
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 3. Install Python Dependencies

**Option A: Install from requirements.txt (Recommended)**

```bash
pip install -r requirements.txt
```

This installs all dependencies including TensorFlow. If you prefer not to install TensorFlow, you can comment it out in `requirements.txt` and the sequence model will use scikit-learn as a fallback.

**Option B: Install manually**

```bash
pip install opencv-python mediapipe numpy scikit-learn requests

# Optional: For LSTM sequence model (recommended)
pip install tensorflow
```

### 4. Verify Installation

Check that all packages are installed correctly:

```bash
python -c "import cv2, mediapipe, numpy, sklearn, requests; print('All packages installed successfully!')"
```

If you installed TensorFlow, verify it:

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

## Hardware Setup

### ESP32 Camera Setup

1. **Assemble the Hardware**:
   - Connect the OV2640 camera module to your ESP32-CAM board
   - Connect a light sensor (photoresistor with voltage divider) to GPIO 32 (optional)
   - Ensure proper power supply (5V via USB or external supply)

2. **Configure WiFi Settings**:
   - Open `CameraWebServer/CameraWebServer.ino` in Arduino IDE
   - Update WiFi credentials:
     ```cpp
     const char* ssid = "YOUR_WIFI_SSID";
     const char* password = "YOUR_WIFI_PASSWORD";
     ```

3. **Select Board and Upload**:
   - In Arduino IDE: **Tools** → **Board** → Select your ESP32 board
   - Select the correct port: **Tools** → **Port**
   - Upload the sketch to your ESP32

4. **Get ESP32 IP Address**:
   - Open Serial Monitor (115200 baud)
   - After WiFi connection, note the IP address printed (e.g., `172.20.10.3`)

5. **Configure Python Scripts**:
   - Update the IP address in detection scripts:
     - `Image_Detection.py`: Change `ESP32_IP = "172.20.10.3"`
     - `Video_Detection.py`: Change `ESP32_IP = "172.20.10.3"`
     - `streamer.py`: Update the URL if needed

6. **Test the Stream**:
   - Run `python streamer.py` to verify the camera stream is working
   - Press 'q' to quit

### Light Sensor Setup (Optional)

If you want automatic light level detection:

1. Connect a photoresistor with a voltage divider circuit to GPIO 32
2. The system will automatically detect low light conditions
3. Adjust the threshold in `config.py`: `LOW_LIGHT = 2500`

## Configuration

Edit `config.py` to customize your setup:

```python
# Static gestures (letters/characters)
GESTURES = ["A", "B", "C", "L", "V"]

# Number of samples per gesture during training
SAMPLES_PER_GESTURE = 300

# Movement-based gestures (sequences)
MOVEMENT_GESTURES = ["Hello", "We"]

# Number of sequences per movement gesture
SEQUENCES_PER_GESTURE = 20

# Sequence length (number of frames)
SEQUENCE_LENGTH = 30

# Frame skip during sequence collection
FRAME_SKIP = 1

# Low light threshold (ADC value, lower = darker)
LOW_LIGHT = 2500
```

## Training Instructions

### Training Static Gesture Model

Static gestures are single-frame hand poses (like letters).

#### Step 1: Collect Training Data

```bash
python medaipipe_Trainer.py
```

**Process**:
1. The script will check for existing gestures and only collect data for new ones
2. For each gesture:
   - Press **'s'** to start collecting samples
   - Hold the hand sign steady in front of the camera
   - The script collects 300 samples (configurable) automatically
   - Hand landmarks are extracted and normalized
3. Press **'q'** to quit early if needed

**Tips**:
- Ensure good lighting
- Hold the gesture steady and clear
- Vary your hand position slightly for better generalization
- Collect data in different lighting conditions if possible

#### Step 2: Train the Model

```bash
python train_model.py
```

**Process**:
1. Loads collected landmarks and labels from `data/` directory
2. Trains a Random Forest classifier (200 estimators)
3. Saves the trained model to `models/hand_sign_model.pkl`
4. Prints gesture mapping and model information

**Output**:
- `models/hand_sign_model.pkl` - Trained model with gesture mapping
- Console output showing gesture mapping and class count

#### Step 3: Test the Model

```bash
python Image_Detection.py
```

**Controls**:
- **'q'** - Quit
- **'w'** - Switch between ESP32 stream and webcam

The model will recognize static gestures in real-time!

### Training Sequence/Movement Model

Movement gestures require sequences of frames (like waving or signing words).

#### Step 1: Collect Sequence Data

```bash
python sequence_trainer.py
```

**Process**:
1. The script checks for existing movement gestures
2. For each movement gesture:
   - Press **'s'** to start recording a sequence
   - Perform the gesture/movement (it will record 30 frames by default)
   - Repeat for the required number of sequences (20 by default)
   - Press **'q'** to quit early
3. Data is saved incrementally - you can add more sequences later

**Tips**:
- Perform the movement smoothly and consistently
- Complete each sequence within the frame limit
- Ensure hand is visible throughout the entire sequence
- Vary the speed slightly for better generalization

#### Step 2: Train the Sequence Model

**Option A: Using TensorFlow (Recommended)**

If TensorFlow is installed:

```bash
python train_sequence_model.py
```

**Process**:
1. Loads sequence data from `data/sequences.npy`
2. Splits data into training (80%) and validation (20%) sets
3. Trains an LSTM neural network:
   - Architecture: LSTM(128) → Dropout → LSTM(64) → Dropout → Dense layers
   - Optimizer: Adam
   - Epochs: 50
   - Batch size: 32
4. Saves model to `models/sequence_model.h5` and metadata to `models/sequence_model_metadata.pkl`
5. Displays validation accuracy and loss

**Output**:
- `models/sequence_model.h5` - TensorFlow/Keras model
- `models/sequence_model_metadata.pkl` - Model metadata (gestures, sequence length, etc.)
- Validation accuracy printed to console

**Option B: Using scikit-learn (Fallback)**

If TensorFlow is not installed, the script automatically falls back to Random Forest:

```bash
python train_sequence_model.py
```

**Process**:
1. Flattens sequences and uses StandardScaler
2. Trains Random Forest classifier
3. Saves to `models/sequence_model.pkl`

**Note**: LSTM models typically perform better for sequential data, but Random Forest works as a fallback.

#### Step 3: Test the Sequence Model

```bash
python Video_Detection.py
```

**Controls**:
- **'q'** - Quit
- **'w'** - Switch between ESP32 stream and webcam

The model recognizes movement gestures after collecting 30 frames of hand movement!

### Adding New Gestures

#### Adding Static Gestures

1. Edit `config.py` and add to `GESTURES` list:
   ```python
   GESTURES = ["A", "B", "C", "L", "V", "D", "E"]  # Added D and E
   ```

2. Run data collection:
   ```bash
   python medaipipe_Trainer.py
   ```
   Only new gestures will be collected.

3. Retrain the model:
   ```bash
   python train_model.py
   ```

#### Adding Movement Gestures

1. Edit `config.py` and add to `MOVEMENT_GESTURES` list:
   ```python
   MOVEMENT_GESTURES = ["Hello", "We", "Thank You"]  # Added "Thank You"
   ```

2. Run sequence collection:
   ```bash
   python sequence_trainer.py
   ```

3. Retrain the sequence model:
   ```bash
   python train_sequence_model.py
   ```

## Usage

### Running Static Gesture Recognition

```bash
python Image_Detection.py
```

**Features**:
- Real-time hand detection using MediaPipe
- Static gesture prediction displayed on screen
- Hand landmarks visualization
- Automatic light level detection (ESP32 only)
- Smooth prediction display with buffering

**Controls**:
- **'q'** - Quit application
- **'w'** - Toggle between webcam and ESP32 stream

### Running Movement Gesture Recognition

```bash
python Video_Detection.py
```

**Features**:
- Real-time sequence collection and prediction
- Movement gesture recognition after collecting a full sequence
- Optimized frame processing for performance
- Low light warning for ESP32

**Controls**:
- **'q'** - Quit application
- **'w'** - Toggle between webcam and ESP32 stream

**Note**: Movement gestures require a full sequence to be collected before prediction. The default sequence length is 30 frames.

### Performance Optimization

The scripts include several optimizations:

- **Frame Skipping**: ESP32 streams process every 3rd frame, webcam processes every frame
- **Resized Processing**: MediaPipe runs on 640x480 for speed, display is 1280x960
- **Caching**: Prediction results are cached to prevent flickering
- **Buffer Management**: Efficient JPEG frame extraction from ESP32 stream

Adjust these settings in the detection scripts if needed:
```python
FRAME_SKIP_ESP32 = 3   # Process every Nth frame for ESP32
FRAME_SKIP_WEBCAM = 1  # Process every Nth frame for webcam
PROCESS_WIDTH = 640    # Processing resolution
PROCESS_HEIGHT = 480
```

## Troubleshooting

### Camera Issues

**Problem**: Webcam not detected
- **Solution**: Check camera permissions and ensure no other application is using it
- Try changing camera index: `cv2.VideoCapture(1)` instead of `0`

**Problem**: ESP32 stream not connecting
- **Solution**: 
  - Verify ESP32 IP address is correct
  - Check WiFi connection on ESP32
  - Ensure ESP32 is powered and camera module is connected
  - Try accessing `http://<ESP32_IP>` in a web browser

**Problem**: Stream disconnects frequently
- **Solution**: 
  - Check WiFi signal strength
  - Reduce ESP32 camera resolution in `CameraWebServer.ino`
  - Increase frame skip rate for ESP32

### Model Issues

**Problem**: Model not found error
- **Solution**: Ensure you've trained the model first using `train_model.py` or `train_sequence_model.py`
- Check that model files exist in `models/` directory

**Problem**: Poor recognition accuracy
- **Solution**:
  - Collect more training data (increase `SAMPLES_PER_GESTURE`)
  - Ensure consistent lighting during training and inference
  - Verify hand is clearly visible and gesture is held correctly
  - Retrain the model with more diverse data

**Problem**: Sequence model not loading
- **Solution**: 
  - Check if TensorFlow is installed for `.h5` models
  - Verify `models/sequence_model_metadata.pkl` exists
  - Ensure sequence length matches during training and inference

### Performance Issues

**Problem**: Low FPS / laggy performance
- **Solution**:
  - Increase `FRAME_SKIP_ESP32` or `FRAME_SKIP_WEBCAM`
  - Reduce `PROCESS_WIDTH` and `PROCESS_HEIGHT`
  - Close other applications using CPU/GPU
  - Use a more powerful computer

**Problem**: High CPU usage
- **Solution**: 
  - MediaPipe can be CPU-intensive
  - Reduce processing resolution
  - Increase frame skip rate
  - Consider using a GPU-accelerated OpenCV build

### ESP32 Issues

**Problem**: ESP32 won't connect to WiFi
- **Solution**:
  - Verify SSID and password are correct
  - Check WiFi signal strength (2.4GHz required)
  - Ensure WiFi credentials match exactly (case-sensitive)

**Problem**: Camera init failed
- **Solution**:
  - Verify camera module is properly connected
  - Check that ESP32 has PSRAM (required for camera)
  - Try selecting a different camera model in `CameraWebServer.ino`

**Problem**: ADC not reading correctly
- **Solution**:
  - Verify light sensor is connected to GPIO 32
  - Check voltage divider circuit
  - Adjust `LOW_LIGHT` threshold in `config.py`

## Technical Details

### Hand Landmark Normalization

Landmarks are normalized to be position and scale invariant:

1. **Translation**: Subtract wrist position (landmark 0) from all landmarks
2. **Scaling**: Divide by maximum absolute value across all landmarks

This ensures the model recognizes gestures regardless of hand position or size.

### Model Architectures

**Static Gesture Model**:
- Algorithm: Random Forest Classifier
- Estimators: 200
- Input: 63 features (21 landmarks × 3 coordinates, normalized)
- Output: Gesture class probabilities

**Sequence Model (TensorFlow)**:
- Architecture: LSTM-based neural network
  - Input: (sequence_length, 63)
  - LSTM(128) with dropout(0.3)
  - LSTM(64) with dropout(0.3)
  - Dense(64) with dropout(0.2)
  - Dense(num_gestures) with softmax
- Optimizer: Adam
- Loss: Sparse categorical crossentropy

**Sequence Model (scikit-learn fallback)**:
- Algorithm: Random Forest Classifier
- Preprocessing: StandardScaler
- Input: Flattened sequence (sequence_length × 63)

### Data Storage

- **Static gestures**: 
  - `data/landmarks.npy` - Feature vectors (N, 63)
  - `data/labels.npy` - Class labels (N,)
  - `data/gesture_mapping.pkl` - Gesture name to index mapping

- **Sequence gestures**:
  - `data/sequences.npy` - Sequences (N, sequence_length, 63)
  - `data/sequence_labels.npy` - Class labels (N,)
  - `data/movement_gesture_mapping.pkl` - Gesture name to index mapping

### Incremental Training

Both training scripts support incremental data collection:
- Existing data is preserved
- Only new gestures need data collection
- Old and new data are combined during training
- Gesture mappings are automatically updated

This allows adding new gestures without retraining from scratch.

---

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Acknowledgments

- MediaPipe for hand tracking
- ESP32 Camera Web Server example as a base
- TensorFlow/Keras for sequence modeling

---

**Happy Signing!** 🤟

