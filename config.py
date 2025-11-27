# Configuration file for Hand Sign Translator
# This is the single source of truth for gesture definitions

# List of all gestures/letters to recognize
# Add new letters here when you want to train them
GESTURES = ["A", "B", "C", "L", "V"]

# Number of samples to collect per gesture during training
SAMPLES_PER_GESTURE = 300

# === Sequence-based (movement) gesture settings ===
# Gestures that require movement to be recognized
MOVEMENT_GESTURES = ["Hello", "We"]

# Number of sequences to collect per movement gesture
SEQUENCES_PER_GESTURE = 20

# Sequence length: number of frames in each sequence
# Should capture the full movement (e.g., 30 frames at 30fps = 1 second)
SEQUENCE_LENGTH = 30

# Frame skip: collect every Nth frame (1 = all frames, 2 = every other frame, etc.)
FRAME_SKIP = 1

LOW_LIGHT = 2500

