import os
import re
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# ----------------------------
# CONFIG
# ----------------------------
DATA_DIR = r"mixed/unzipped"  # change if needed
IMG_SIZE = 96
NUM_CLASSES = 7

# Emotion label codes (case-insensitive in filenames)
CODE_TO_INDEX = {
    "ha": 0,  # happy
    "sa": 1,  # sad
    "su": 2,  # surprised
    "an": 3,  # angry
    "fe": 4,  # fear
    "ne": 5,  # neutral
    "di": 6,  # disgust
}

INDEX_TO_NAME = {
    0: "happy (HA)",
    1: "sad (SA)",
    2: "surprised (SU)",
    3: "angry (AN)",
    4: "fear (FE)",
    5: "neutral (NE)",
    6: "disgust (DI)",
}

# Compile regex patterns to avoid false matches like "an" in "random"
# Pattern: (^|non-letter)CODE(non-letter|$), case-insensitive
CODE_PATTERNS = {
    code: re.compile(rf"(?:^|[^a-z]){code}(?:[^a-z]|$)", re.IGNORECASE)
    for code in CODE_TO_INDEX.keys()
}

# ----------------------------
# HELPER: Get label from filename
# ----------------------------
def get_label_from_filename(filename: str):
    """
    Return (class_index, code) if filename contains one of HA/SA/SU/AN/FE/NE/DI
    (case-insensitive), otherwise (None, None).
    """
    name_lower = filename.lower()

    matched_codes = []
    for code, pattern in CODE_PATTERNS.items():
        if pattern.search(name_lower):
            matched_codes.append(code)

    if len(matched_codes) == 1:
        code = matched_codes[0]
        return CODE_TO_INDEX[code], code.upper()
    elif len(matched_codes) == 0:
        return None, None
    else:
        # ambiguous: more than one code found in the same filename
        return None, None

# ----------------------------
# LOAD OPENCV FACE DETECTOR
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load haarcascade_frontalface_default.xml")

# ----------------------------
# LOAD DATA: face detection + label from filename
# ----------------------------
X = []
y = []
skipped = []  # list of (path, reason)

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")

for root, dirs, files in os.walk(DATA_DIR):
    for fname in files:
        if not fname.lower().endswith(VALID_EXT):
            # ignore non-image files
            continue

        fpath = os.path.join(root, fname)

        # 1) Label from filename
        label_index, used_code = get_label_from_filename(fname)
        if label_index is None:
            skipped.append((fpath, "no valid emotion code in filename or ambiguous codes"))
            continue

        # 2) Read image
        img = cv2.imread(fpath)
        if img is None:
            skipped.append((fpath, "cv2.imread failed"))
            continue

        # 3) Convert to grayscale for face detection / model
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 4) Face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=4,
            minSize=(30, 30),
        )

        if len(faces) == 0:
            skipped.append((fpath, "no face detected"))
            continue

        # If multiple faces, take the largest
        x, y0, w, h = max(faces, key=lambda b: b[2] * b[3])
        face = gray[y0 : y0 + h, x : x + w]

        # 5) Resize to fixed size
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        X.append(face)
        y.append(label_index)

# ----------------------------
# CONVERT TO NUMPY + ONE-HOT
# ----------------------------
X = np.array(X, dtype="float32")
y = np.array(y, dtype="int32")

print(f"Total images used: {len(X)}")
print(f"Total images skipped: {len(skipped)}")

if len(X) == 0:
    raise RuntimeError("No images were loaded. Check your path and filename labels.")

# Normalize to [0,1]
X = X / 255.0

# Add channel dimension for CNN (grayscale -> 1 channel)
X = np.expand_dims(X, axis=-1)

# One-hot labels
y_cat = tf.keras.utils.to_categorical(y, num_classes=NUM_CLASSES)

# ----------------------------
# TRAIN / VAL SPLIT
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)

print("Train samples:", X_train.shape[0])
print("Val samples:", X_val.shape[0])

# ----------------------------
# DEFINE A SIMPLE CNN MODEL
# ----------------------------
model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ----------------------------
# TRAIN
# ----------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
)

# ----------------------------
# SAVE MODEL
# ----------------------------
model.save("emotion_cnn_7classes.h5")
print("Model saved as emotion_cnn_7classes.h5")

# ----------------------------
# SHOW SKIPPED IMAGES
# ----------------------------
print("\nExamples of images that were NOT used:")

# show up to 50 skipped images, with reason
for i, (path, reason) in enumerate(skipped[:50]):
    print(f"[{i+1:02d}] {reason} -> {path}")

print("\nTotal skipped:", len(skipped))
