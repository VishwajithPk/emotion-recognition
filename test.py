import cv2
import numpy as np
import tensorflow as tf

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "emotion_cnn_7classes.h5"
IMG_SIZE = 160

# index -> (code, full name)
INDEX_TO_LABEL = {
    0: ("HA", "Happy"),
    1: ("SA", "Sad"),
    2: ("SU", "Surprised"),
    3: ("AN", "Angry"),
    4: ("FE", "Fear"),
    5: ("NE", "Neutral"),
    6: ("DI", "Disgust"),
}

# ----------------------------
# LOAD MODEL
# ----------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ----------------------------
# LOAD FACE DETECTOR
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Failed to load Haar Cascade for face detection")

# ----------------------------
# START VIDEO CAPTURE
# ----------------------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Make a copy for drawing
    display_frame = frame.copy()

    # Convert to grayscale for face detection and model input
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=4,
        minSize=(30, 30),
    )

    for (x, y, w, h) in faces:
        # Crop face region
        face_gray = gray[y : y + h, x : x + w]

        # Resize to model input size
        face_resized = cv2.resize(face_gray, (IMG_SIZE, IMG_SIZE))

        # Normalize and add batch + channel dimension
        face_norm = face_resized.astype("float32") / 255.0
        face_input = np.expand_dims(face_norm, axis=-1)  # (H, W, 1)
        face_input = np.expand_dims(face_input, axis=0)  # (1, H, W, 1)

        # Predict
        preds = model.predict(face_input, verbose=0)[0]  # shape: (7,)
        class_idx = int(np.argmax(preds))
        prob = float(np.max(preds))

        code, name = INDEX_TO_LABEL.get(class_idx, ("??", "Unknown"))

        label_text = f"{code} ({name}) {prob*100:.1f}%"

        # Draw rectangle and label
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put label above the face box
        cv2.putText(
            display_frame,
            label_text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Show frame
    cv2.imshow("Live Emotion Recognition", display_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
