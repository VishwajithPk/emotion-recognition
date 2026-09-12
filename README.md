# Emotion Recognition using CNN

A real-time **facial emotion recognition system** built using **Python, OpenCV, TensorFlow, and a Convolutional Neural Network (CNN)**.

The system detects a face from an image or webcam feed and classifies the facial expression into one of **7 emotions**:

* 😊 Happy
* 😢 Sad
* 😲 Surprised
* 😠 Angry
* 😨 Fear
* 😐 Neutral
* 🤢 Disgust

The project contains a training pipeline that automatically detects faces from the dataset, extracts emotion labels from filenames, trains a CNN model, and saves the trained model for real-time prediction.

---

## Features

* Real-time emotion recognition using a webcam
* Face detection using OpenCV Haar Cascade
* CNN-based emotion classification
* Supports 7 different emotions
* Automatic emotion labeling from image filenames
* Grayscale image processing
* Image normalization
* Stratified train/validation split
* Prediction confidence displayed on the webcam feed
* Trained model saved as an `.h5` file

---

## How It Works

The project consists of two main stages:

### 1. Model Training

`training.py` performs the following steps:

```text
Dataset
   ↓
Read image filenames
   ↓
Extract emotion label
   ↓
Detect face using Haar Cascade
   ↓
Crop largest detected face
   ↓
Resize to 96 × 96
   ↓
Normalize pixel values
   ↓
Train / Validation Split
   ↓
CNN Training
   ↓
Save trained model
```

The trained model is saved as:

```text
emotion_cnn_7classes.h5
```

### 2. Real-Time Testing

`test.py` loads the trained model and accesses the computer's webcam.

```text
Webcam
   ↓
Capture frame
   ↓
Convert to grayscale
   ↓
Detect face
   ↓
Crop face
   ↓
Resize
   ↓
Normalize
   ↓
CNN prediction
   ↓
Display emotion + confidence
```

Press **`q`** to close the webcam window.

---

## Emotion Classes

| Index | Code | Emotion   |
| ----: | ---- | --------- |
|     0 | HA   | Happy     |
|     1 | SA   | Sad       |
|     2 | SU   | Surprised |
|     3 | AN   | Angry     |
|     4 | FE   | Fear      |
|     5 | NE   | Neutral   |
|     6 | DI   | Disgust   |

---

## CNN Architecture

The project uses a relatively simple CNN architecture:

```text
Input: 96 × 96 × 1
        ↓
Conv2D (32 filters)
        ↓
MaxPooling
        ↓
Conv2D (64 filters)
        ↓
MaxPooling
        ↓
Conv2D (128 filters)
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense (128 neurons)
        ↓
Dropout (0.5)
        ↓
Dense (7 neurons)
        ↓
Softmax
```

### Configuration

* **Input image size:** 96 × 96
* **Input channels:** 1 (grayscale)
* **Number of classes:** 7
* **Optimizer:** Adam
* **Learning rate:** `0.001`
* **Loss function:** Categorical Crossentropy
* **Epochs:** 20
* **Batch size:** 32
* **Dropout:** 0.5

---

## Dataset

The training script expects the dataset to be located at:

```text
mixed/unzipped
```

You can change the dataset location in `training.py`:

```python
DATA_DIR = r"mixed/unzipped"
```

### Filename-Based Labeling

Emotion labels are extracted from the image filename.

The supported codes are:

```text
HA → Happy
SA → Sad
SU → Surprised
AN → Angry
FE → Fear
NE → Neutral
DI → Disgust
```

For example:

```text
person_HA_001.jpg
person_SA_002.jpg
person_AN_003.jpg
```

The training script uses regular expressions to identify the emotion code and avoid accidental matches inside unrelated words.

Images with:

* No valid emotion code
* Multiple emotion codes
* Unreadable image files
* No detected face

are skipped automatically.

If multiple faces are detected, the **largest detected face** is selected for training.

---

## Project Structure

A typical project structure is:

```text
emotion-recognition/
│
├── training.py
├── test.py
├── requirements.txt
├── emotion_cnn_7classes.h5
│
└── mixed/
    └── unzipped/
        ├── image1.jpg
        ├── image2.jpg
        ├── image3.jpg
        └── ...
```

> The dataset and trained model may be excluded from GitHub depending on their size and licensing.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/emotion-recognition.git
cd emotion-recognition
```

Replace the repository URL with your actual GitHub repository URL.

### 2. Create a Virtual Environment

It is recommended to use a virtual environment.

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

The project uses:

```text
opencv-python
numpy
tensorflow
scikit-learn
```

These are provided in `requirements.txt`.

---

## Training the Model

Place your dataset inside:

```text
mixed/unzipped
```

Then run:

```bash
python training.py
```

The script will:

1. Scan the dataset.
2. Identify emotion labels from filenames.
3. Detect faces.
4. Extract the largest face.
5. Resize the face to `96 × 96`.
6. Normalize pixel values.
7. Split the data into training and validation sets.
8. Train the CNN for 20 epochs.
9. Save the trained model.

After successful training:

```text
emotion_cnn_7classes.h5
```

will be created.

---

## Running Real-Time Emotion Recognition

After training the model, run:

```bash
python test.py
```

The program will open your default webcam.

A detected face will be displayed with its predicted emotion and confidence score.

Example:

```text
HA (Happy) 94.3%
```

To exit:

```text
Press Q
```

---

## Important Note About Image Size

The training script currently uses:

```python
IMG_SIZE = 96
```

However, `test.py` currently uses:

```python
IMG_SIZE = 160
```

For consistency, the inference preprocessing should ideally use the **same input dimensions as the model**.

Since the CNN is trained with:

```text
96 × 96 × 1
```

the testing script should use:

```python
IMG_SIZE = 96
```

instead of `160`.

This ensures that the preprocessing pipeline matches the dimensions used during training.

---

## Face Detection

The project uses OpenCV's Haar Cascade face detector:

```python
haarcascade_frontalface_default.xml
```

The detector searches for faces in grayscale frames.

The main detection parameters are:

```python
scaleFactor=1.3
minNeighbors=4
minSize=(30, 30)
```

When multiple faces are detected, the system selects the face with the largest area.

---

## Model Output

The CNN produces seven probabilities using the Softmax activation function.

The class with the highest probability is selected as the predicted emotion.

For example:

```text
Happy      0.82
Sad        0.04
Surprised  0.03
Angry      0.05
Fear       0.02
Neutral    0.03
Disgust    0.01
```

The predicted result would be:

```text
Happy — 82%
```

---

## Limitations

The current implementation has several limitations:

* Haar Cascade detection may struggle with difficult lighting conditions.
* Recognition accuracy depends heavily on the quality and diversity of the training dataset.
* Facial expressions can be ambiguous between different emotion classes.
* The model is relatively lightweight and uses a simple CNN architecture.
* Real-time performance depends on the computer's CPU/GPU and webcam.
* The current system processes the largest detected face when multiple faces are present.
* No data augmentation is currently implemented.
* The model's confidence score should not necessarily be interpreted as a calibrated probability.

---

## Future Improvements

Possible improvements include:

* Add image augmentation.
* Add Batch Normalization.
* Use data augmentation such as rotation, flipping, zoom, and brightness changes.
* Experiment with deeper CNN architectures.
* Use transfer learning with models such as MobileNet or EfficientNet.
* Improve face detection using modern detectors.
* Add emotion tracking across video frames.
* Add support for multiple simultaneous faces.
* Display an emotion distribution chart.
* Improve inference performance using GPU acceleration.
* Add model evaluation using confusion matrices and classification reports.
* Add a graphical user interface.
* Deploy the model as a web application.

---

## Technologies Used

| Technology         | Purpose                              |
| ------------------ | ------------------------------------ |
| Python             | Programming language                 |
| OpenCV             | Face detection and webcam processing |
| TensorFlow / Keras | CNN model and training               |
| NumPy              | Numerical and image processing       |
| Scikit-learn       | Dataset splitting                    |

---

## Files

### `training.py`

Responsible for:

* Dataset loading
* Filename-based emotion labeling
* Face detection
* Face extraction
* Image preprocessing
* Train/validation splitting
* CNN creation
* Model training
* Model saving

### `test.py`

Responsible for:

* Loading the trained model
* Opening the webcam
* Detecting faces
* Preprocessing detected faces
* Predicting emotions
* Displaying predictions and confidence scores

### `requirements.txt`

Contains the Python dependencies required to run the project.

---

## Example

After starting the testing script:

```bash
python test.py
```

the webcam window detects a face and displays something similar to:

```text
┌─────────────────────────┐
│                         │
│       FACE              │
│                         │
└─────────────────────────┘
  HA (Happy) 94.3%
```

The prediction is updated continuously as the webcam captures new frames.

---

## License

This project is intended for educational and experimental purposes.

If you use a third-party dataset, model, or other external resource, make sure to comply with its respective license and usage requirements.

---

## Author

**Vishwajith P K**

GitHub:
https://github.com/VishwajithPk

---
