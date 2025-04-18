# Real-Time-Emotion-Recognition-Model
This project implements a convolutional neural network (CNN) to recognize human facial emotions using the FER2013 dataset. It includes model training, evaluation, real-time inference, and deployment options.
## 🧠 Model Architecture

The model uses a custom CNN consisting of:

- Convolutional layers with ReLU activation
- Batch normalization and max pooling
- Dropout layers for regularization
- Dense fully connected layers
- Final softmax layer for classification into 7 emotion categories

### Emotion Classes:
`['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']`

## 🗂 Dataset

- **Dataset:** FER2013 (Kaggle)
- **Format:** 48×48 grayscale face images with emotion labels
- **Classes:** 7
- **Split:** Pre-divided into training and test sets

## 🚀 Training Details

- **Framework:** TensorFlow/Keras
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- **Epochs:** Up to 50
- **Batch Size:** 64
- **Validation Split:** 10%

## 📊 Evaluation and Visualization

- **Accuracy:** 63.99%
- **Metrics:** Precision, Recall, F1-Score (via classification report)
- **Confusion Matrix:** Available in `emotion_recognition_project/confusion_matrix.png`
- **Training Curves:** `emotion_recognition_project/training_history.png`
- **Visualizations:**
  - `correct_predictions.png`
  - `incorrect_predictions.png`
  - `training_examples.png`

## 🖼️ Inference and Face Detection

- Face detection using OpenCV Haar cascades
- Real-time emotion prediction from images
- Annotated outputs saved as `emotion_detection_result.jpg`

## 📦 Deployment

- Trained model saved in HDF5 format (`.h5`)
- Optionally exported as TensorFlow SavedModel in `emotion_recognition_model_tf/` for:
  - TensorFlow Lite conversion
  - ONNX export
  - Web or mobile deployment

## ✅ Requirements

Install dependencies with:

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn
