import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the model
model = load_model('./cnn_emotion_detection.h5')

# Define emotion labels and emojis
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
CLASS_LABELS_EMOJIS = ["👿", "🤢", "😱", "😊", "😐", "😔", "😲"]

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1) # Add channel dimension
    return img

# Function to predict emotion
def predict_emotion(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)  # Get index of highest probability
    emotion_label = CLASS_LABELS[emotion_index]
    emotion_emoji = CLASS_LABELS_EMOJIS[emotion_index]
    return emotion_label, emotion_emoji

# Example usage
image_path = 'ouss2.jpg'
emotion_label, emotion_emoji = predict_emotion(image_path)

print(f'The predicted emotion is: {emotion_label} {emotion_emoji}')
