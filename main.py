import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import time  # Import time module
from collections import Counter  # For counting emotion occurrences

# Load the model
model = load_model('./cnn_emotion_detection.h5')

# Define emotion labels and emojis
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
CLASS_LABELS_EMOJIS = ["👿", "🤢", "😱", "😊", "😐", "😔", "😲"]

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_REDUCED_GRAYSCALE_2)  # Lecture de l'image en niveaux de gris
    img = cv2.resize(img, (48, 48))  # Redimensionnement à 48x48 pixels (taille attendue par le modèle)
    img = cv2.equalizeHist(img)  # Égalisation de l'histogramme pour améliorer le contraste
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Flou léger pour réduire le bruit
    # Filtre pour renforcer les contours (netteté)
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    img = cv2.filter2D(img, -1, sharpening_kernel)
    img = img.astype('float32') / 255.0  # Normalisation des pixels entre 0 et 1
    img = np.expand_dims(img, axis=0)  # Ajout de la dimension batch
    img = np.expand_dims(img, axis=-1)  # Ajout de la dimension channel
    return img

# Function to predict emotion
def predict_emotion(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)  # Get index of highest probability
    emotion_label = CLASS_LABELS[emotion_index]
    emotion_emoji = CLASS_LABELS_EMOJIS[emotion_index]
    return emotion_label, emotion_emoji

# Function to process multiple images in all subfolders under a folder and count emotions
def predict_emotions_in_folders_and_count(parent_folder):
    results = []
    subfolder_emotion_counts = {}  # Dictionary to store emotion counts per subfolder
    start_time = time.time()  # Start timing

    # Loop through all subdirectories in the parent folder
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        if os.path.isdir(subfolder_path):  # Process only subfolders
            print(f"\nProcessing images in {subfolder}...")
            emotion_counts = Counter()  # Counter to track emotion counts for this subfolder
            
            for root, _, files in os.walk(subfolder_path):  # Walk through files in the subfolder
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only process image files
                        image_path = os.path.join(root, file)
                        emotion_label, emotion_emoji = predict_emotion(image_path)
                        results.append((image_path, emotion_label, emotion_emoji))
                        emotion_counts[emotion_label] += 1  # Increment the count for the detected emotion
                        print(f"{file}: {emotion_label} {emotion_emoji}")
            
            # Store the emotion counts for this subfolder
            subfolder_emotion_counts[subfolder] = dict(emotion_counts)

    end_time = time.time()  # End timing
    total_time = end_time - start_time
    # Calculate minutes and seconds
    minutes = total_time // 60

    # Print the time in seconds and minutes
    print(f"\nProcessing time for all images: {total_time:.2f} seconds ({int(minutes)} minute)")

    return results, total_time, subfolder_emotion_counts

# Specify the parent folder containing subfolders
test_folder = "./test_images"

# Run predictions on all images in all subfolders and measure time
results, total_time, subfolder_emotion_counts = predict_emotions_in_folders_and_count(test_folder)

# Optionally, save results to a file with UTF-8 encoding
with open("emotion_results.txt", "w", encoding="utf-8") as f:
    for image_path, label, emoji in results:
        f.write(f"{image_path}: {label} {emoji}\n")

print("\nAll predictions are saved in 'emotion_results.txt'")

# Print emotion counts for each subfolder
print("\nEmotion counts per subfolder:")
for subfolder, emotion_counts in subfolder_emotion_counts.items():
    print(f"\n{subfolder}:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}")
