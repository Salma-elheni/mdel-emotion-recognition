import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import time
import threading
from collections import Counter

# Load the model
model = load_model('./cnn_emotion_detection.h5')

# Define emotion labels and emojis
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
CLASS_LABELS_EMOJIS = ["👿", "🤢", "😱", "😊", "😐", "😔", "😲"]

# Semaphore to limit the number of concurrent threads
semaphore = threading.Semaphore(4)  # Max 4 threads can run simultaneously
lock = threading.Lock()  # Lock for protecting shared resources

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
    emotion_index = np.argmax(prediction)
    return CLASS_LABELS[emotion_index], CLASS_LABELS_EMOJIS[emotion_index]

# Thread function to process images
def process_image(image_path, results, emotion_counts):
    with semaphore:
        emotion_label, emotion_emoji = predict_emotion(image_path)
        with lock:
            results.append((image_path, emotion_label, emotion_emoji))
            emotion_counts[emotion_label] += 1
        print(f"{os.path.basename(image_path)}: {emotion_label} {emotion_emoji}")

# Function to process multiple images using threads
def predict_emotions_in_folders_and_count(parent_folder):
    results = []
    subfolder_emotion_counts = {}
    start_time = time.time()
    threads = []

    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            print(f"\nProcessing images in {subfolder}...")
            emotion_counts = Counter()
            
            for root, _, files in os.walk(subfolder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(root, file)
                        thread = threading.Thread(target=process_image, args=(image_path, results, emotion_counts))
                        threads.append(thread)
                        thread.start()
            
            subfolder_emotion_counts[subfolder] = emotion_counts
    
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nProcessing time for all images: {total_time:.2f} seconds ({int(total_time // 60)} minute)")
    
    return results, total_time, subfolder_emotion_counts

# Specify the parent folder containing subfolders
test_folder = "./test_images"

# Run predictions on all images in all subfolders
results, total_time, subfolder_emotion_counts = predict_emotions_in_folders_and_count(test_folder)

# Save results to a file
with open("emotion_results.txt", "w", encoding="utf-8") as f:
    for image_path, label, emoji in results:
        f.write(f"{image_path}: {label} {emoji}\n")

print("\nAll predictions are saved in 'emotion_results.txt'")

# Print emotion counts per subfolder
print("\nEmotion counts per subfolder:")
for subfolder, emotion_counts in subfolder_emotion_counts.items():
    print(f"\n{subfolder}:")
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count}")
