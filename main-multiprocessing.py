import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import time
import multiprocessing as mp
from collections import Counter

# ================================
# 🔥 TensorFlow Optimizations
# ================================
import os
os.environ["OMP_NUM_THREADS"] = "4"  # Limits CPU threads
os.environ["TF_NUM_INTEROP_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# ================================
# 🎭 Emotion Labels
# ================================
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
CLASS_LABELS_EMOJIS = ["👿", "🤢", "😱", "😊", "😐", "😔", "😲"]

# ================================
# 📌 Load model once per process
# ================================
model = None  # Global model instance

def load_model_once():
    """ Load the model only once per process to optimize performance """
    global model
    if model is None:
        model = load_model('./cnn_emotion_detection.h5')

# ================================
# 🎯 Image Preprocessing
# ================================
def preprocess_image(image_path):
    """ Load and preprocess the image for emotion detection """
    img = cv2.imread(image_path, cv2.IMREAD_REDUCED_GRAYSCALE_2)  # Load in lower resolution
    img = cv2.resize(img, (48, 48))  # Resize to model's expected input
    img = img.astype('float32') / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

# ================================
# 🔍 Emotion Prediction
# ================================
def predict_emotion(image_path):
    """ Predict emotion for a single image """
    load_model_once()  # Ensure model is loaded
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    return CLASS_LABELS[emotion_index], CLASS_LABELS_EMOJIS[emotion_index]

# ================================
# ⚡ Parallel Image Processing
# ================================
def process_image(image_path):
    """ Wrapper function for multiprocessing """
    emotion_label, emotion_emoji = predict_emotion(image_path)
    return (image_path, emotion_label, emotion_emoji)

def predict_emotions_parallel(parent_folder):
    """ Process all images in parallel across multiple folders """
    results = []
    subfolder_emotion_counts = {}
    start_time = time.time()

    all_image_paths = []
    
    # Collect all image paths
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            emotion_counts = Counter()
            for root, _, files in os.walk(subfolder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_image_paths.append(os.path.join(root, file))

    # Process images in parallel
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:  # Use N-1 cores
        results = pool.map(process_image, all_image_paths)

    # Organize results
    for image_path, label, emoji in results:
        folder_name = os.path.basename(os.path.dirname(image_path))
        if folder_name not in subfolder_emotion_counts:
            subfolder_emotion_counts[folder_name] = Counter()
        subfolder_emotion_counts[folder_name][label] += 1

    end_time = time.time()
    total_time = end_time - start_time
    minutes = total_time // 60
    print(f"\nTotal execution time: {total_time:.2f} sec ({int(minutes)} min)")

    return results, total_time, subfolder_emotion_counts

# ================================
# 🏁 Run Emotion Prediction
# ================================
if __name__ == "__main__":
    test_folder = "./test_images"

    results, total_time, subfolder_emotion_counts = predict_emotions_parallel(test_folder)

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
