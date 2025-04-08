import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import time
import threading
import queue
import time
from tqdm import tqdm
from collections import defaultdict

# ================================
# 🎭 Emotion Labels
# ================================
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
CLASS_LABELS_EMOJIS = ["👿", "🤢", "😱", "😊", "😐", "😔", "😲"]

# ================================
# 📌 Load model globally
# ================================
model = load_model('./cnn_emotion_detection.h5')
model_lock = threading.Lock()

# ================================
# 🎯 Preprocessing
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
    img = preprocess_image(image_path)
    with model_lock:
        prediction = model.predict(img)
    emotion_index = np.argmax(prediction)
    return CLASS_LABELS[emotion_index], CLASS_LABELS_EMOJIS[emotion_index]

# ================================
# 🧵 Threads with Sleep/Wakeup
# ================================
queue = Queue()
condition = threading.Condition()
results = []
subfolder_emotion_counts = {}

def producer(image_paths):
    with condition:
        for path in image_paths:
            queue.put(path)
        condition.notify_all()  # Wake all consumers
        queue.put(None)  # Sentinel value for termination (one per thread)

def consumer(thread_id):
    while True:
        with condition:
            while queue.empty():
                condition.wait()
            image_path = queue.get()
            if image_path is None:
                queue.put(None)  # Pass on the sentinel
                break

        emotion_label, emotion_emoji = predict_emotion(image_path)
        folder_name = os.path.basename(os.path.dirname(image_path))

        with condition:
            results.append((image_path, emotion_label, emotion_emoji))
            if folder_name not in subfolder_emotion_counts:
                subfolder_emotion_counts[folder_name] = Counter()
            subfolder_emotion_counts[folder_name][emotion_label] += 1

# ================================
# 🏁 Main Execution
# ================================
def run_threaded_pipeline(parent_folder, num_threads=4):
    all_image_paths = []

    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for root, _, files in os.walk(subfolder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_image_paths.append(os.path.join(root, file))

    start_time = time.time()

    # Start consumers
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=consumer, args=(i,))
        t.start()
        threads.append(t)

    # Start producer
    producer(all_image_paths)

    for t in threads:
        t.join()

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n⏱️ Total execution time: {total_time:.2f} sec ({int(total_time // 60)} min)")
    return results, total_time, subfolder_emotion_counts

# ================================
# 🚀 Run
# ================================
if __name__ == "__main__":
    test_folder = "./test_images"

    results, total_time, subfolder_emotion_counts = run_threaded_pipeline(test_folder, num_threads=4)

    with open("emotion_results.txt", "w", encoding="utf-8") as f:
        for image_path, label, emoji in results:
            f.write(f"{image_path}: {label} {emoji}\n")

    print("\n✅ All predictions are saved in 'emotion_results.txt'")

    print("\n📊 Emotion counts per subfolder:")
    for subfolder, emotion_counts in subfolder_emotion_counts.items():
        print(f"\n{subfolder}:")
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count}")



