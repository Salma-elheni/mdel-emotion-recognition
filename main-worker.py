import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import time
import threading
from queue import Queue
from collections import Counter, defaultdict

# ================================
# 🎭 Emotion Labels
# ================================
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
CLASS_LABELS_EMOJIS = ["👿", "🤢", "😱", "😊", "😐", "😔", "😲"]

# ================================
# 📌 Load model once per thread
# ================================
model = None
model_lock = threading.Lock()

def load_model_once(model_path):
    global model
    with model_lock:
        if model is None:
            model = load_model(model_path)

# ================================
# 🎯 Image Preprocessing
# ================================
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    img = cv2.resize(img, (48, 48))
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    sharpening_kernel = np.array([[0, -1, 0],
                                  [-1, 5, -1],
                                  [0, -1, 0]])
    img = cv2.filter2D(img, -1, sharpening_kernel)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# ================================
# 🔍 Emotion Prediction
# ================================
def predict_emotion(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img, verbose=0)
    emotion_index = np.argmax(prediction)
    return CLASS_LABELS[emotion_index], CLASS_LABELS_EMOJIS[emotion_index]

# ================================
# ⚡ Worker Function
# ================================
def worker(task_queue, results, counter_dict, lock, model_path, image_count_by_thread):
    thread_name = threading.current_thread().name
    local_model = load_model(model_path)

    global model
    with model_lock:
        model = local_model  # Set model for predict_emotion()

    while True:
        image_path = task_queue.get()
        if image_path is None:
            break

        start = time.time()
        emotion, emoji = predict_emotion(image_path)
        duration = time.time() - start

        with lock:
            folder = os.path.basename(os.path.dirname(image_path))
            if folder not in counter_dict:
                counter_dict[folder] = Counter()
            counter_dict[folder][emotion] += 1
            results.append((image_path, emotion, emoji))
            image_count_by_thread[thread_name] += 1

        print(f"[{thread_name}] finished {os.path.basename(image_path)} in {duration:.2f} sec")

# ================================
# 🚀 Main Execution
# ================================
def predict_emotions_threaded(parent_folder):
    start_time = time.time()
    all_image_paths = []

    # Gather all image paths
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for root, _, files in os.walk(subfolder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_image_paths.append(os.path.join(root, file))

    # Shared resources
    image_queue = Queue()
    for path in all_image_paths:
        image_queue.put(path)

    num_threads = min(os.cpu_count() - 1, len(all_image_paths))
    results = []
    emotion_counts = {}
    lock = threading.Lock()
    model_path = './cnn_emotion_detection.h5'
    image_count_by_thread = defaultdict(int)

    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(image_queue, results, emotion_counts, lock, model_path, image_count_by_thread))
        t.start()
        threads.append(t)

    for _ in threads:
        image_queue.put(None)

    for t in threads:
        t.join()

    total_time = time.time() - start_time
    print(f"\n✅ Total execution time: {total_time:.2f} sec ({int(total_time // 60)} min)")

    print("\n📊 Images processed per thread:")
    for thread_name, count in image_count_by_thread.items():
        print(f"  {thread_name}: {count} images")

    print("\n📁 Emotion counts per folder:")
    for folder, counter in emotion_counts.items():
        print(f"\n{folder}:")
        for emotion, count in counter.items():
            print(f"  {emotion}: {count}")

    return results

# ================================
# 🏁 Main
# ================================
if __name__ == "__main__":
    folder_path = "./test_images"
    results = predict_emotions_threaded(folder_path)

    with open("emotion_results.txt", "w", encoding="utf-8") as f:
        for image_path, label, emoji in results:
            f.write(f"{image_path}: {label} {emoji}\n")

    print("\n📝 Results saved to 'emotion_results.txt'")
