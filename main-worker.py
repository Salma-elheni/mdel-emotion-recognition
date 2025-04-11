import tensorflow as tf  # Pour utiliser le modèle de deep learning
from tensorflow.keras.models import load_model  # Pour charger un modèle entraîné
import numpy as np  # Pour les opérations sur les tableaux
import cv2  # Pour le traitement d’image
import os  # Pour parcourir les fichiers et dossiers
import time  # Pour mesurer le temps d'exécution
import threading  # Pour le multithreading
from queue import Queue  # File de tâches thread-safe
from collections import Counter, defaultdict  # Pour compter les émotions

# Liste des émotions possibles que le modèle peut prédire
CLASS_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
# Emoji associé à chaque émotion
CLASS_LABELS_EMOJIS = ["👿", "🤢", "😱", "😊", "😐", "😔", "😲"]

# Variable globale pour stocker le modèle et un verrou pour éviter les conflits entre threads
model = None
model_lock = threading.Lock()

# Fonction pour charger le modèle une seule fois (protégée par un verrou)
def load_model_once(model_path):
    global model
    with model_lock:
        if model is None:
            model = load_model(model_path)

# Fonction de prétraitement d'image avant de la passer au modèle
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_REDUCED_GRAYSCALE_2)  # Lecture de l'image en niveaux de gris
    img = cv2.resize(img, (48, 48))  # Redimensionnement à 48x48 pixels (taille attendue par le modèle)
    img = cv2.equalizeHist(img)  # améliorer le contraste
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

# Prédiction de l’émotion pour une image donnée
def predict_emotion(image_path):
    img = preprocess_image(image_path)  # Prétraitement
    prediction = model.predict(img, verbose=0)  # Prédiction avec le modèle
    emotion_index = np.argmax(prediction)  # Index de l’émotion prédite
    return CLASS_LABELS[emotion_index], CLASS_LABELS_EMOJIS[emotion_index]  # Émotion + emoji associé

# Fonction exécutée par chaque thread pour traiter les images
def worker(task_queue, results, counter_dict, lock, model_path, image_count_by_thread):
    thread_name = threading.current_thread().name  # Nom du thread en cours
    local_model = load_model(model_path)  # Chaque thread charge sa propre copie du modèle

    global model
    with model_lock:
        model = local_model  # Définir le modèle global utilisé dans predict_emotion()

    while True:
        image_path = task_queue.get()  # Récupère une image à traiter
        if image_path is None:
            break  # Fin du traitement pour ce thread

        start = time.time()  # Début du chrono
        emotion, emoji = predict_emotion(image_path)  # Prédiction
        duration = time.time() - start  # Temps pris pour traiter l'image

        with lock:  # Accès aux structures partagées
            folder = os.path.basename(os.path.dirname(image_path))  # Nom du dossier de l'image
            if folder not in counter_dict:
                counter_dict[folder] = Counter()
            counter_dict[folder][emotion] += 1  # Incrémentation du compteur d’émotions
            results.append((image_path, emotion, emoji))  # Stockage du résultat
            image_count_by_thread[thread_name] += 1  # Compteur d’images traitées par ce thread

        print(f"[{thread_name}] finished {os.path.basename(image_path)} in {duration:.2f} sec")

# Fonction principale qui gère le traitement multithreadé
def predict_emotions_threaded(parent_folder):
    start_time = time.time()  # Début du chrono global
    all_image_paths = []  # Liste de toutes les images

    # Parcours des sous-dossiers pour récupérer les chemins des images
    for subfolder in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for root, _, files in os.walk(subfolder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_image_paths.append(os.path.join(root, file))

    image_queue = Queue()  # File de tâches
    for path in all_image_paths:
        image_queue.put(path)  # Ajout de chaque image à la file

    num_threads = min(os.cpu_count() - 1, len(all_image_paths))  # Choix du nombre optimal de threads

    results = []  # Stockage des résultats (image, label, emoji)
    emotion_counts = {}  # Compteur d’émotions par dossier
    lock = threading.Lock()  # Verrou partagé pour synchroniser les accès
    model_path = './cnn_emotion_detection.h5'  # Chemin vers le modèle
    image_count_by_thread = defaultdict(int)  # Nombre d'images traitées par thread

    threads = []  # Liste des threads
    for _ in range(num_threads):
        t = threading.Thread(
            target=worker,
            args=(image_queue, results, emotion_counts, lock, model_path, image_count_by_thread)
        )
        t.start()
        threads.append(t)

    for _ in threads:
        image_queue.put(None)  # Envoi du signal de fin à chaque thread

    for t in threads:
        t.join()  # Attente de la fin de tous les threads

    total_time = time.time() - start_time  # Temps total de traitement
    print(f"\n✅ Total execution time: {total_time:.2f} sec ({int(total_time // 60)} min)")

    print("\n📊 Images processed per thread:")
    for thread_name, count in image_count_by_thread.items():
        print(f"  {thread_name}: {count} images")

    print("\n📁 Emotion counts per folder:")
    for folder, counter in emotion_counts.items():
        print(f"\n{folder}:")
        for emotion, count in counter.items():
            print(f"  {emotion}: {count}")

    return results  # Retourne tous les résultats

# Point d’entrée du programme
if __name__ == "__main__":
    folder_path = "./test_images"  # Chemin vers le dossier à analyser
    results = predict_emotions_threaded(folder_path)  # Lancement du traitement

    # Sauvegarde des résultats dans un fichier texte
    with open("emotion_results.txt", "w", encoding="utf-8") as f:
        for image_path, label, emoji in results:
            f.write(f"{image_path}: {label} {emoji}\n")

    print("\n📝 Results saved to 'emotion_results.txt'")
