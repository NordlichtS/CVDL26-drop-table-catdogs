"""Evaluation harness for the Fine-grained Animal Recognition project.

We run this script on the held-out test set, so do not change the interface.
Implement your solution as the `Model` below: an `nn.Module` whose `forward`
takes a PIL image and returns a predicted class index, an integer in
{-1, 0, ..., 19}, where -1 means "reject", i.e. no target species is present.
Inside `forward` you are free to do anything you like: run an off-the-shelf
detector, find bounding boxes, crop the largest animal, classify the crop,
decide when to return -1, and so on.

The script reads `labels.csv` from the image folder, with columns
`filename,label`, where `label` is the integer class index from CLASSES (or -1
for confounders / images with no target species). The images themselves are a
flat, numbered set (0001.jpg, 0002.jpg, ...) sitting next to `labels.csv`. The
script runs your model on every image and prints the standard classification
metrics.
python inference.py --image-folder <folder> """


import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TORCH_HUB_TRUST_REPO"] = "1"
import subprocess
import argparse
import random
import numpy as np
import time
from pathlib import Path

import pandas as pd

import torch
torch.serialization.add_safe_globals(["yolov6.models.yolo.Model"])
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

from Final_Project.detector import AnimalDetector
from Final_Project.animalClassifier import AnimalClassifier
#from Final_Project.compare_classifier import CompareClassifier


REJECT = -1

# Official class mapping fixed by the chair (index -> species). Train your
# classifier against this exact order so your labels match our evaluation.
CLASSES = [
    "Abyssinian",         #  0
    "Bengal",             #  1
    "Birman",             #  2
    "Bombay",             #  3
    "British_Shorthair",  #  4
    "Maine_Coon",         #  5
    "Ragdoll",            #  6
    "Sphynx",             #  7
    "Tabby",              #  8
    "Tiger_Cat",          #  9
    "Beagle",             # 10
    "Pug",                # 11
    "Boxer",              # 12
    "Shiba_Inu",          # 13
    "Samoyed",            # 14
    "Golden_Retriever",   # 15
    "German_Shepherd",    # 16
    "Siberian_Husky",     # 17
    "Dalmatian",          # 18
    "Rottweiler",         # 19
]
NUM_CLASSES = len(CLASSES)


class Model(nn.Module):
    """TODO (students): replace this with your own model.

    Contract: given a PIL image, return a class index in {-1, 0, ..., 19}.
    The placeholder below is a uniform random guesser so the script runs.
    """
    def __init__(self):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        weights_file = "yolov6s.pt"
        parent_dir_weights = os.path.join("..", "yolov6s.pt")
        
        # 2. Logik: Wenn sie nirgendwo gefunden wird, erst DANN downloaden
        if os.path.exists(weights_file):
            final_weights_path = weights_file
            print(f"[INFO] Modell '{weights_file}' lokal gefunden.")
        elif os.path.exists(parent_dir_weights):
            final_weights_path = parent_dir_weights
            print(f"[INFO] Modell '{parent_dir_weights}' lokal gefunden.")
        else:
            print(f"[INFO] Modell nicht gefunden. Starte Download...")
            download_script = os.path.join("..", "Final_Project/download_model_detector.py")
            try:
                subprocess.run(["python3", download_script], check=True)
                final_weights_path = weights_file
            except subprocess.CalledProcessError as e:
                print(f"[FEHLER] Download fehlgeschlagen: {e}")
                raise e

        # 3. Jetzt den Detektor mit dem gefundenen Pfad initialisieren
        self.animal_detector = AnimalDetector(weights_path=final_weights_path)

        #TODO: Implement the trained Race-Classifier
        print(os.getcwd())
        current_dir = os.getcwd()
        cat_path = os.path.join(current_dir, "cat_scratch.pth")
        dog_path = os.path.join(current_dir, "dog_scratch.pth")
        if not os.path.exists(cat_path) and not os.path.exists(dog_path):
            print(f"[FEHLER] Datei nicht gefunden unter: {os.path.abspath(cat_path)}")
        self.cat_classifier = AnimalClassifier(weights_path=cat_path, device=self.device)
        self.dog_classifier = AnimalClassifier(weights_path=dog_path, device=self.device)
#        self.compare_model = CompareClassifier(device=self.device)

    def forward(self, image: Image.Image) -> int:
        import time
        t_start = time.perf_counter()
        image_np = np.array(image)
        img_bgr = image_np[:, :, ::-1].copy()
        t_prep_end = time.perf_counter()
        prep_time = t_prep_end - t_start
        resized_crop_np, species, meta = self.animal_detector.detect_largest_animal(img_bgr)
        t_yolo_end = time.perf_counter()
        yolo_time = t_yolo_end - t_prep_end
        # REJECT-FALL: Wenn YOLO kein Tier findet
        if meta is None or species == "neither":
            print(f"[REJECT] Kein Zieltier gefunden (YOLO sagt 'neither' oder Fehler). "
                  f"Zeit: [Prep: {prep_time:.4f}s | YOLO: {yolo_time:.4f}s]")
            return REJECT

        t_class_start = time.perf_counter()

        if species == "cat":
            output = self.cat_classifier(resized_crop_np)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, local_idx = torch.max(probs, dim=1)
            
            t_class_end = time.perf_counter()
            class_time = t_class_end - t_class_start
            total_time = t_class_end - t_start
            
            print(f"[SCRATCH] Vorschlag (Katze): {CLASSES[local_idx.item()]} (Sicherheit: {confidence.item():.2%})")
            print(f"[PROFILING] Prep: {prep_time:.4f}s | YOLO: {yolo_time:.4f}s | Classifier: {class_time:.4f}s | Gesamt: {total_time:.4f}s")
            
            return local_idx.item()

        elif species == "dog":
            output = self.dog_classifier(resized_crop_np)
            probs = torch.nn.functional.softmax(output, dim=1)
            confidence, local_idx = torch.max(probs, dim=1)
            
            t_class_end = time.perf_counter()
            class_time = t_class_end - t_class_start
            total_time = t_class_end - t_start
            
            # Index + 10 für Hunde
            print(f"[SCRATCH] Vorschlag (Hund): {CLASSES[local_idx.item() + 10]} (Sicherheit: {confidence.item():.2%})")
            print(f"[PROFILING] Prep: {prep_time:.4f}s | YOLO: {yolo_time:.4f}s | Classifier: {class_time:.4f}s | Gesamt: {total_time:.4f}s")
            
            return local_idx.item() + 10

        # FALLBACK-FALL: Wenn der Detektor eine unerwartete Spezies liefert
        print(f"[REJECT / FALLBACK] Unbekannte Spezies erkannt: '{species}'. "
              f"Zeit: [Prep: {prep_time:.4f}s | YOLO: {yolo_time:.4f}s]")
        return random.randint(-1, NUM_CLASSES - 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-folder", type=Path, default=Path(__file__).resolve().parent / "images")
    args = parser.parse_args()

    df = pd.read_csv(args.image_folder / "labels.csv")
    df = df.sample(n=1)
    print(f"\n--- zufälliges Bild: {df['filename'].values[0]} ---")
    model = Model().eval()

    y_true, y_pred = [], []
    
    start_inf = time.time()

    with torch.no_grad():
        for filename, label in tqdm(zip(df["filename"], df["label"]), total=len(df)):
            image = Image.open(args.image_folder / filename).convert("RGB")
            pred = model(image)
            y_true.append(int(label))
            y_pred.append(int(pred))

    end_inf = time.time()
    print(f"Durchschnittliche Inferenzzeit pro Bild: {(end_inf - start_inf) / len(df):.4f} Sekunden")
    labels = [REJECT] + list(range(NUM_CLASSES))
    target_names = ["reject(-1)"] + CLASSES
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=target_names, digits=3,
                                zero_division=0))
    #print("Confusion matrix (rows=true, cols=pred):")
    #print(confusion_matrix(y_true, y_pred, labels=labels))