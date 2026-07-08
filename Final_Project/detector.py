"""This file contains the Animal detector which we use to 
a) search for animals in a given picture
b) crop out the biggest animal and forward it in the size 224x224
c) already predefine the animal type"""

import os
import torch
torch.serialization.add_safe_globals(["yolov6.models.yolo.Model"])
import cv2
import numpy as np

class AnimalDetector:
    def __init__(self, weights_path='yolov6s.pt', device=None):
            """
            Initialisiert den YOLOv6 Detektor. Lädt das Repo beim ersten Start
            automatisch in den torch.hub Cache (Internet nur beim ersten Mal nötig).
            """
            self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

            # PyTorch 2.6+ Patch: Verhindert den restriktiven Modus beim Laden
            original_load = torch.load
            def custom_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = custom_load

            try:
                self.model = torch.hub.load(
                    'meituan/YOLOv6', 'custom',
                    ckpt_path=weights_path, class_names=[],
                    source='github',            # <- статт локального кешу
                    trust_repo=True,            # без інтерактивного prompt'а
                )
            finally:
                torch.load = original_load

            self.model.to(self.device).eval()

            # COCO Klassen-IDs für Katze (15) und Hund (16)
            self.CAT_CLASS_ID = 15
            self.DOG_CLASS_ID = 16

    
    def detect_largest_animal(self, image_input):
        """
        image_input: Entweder ein Dateipfad (String) ODER direkt ein numpy array (BGR).
        """
        # Wenn ein Pfad übergeben wird (z.B. fürs Training), lade es
        if isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
        # Wenn direkt ein Array übergeben wird (Inferenz), nutze es direkt
        else:
            img_bgr = image_input
            
        if img_bgr is None:
            raise ValueError("Bild ist leer oder konnte nicht geladen werden!")

        h_orig, w_orig, _ = img_bgr.shape
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- YOLO PREPROCESSING (Wieder aktiv für den Detector Wrapper) ---
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_chw = img_resized.transpose((2, 0, 1))
        img_tensor = torch.from_numpy(img_chw).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inferenz MIT src_shape aufrufen, damit die Signatur stimmt
        raw_output = self.model(img_tensor, src_shape=(h_orig, w_orig))
        
        # --- KORREKTUR: SAUBERES EXTRAHIEREN AUS DEM DETECTOR-OUTPUT ---
        # Der YOLOv6 Detector gibt oft eine Liste/Tuple zurück. 
        # Der ERSTE Eintrag enthält die numerischen Bounding Boxes.
        if isinstance(raw_output, dict):
            # YOLOv6 gibt oft ein Dict zurück: {'boxes': ..., 'scores': ..., 'labels': ...}
            boxes = raw_output.get('boxes', [])
            scores = raw_output.get('scores', [])
            labels = raw_output.get('labels', [])
            
            # Kombiniere diese zu einer Liste, die dein restlicher Code versteht
            # Wir bauen uns eine Liste, wo jede Zeile [x1, y1, x2, y2, conf, label] ist
            predictions = []
            for i in range(len(boxes)):
                box = boxes[i]
                predictions.append([box[0], box[1], box[2], box[3], scores[i], labels[i]])
        else:
            # Falls es doch ein Array ist, lass es so wie es war
            predictions = raw_output
            
        # Jetzt wie gewohnt zu numpy konvertieren
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().detach().numpy()
        
        else:
            predictions = np.array(predictions)

        # Fallback, falls absolut nichts im Bild gefunden wurde
        if predictions is None or len(predictions) == 0:
            resized_crop = cv2.resize(img_rgb, (224, 224))
            return resized_crop, "neither", None

        largest_area = 0
        best_box = None
        species = "neither"

        for pred in predictions:
            # Sicherheitscheck gegen Metadaten/Zeilenüberschriften
            if isinstance(pred, (str, bytes)) or len(pred) < 5:
                continue
                
            # Flexibles Entpacken, falls YOLOv6 nur 5 statt 6 Werte liefert
            if len(pred) >= 6:
                x1, y1, x2, y2, conf, class_id = pred[:6]
            else:
                x1, y1, x2, y2, conf = pred[:5]
                class_id = self.CAT_CLASS_ID

            # Erst hier zu Float konvertieren, falls alles okay ist
            try:
                x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                class_id = int(float(class_id))
            except (ValueError, TypeError):
                continue  # Falls doch mal ein 'b' durchrutscht, ignorieren

            if class_id in [self.CAT_CLASS_ID, self.DOG_CLASS_ID] or len(pred) == 5:
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    best_box = (int(x1), int(y1), int(x2), int(y2))
                    species = "dog" if class_id == self.DOG_CLASS_ID else "cat"
        
#        # zero-out masking of overlapping animals
#        if best_box is not None:
#            for pred in predictions:
#                if isinstance(pred, (str, bytes)) or len(pred) < 4:
#                    continue
#                try:
#                    x1, y1, x2, y2 = pred[:4]
#                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
#                    current_box = (int(x1), int(y1), int(x2), int(y2))
#                    
#                    if current_box != best_box:
#                        cx1, cy1 = max(0, current_box[0]), max(0, current_box[1])
#                        cx2, cy2 = min(w_orig, current_box[2]), min(h_orig, current_box[3])
#                        img_rgb[cy1:cy2, cx1:cx2] = [0, 0, 0]
#                except (ValueError, TypeError):
#                    continue
                        
        if best_box is None:
            resized_crop = cv2.resize(img_rgb, (224, 224))
            # Test ausgabe um das Bild zu überprüfen
            crop_bgr = cv2.cvtColor(resized_crop, cv2.COLOR_RGB2BGR)
            return resized_crop, "neither", None

        x1, y1, x2, y2 = best_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_orig, x2), min(h_orig, y2)
        
        crop = img_rgb[y1:y2, x1:x2]
        resized_crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        
        # Das Bild von RGB zurück nach BGR konvertieren, damit OpenCV es richtig speichert
        crop_bgr = cv2.cvtColor(resized_crop, cv2.COLOR_RGB2BGR)
        return resized_crop, species, {"box": (x1, y1, x2, y2), "orig_dim": (w_orig, h_orig)}