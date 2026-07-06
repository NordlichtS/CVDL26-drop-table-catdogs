import torch
import torch.nn.functional as F
import cv2
from torchvision import models, transforms

class CompareClassifier:
    def __init__(self, device='cpu'):
        self.device = device
        
        # 1. Vortrainiertes ResNet18 mit den besten (DEFAULT) ImageNet-Gewichten laden
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights).to(self.device)
        self.model.eval() # Direkt in den Evaluierungsmodus setzen
        
        # 2. Die Namen der 1000 ImageNet-Klassen extrahieren (damit wir Text statt IDs bekommen)
        self.class_names = weights.meta["categories"]
        
        # 3. Standard-Transformationen für vortrainierte ImageNet-Modelle
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Erwartet ein NumPy-Array (224, 224, 3)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict_and_save(self, crop_np):
        """
        Nimmt das zugeschnittene Bild (NumPy RGB), speichert es als 'compare_crop.jpg'
        und gibt die vorhergesagte Klasse sowie den Confidence-Score zurück.
        """
        # --- TEIL A: Bild abspeichern ---
        # OpenCV braucht das Bild im BGR-Format zum Speichern
        crop_bgr = cv2.cvtColor(crop_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite("compare_crop.jpg", crop_bgr)
        print("[INFO] Vergleichsbild wurde als 'compare_crop.jpg' gespeichert.")
        
        # --- TEIL B: Vorhersage treffen ---
        # Tensor erstellen und auf die GPU/CPU schieben
        input_tensor = self.transform(crop_np).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
        # Softmax wandelt die rohen Outputs in Wahrscheinlichkeiten (0.0 bis 1.0) um
        probabilities = F.softmax(output, dim=1)[0]
        
        # Den höchsten Wert (Confidence) und seinen Index finden
        top_prob, top_idx = torch.max(probabilities, dim=0)
        
        # In menschliche Formate umwandeln
        predicted_class = self.class_names[top_idx.item()]
        confidence = top_prob.item()
        
        return predicted_class, confidence