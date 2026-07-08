import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

class AnimalClassifier(nn.Module):
    def __init__(self, weights_path=None, num_classes=10, device='cpu'):
        super().__init__()
        self.device = device
        
        # --- 1. ARCHITEKTUR (From Scratch) ---
        # Wir laden ResNet18 explizit OHNE vortrainierte ImageNet-Gewichte
        self.backbone = resnet18(weights=None)
        
        # Den letzten Layer anpassen: 
        # Standardmäßig gibt ResNet 1000 Zahlen aus, wir brauchen nur 10 (pro Modell)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        # --- 2. GEWICHTE LADEN ---
        # Sucht nach der .pth Datei. Wenn sie da ist (nach dem Training), wird sie geladen.
# --- 2. GEWICHTE LADEN ---
        # Wir versuchen den Pfad genau so zu laden, wie er übergeben wurde.
        if weights_path and os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            print(f"[INFO] Gewichte erfolgreich geladen von: {os.path.abspath(weights_path)}")
        elif weights_path:
            # Nur wenn der Pfad übergeben wurde, aber nicht existiert, warnen wir.
            print(f"[WARNUNG] Datei nicht gefunden unter: {os.path.abspath(weights_path)}")
            print("[WARNUNG] Modell rechnet aktuell mit Zufallswerten!")
        else:
            print("[INFO] Kein Pfad angegeben. Modell rechnet mit Zufallswerten.")
            
        self.to(self.device)
        self.eval() # Für Inferenz in den Evaluierungs-Modus schalten
        
        # --- 3. DATEN-TRANSFORMATION ---
        # Erwartet das (224, 224, 3) NumPy-Array direkt aus der inference.py / detector.py
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Macht aus dem Array einen Tensor (C, H, W) im Wertebereich 0.0-1.0
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, crop_np):
        """
        Nimmt den ausgeschnittenen Bild-Array (crop_np) entgegen,
        transformiert ihn und wirft die Logits (Scores für jede der 10 Rassen) zurück.
        """
        # 1. Transformieren & Batch-Dimension hinzufügen (1, 3, 224, 224)
        input_tensor = self.transform(crop_np).unsqueeze(0).to(self.device)
        
        # 2. Inferenz
        with torch.no_grad():
            output = self.backbone(input_tensor)
            
        