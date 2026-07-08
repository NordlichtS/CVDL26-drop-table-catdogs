import os
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from PIL import Image

class AnimalClassifier(nn.Module):
    def __init__(self, weights_path=None, num_classes=10, device='cpu'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        
        # --- 1. ARCHITEKTUR (IMMER FROM SCRATCH) ---
        # weights=None garantiert, dass NIEMALS vortrainierte ImageNet-Gewichte geladen werden!
        self.backbone = efficientnet_v2_s(weights=None)
        
        # Den letzten Klassifikations-Layer auf deine 10 Klassen anpassen
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
        # --- 2. GEWICHTE LADEN (Nur wenn die Datei im Hauptordner existiert) ---
        if weights_path and os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)
            # Kompatibilität mit dem traineffnet-Checkpoint-Format herstellen
            if isinstance(checkpoint, dict) and "model_state" in checkpoint:
                self.load_state_dict(checkpoint["model_state"])
            else:
                self.load_state_dict(checkpoint)
            print(f"[INFO] Eigene Gewichte erfolgreich geladen von: {os.path.abspath(weights_path)}")
        elif weights_path:
            print(f"[WARNUNG] '{weights_path}' nicht gefunden! Modell startet komplett von Null (Zufallswerte).")
        else:
            print("[INFO] Kein Gewichts-Pfad angegeben. Modell startet komplett von Null (Zufallswerte).")
            
        self.to(self.device)
        self.eval() # Für Inferenz standardmäßig im Evaluierungsmodus
        
        # --- 3. DATEN-TRANSFORMATION FÜR INFERENZ ---
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(int(384 * 1.14)),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, crop_np):
        """
        Nimmt das Bild entgegen, transformiert es auf 384x384 
        und wirft die Logits für die 10 Klassen zurück.
        """
        if isinstance(crop_np, Image.Image):
            input_tensor = self.transform(crop_np).unsqueeze(0).to(self.device)
        else:
            input_tensor = self.transform(crop_np).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.backbone(input_tensor)
        return output