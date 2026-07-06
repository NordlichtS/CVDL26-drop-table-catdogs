import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

# Eure eigenen Module importieren
from Final_Project.detector import AnimalDetector
from animalClassifier import AnimalClassifier

class CroppedAnimalDataset(Dataset):
    def __init__(self, csv_file, img_dir, species, detector, transform=None):
        self.img_dir = img_dir
        self.species = species
        self.detector = detector
        self.transform = transform
        
        # Labels laden und filtern
        df = pd.read_csv(csv_file)
        if self.species == "cat":
            # Katzen haben die Labels 0 bis 9
            self.data = df[df['label'].between(0, 9)].reset_index(drop=True)
        elif self.species == "dog":
            # Hunde haben die Labels 10 bis 19
            self.data = df[df['label'].between(10, 19)].reset_index(drop=True)
            
        # Cache-Ordner für die gecroppten Bilder erstellen
        self.cache_dir = os.path.join(img_dir, f"cropped_cache_{species}")
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        original_label = int(self.data.iloc[idx, 1])
        
        # Label auf 0-9 mappen (für Hunde: 10-19 wird zu 0-9)
        label = original_label if self.species == "cat" else original_label - 10
        
        orig_img_path = os.path.join(self.img_dir, img_name)
        cached_img_path = os.path.join(self.cache_dir, img_name)
        
        # CACHING-LOGIK: Prüfen, ob das Bild schon von YOLO ausgeschnitten wurde
        if os.path.exists(cached_img_path):
            # Lade den fertigen Crop direkt von der Festplatte
            crop_image = Image.open(cached_img_path).convert("RGB")
        else:
            # YOLO den Crop machen lassen (nur beim allerersten Mal nötig!)
            try:
                crop_np, detected_species, meta = self.detector.detect_largest_animal(orig_img_path)
                
                # NumPy Array zu PIL Image konvertieren für torchvision transforms
                crop_image = Image.fromarray(crop_np)
                
                # In den Cache speichern für die nächste Epoche
                crop_image.save(cached_img_path)
            except Exception as e:
                # Fallback, falls YOLO bei einem Bild crasht (sollte nicht passieren)
                print(f"[WARNUNG] Fehler beim Croppen von {img_name}: {e}")
                crop_image = Image.open(orig_img_path).convert("RGB").resize((224, 224))
        
        # Transformationen anwenden (die aus animalClassifier.py)
        if self.transform:
            crop_tensor = self.transform(crop_image)
            
        return crop_tensor, label

# --- 2. TRAININGS-FUNKTION ---
def train_model(model, dataloader, criterion, optimizer, num_epochs, device, model_name):
    model.train()
    print(f"\n--- Starte Training für {model_name} ({num_epochs} Epochen) ---")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # TQDM für eine schöne Fortschrittsanzeige
        progress_bar = tqdm(dataloader, desc=f"Epoche {epoch+1}/{num_epochs}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad() # Gradienten zurücksetzen
            
            # Forward Pass
            outputs = model.backbone(inputs) # Wir rufen den Backbone direkt auf (für Gradienten)
            loss = criterion(outputs, labels)
            
            # Backward Pass (Backpropagation)
            loss.backward()
            optimizer.step()
            
            # Statistiken
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        print(f"[{model_name}] Epoche {epoch+1} abgeschlossen | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # Gewichte speichern
    save_path = f"{model_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[ERFOLG] Gewichte für {model_name} in '{save_path}' gespeichert!\n")



# --- 3. HAUPTPROGRAMM (MAIN) ---
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training läuft auf: {device}")
    
    # HYPERPARAMETER
    batch_size = 32
    num_epochs = 10 # Kannst du später auf 30-50 erhöhen
    learning_rate = 0.001
    
    # Pfade (Passe diese an deinen Projektordner an)
    img_dir = "Final_Project/training_images" 
    csv_file = os.path.join(img_dir, "labels.csv")
    print(f"[DEBUG] Ich suche die Datei unter: {os.path.abspath(csv_file)}")
    
    # Detektor initialisieren (für das Pre-Cropping)
    detector = AnimalDetector(weights_path='yolov6s.pt', device=device)
    
    # Transformation (Die gleiche wie in animalClassifier.py)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# 1. KATZEN-MODELL TRAINIEREN
    cat_dataset = CroppedAnimalDataset(csv_file, img_dir, species="cat", detector=detector, transform=train_transform)
    cat_loader = DataLoader(cat_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Trainieren, schon bestehende Gewichte übernehmen und darauf weiter trainieren
    print(f"Info: Path Training Model: {os.getcwd()}")
    cat_weights = "cat_scratch.pth"
    
    cat_model = AnimalClassifier(
        weights_path=cat_weights if os.path.exists(cat_weights) else None, 
        num_classes=10, 
        device=device
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_cat = optim.Adam(cat_model.backbone.parameters(), lr=learning_rate)
    
    train_model(cat_model, cat_loader, criterion, optimizer_cat, num_epochs, device, "cat_scratch")
    

    # 2. HUNDE-MODELL TRAINIEREN
    dog_dataset = CroppedAnimalDataset(csv_file, img_dir, species="dog", detector=detector, transform=train_transform)
    dog_loader = DataLoader(dog_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    dog_weights = "dog_scratch.pth"
    dog_model = AnimalClassifier(
        weights_path=dog_weights if os.path.exists(dog_weights) else None, 
        num_classes=10, 
        device=device
    ).to(device)
    
    optimizer_dog = optim.Adam(dog_model.backbone.parameters(), lr=learning_rate)
    
    train_model(dog_model, dog_loader, criterion, optimizer_dog, num_epochs, device, "dog_scratch")
    
    print("ALL DONE! Deine Modelle sind trainiert und bereit für die inference.py!")