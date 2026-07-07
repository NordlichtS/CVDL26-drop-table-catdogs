import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from Final_Project.detector import AnimalDetector
from animalClassifier import AnimalClassifier

class CroppedAnimalDataset(Dataset):
    def __init__(self, csv_file, img_dir, species, detector, transform=None):
        self.img_dir = img_dir
        self.species = species
        self.detector = detector
        self.transform = transform
        
        df = pd.read_csv(csv_file)
        if self.species == "cat":
            self.data = df[df['label'].between(0, 9)].reset_index(drop=True)
        elif self.species == "dog":
            self.data = df[df['label'].between(10, 19)].reset_index(drop=True)
            
        self.cache_dir = os.path.join(img_dir, f"cropped_cache_{species}")
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        original_label = int(self.data.iloc[idx, 1])
        
        label = original_label if self.species == "cat" else original_label - 10
        
        orig_img_path = os.path.join(self.img_dir, img_name)
        cached_img_path = os.path.join(self.cache_dir, img_name)
        
        if os.path.exists(cached_img_path):
            crop_image = Image.open(cached_img_path).convert("RGB")
        else:
            try:
                with torch.no_grad():
                    crop_np, detected_species, meta = self.detector.detect_largest_animal(orig_img_path)
                    
                    if isinstance(crop_np, torch.Tensor):
                        crop_np = crop_np.cpu().detach().numpy()
                    # Wenn es ein Numpy-Array ist, aber noch auf der GPU-Logik basiert:
                    elif hasattr(crop_np, 'cpu'):
                        crop_np = crop_np.cpu().numpy()
                    
                    crop_image = Image.fromarray(crop_np)
                    crop_image.save(cached_img_path)
            except Exception as e:
                print(f"[WARNUNG] Fehler beim Croppen von {img_name}: {e}")
                crop_image = Image.open(orig_img_path).convert("RGB")
        
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
        
        progress_bar = tqdm(dataloader, desc=f"Epoche {epoch+1}/{num_epochs}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # FEHLER BEHOBEN: Modell direkt aufrufen, nicht das backbone!
            outputs = model.backbone(inputs) 
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=loss.item(), acc=100.*correct/total)
            
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        print(f"[{model_name}] Epoche {epoch+1} abgeschlossen | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    save_path = f"{model_name}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"[ERFOLG] Gewichte für {model_name} in '{save_path}' gespeichert!\n")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training läuft auf: {device}")
    
    batch_size = 5
    num_epochs = 1
    learning_rate = 0.001
    
    img_dir = "images" 
    csv_file = os.path.join(img_dir, "labels.csv")
    
    detector = AnimalDetector(weights_path='yolov6s.pt', device=device)
    
    # FEHLER BEHOBEN: Resize hinzugefügt, damit alle Tektoren dieselbe Form besitzen
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. KATZEN-MODELL
    cat_dataset = CroppedAnimalDataset(csv_file, img_dir, species="cat", detector=detector, transform=train_transform)
    # FEHLER BEHOBEN: num_workers=0, um YOLO-Multiprocessing-Abstürze zu verhindern
    cat_loader = DataLoader(cat_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    cat_weights = "cat_scratch.pth"
    cat_model = AnimalClassifier(
        weights_path=cat_weights if os.path.exists(cat_weights) else None, 
        num_classes=10, 
        device=device
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # Achte darauf, ob du nur das Backbone oder das gesamte cat_model optimieren willst!
    optimizer_cat = optim.Adam(cat_model.parameters(), lr=learning_rate)
    
    train_model(cat_model, cat_loader, criterion, optimizer_cat, num_epochs, device, "cat_scratch")
    

    # 2. HUNDE-MODELL
    dog_dataset = CroppedAnimalDataset(csv_file, img_dir, species="dog", detector=detector, transform=train_transform)
    dog_loader = DataLoader(dog_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    dog_weights = "dog_scratch.pth"
    dog_model = AnimalClassifier(
        weights_path=dog_weights if os.path.exists(dog_weights) else None, 
        num_classes=10, 
        device=device
    ).to(device)
    
    optimizer_dog = optim.Adam(dog_model.parameters(), lr=learning_rate)
    
    train_model(dog_model, dog_loader, criterion, optimizer_dog, num_epochs, device, "dog_scratch")
    
    print("ALL DONE!")