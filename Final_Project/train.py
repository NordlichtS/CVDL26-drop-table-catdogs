import os
import time
import random
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from Final_Project.detector import AnimalDetector
from animalClassifier import AnimalClassifier

CAT_CLASSES = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat"]
DOG_CLASSES = ["Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed", "Golden_Retriever", "German_Shepherd", "Siberian_Husky", "Dalmatian", "Rottweiler"]
NUM_CLASSES = 10

class CroppedAnimalDataset(Dataset):
    def __init__(self, csv_file, img_dir, species, detector, transform=None, classes_dir=None):
        self.img_dir = img_dir
        self.species = species
        self.detector = detector
        self.transform = transform
        self.classes_dir = classes_dir  # Neu: Pfad zum "classes"-Ordner des Freundes
        
        self.cache_dir = os.path.join(img_dir, f"cropped_cache_{species}")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Modus 1: Ordnerstruktur von Ivan nutzen (wenn classes_dir existiert)
        if self.classes_dir and os.path.exists(self.classes_dir):
            rows = []
            # Gehe durch alle Katzen- oder Hundeklassen
            target_classes = CAT_CLASSES if species == "cat" else DOG_CLASSES
            all_classes = CAT_CLASSES + DOG_CLASSES # Gesamte Liste für das globale Label (0-19)
            
            for class_name in target_classes:
                class_folder = os.path.join(self.classes_dir, class_name)
                if os.path.exists(class_folder):
                    global_label = all_classes.index(class_name) # Bestimmt das originale Label 0-19
                    for img_name in os.listdir(class_folder):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            # Wir speichern den relativen Pfad ab dem Klassenordner
                            rows.append({'filename': os.path.join(class_name, img_name), 'label': global_label})
            
            self.data = pd.DataFrame(rows)
            self.is_folder_mode = True

        # Modus 2: Deine eigene labels.csv nutzen (Fallback)
        else:
            df = pd.read_csv(csv_file)
            if self.species == "cat":
                self.data = df[df['label'].between(0, 9)].reset_index(drop=True)
            elif self.species == "dog":
                self.data = df[df['label'].between(10, 19)].reset_index(drop=True)
            self.is_folder_mode = False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        original_label = int(self.data.iloc[idx, 1])
        
        label = original_label if self.species == "cat" else original_label - 10
        
        # Pfad-Unterscheidung je nach Daten-Modus
        if getattr(self, 'is_folder_mode', False):
            orig_img_path = os.path.join(self.classes_dir, img_name)
            # Für den Cache ersetzen wir den Slash, damit es im flachen Cache-Ordner landet
            cached_name = img_name.replace(os.sep, "_")
        else:
            orig_img_path = os.path.join(self.img_dir, img_name)
            cached_name = img_name
            
        cached_img_path = os.path.join(self.cache_dir, cached_name)
        
        if os.path.exists(cached_img_path):
            crop_image = Image.open(cached_img_path).convert("RGB")
        else:
            try:
                with torch.no_grad():
                    crop_np, detected_species, meta = self.detector.detect_largest_animal(orig_img_path)
                    if isinstance(crop_np, torch.Tensor):
                        crop_np = crop_np.cpu().detach().numpy()
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

def macro_f1_and_cm(preds, labels):
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for p, t in zip(preds, labels):
        cm[t, p] += 1
    f1s = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)), f1s

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs, device, species_name):
    print(f"\n>>> Starte FROM-SCRATCH Training für {species_name.upper()} ({num_epochs} Epochen) <<<")
    best_f1 = 0.0
    save_path = f"{species_name}_scratch.pth" # Speichert direkt unter cat_scratch.pth / dog_scratch.pth
    
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]
        
        # --- TRAINING ---
        model.train()
        running_loss, correct, total, n_batches = 0.0, 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"{species_name.capitalize()} Epoche {epoch}/{num_epochs}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=scaler is not None):
                outputs = model.backbone(inputs)
                loss = criterion(outputs, labels)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            n_batches += 1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix(loss=f"{running_loss / n_batches:.3f}", acc=f"{100.*correct/total:.2f}%")
            
        train_loss = running_loss / n_batches
        train_acc = correct / total
        scheduler.step()
        
        # --- EVALUIERUNG ---
        model.eval()
        all_preds, all_labels = [], []
        val_loss, n_val_batches = 0.0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels_d = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=scaler is not None):
                    outputs = model.backbone(inputs)
                    val_loss += criterion(outputs, labels_d).item()
                
                n_val_batches += 1
                all_preds.extend(outputs.argmax(1).cpu().tolist())
                all_labels.extend(labels.tolist())
                
        val_loss /= n_val_batches
        val_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
        macro_f1, _ = macro_f1_and_cm(all_preds, all_labels)
        dt = time.time() - t0
        
        print(f"  [{species_name.upper()} EP {epoch}] LR: {cur_lr:.2e} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc * 100:.2f}% | Macro F1: {macro_f1:.4f} ({dt:.0f}s)")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({
                "model_state": model.state_dict(),
                "species": species_name,
                "epoch": epoch,
                "macro_f1": macro_f1
            }, save_path)
            print(f"  >> Neues bestes Modell überschrieben -> '{save_path}'")
    print(f">>>> {species_name.capitalize()}-Training beendet! Beste F1: {best_f1:.4f} <<<<\n")

if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training läuft auf: {device}")
    
    # KORREKTUR AUS TRAINEFFNET FÜR FROM_SCRATCH = TRUE
    batch_size = 16
    num_epochs = 120       # Erhöht von 15 auf 80, da von Scratch gelernt wird

    learning_rate = 1e-3  # Höhere Lernrate für Scratch-Training
    warmup_epochs = 5     # Längerer Warmup für stabile Konvergenz
    img_size = 224 
       
    img_dir = "images" 
    csv_file = os.path.join(img_dir, "labels.csv")
    detector = AnimalDetector(weights_path='yolov6s.pt', device=device)
    
    # Datenaugmentierungen
    train_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for species in ["cat", "dog"]:

        classes_folder_path = "./classes"
        full_dataset = CroppedAnimalDataset(csv_file, img_dir, species=species, detector=detector, transform=train_transform, classes_dir=classes_folder_path)
        
        indices = list(range(len(full_dataset)))
        random.shuffle(indices)
        split = int(0.8 * len(indices))
        train_indices, val_indices = indices[:split], indices[split:]
        
        train_labels = [int(full_dataset.data.iloc[i, 1]) for i in train_indices]
        train_labels_mapped = [l if species == "cat" else l - 10 for l in train_labels]
        counts = Counter(train_labels_mapped)
        wcls = {c: 1.0 / n for c, n in counts.items()}
        sample_weights = [wcls[lbl] for lbl in train_labels_mapped]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        
        train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_indices), batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
        val_loader = DataLoader(torch.utils.data.Subset(CroppedAnimalDataset(csv_file, img_dir, species=species, detector=detector, transform=val_transform), val_indices), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        # PRÜFUNG: Wenn die Datei existiert, lade sie. Wenn nicht, bleibt es bei None (Scratch).
        expected_weights_file = f"{species}_scratch.pth"
        weights_to_pass = expected_weights_file if os.path.exists(expected_weights_file) else None
        
        model = AnimalClassifier(weights_path=weights_to_pass, num_classes=NUM_CLASSES, device=device)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs),
            ],
            milestones=[warmup_epochs],
        )
        scaler = torch.amp.GradScaler() if device == "cuda" else None
        
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs, device, species)

    print("BEIDE SCRATCH-MODELLE BEREIT!")