import os
import time
import random
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import sys
import gc
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from collections import Counter
from PIL import Image, ImageFilter

# Importe aus deinem Projekt
from Final_Project.detector import AnimalDetector
from animalClassifier import AnimalClassifier

# Konstanten
CAT_CLASSES = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat"]
DOG_CLASSES = ["Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed", "Golden_Retriever", "German_Shepherd", "Siberian_Husky", "Dalmatian", "Rottweiler"]
NUM_CLASSES = 10

class CroppedAnimalDataset(Dataset):
    # NEU: 'cropmix' Argument im Init hinzugefügt
    def __init__(self, csv_file, img_dir, species, detector=None, transform=None, classes_dir=None, indices=None, mirror=False, blur=False, cropmix=False):
        self.img_dir = img_dir
        self.species = species
        self.detector = detector
        self.transform = transform
        self.classes_dir = classes_dir
        
        self.cache_dir = os.path.join(img_dir, f"cropped_cache_{species}")
        os.makedirs(self.cache_dir, exist_ok=True)

        if self.classes_dir and os.path.exists(self.classes_dir):
            rows = []
            target_classes = CAT_CLASSES if species == "cat" else DOG_CLASSES
            all_classes = CAT_CLASSES + DOG_CLASSES
            for class_name in target_classes:
                class_folder = os.path.join(self.classes_dir, class_name)
                if os.path.exists(class_folder):
                    global_label = all_classes.index(class_name)
                    for img_name in os.listdir(class_folder):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            rows.append({'filename': os.path.join(class_name, img_name), 'label': global_label})
            self.data = pd.DataFrame(rows)
            self.is_folder_mode = True
        else:
            df = pd.read_csv(csv_file)
            self.data = df[df['label'].between(0, 9)] if species == "cat" else df[df['label'].between(10, 19)]
            self.is_folder_mode = False

        if indices is not None:
            self.indices_list = indices
        else:
            self.indices_list = list(range(len(self.data)))

        # Pipeline dynamisch aufbauen
        self.samples = [(idx, 'normal') for idx in self.indices_list]
        
        if mirror:
            self.samples += [(idx, 'mirror') for idx in self.indices_list]
        if blur:
            self.samples += [(idx, 'blur') for idx in self.indices_list]
        if cropmix:
            self.samples += [(idx, 'cropmix') for idx in self.indices_list]

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        real_idx, aug_type = self.samples[idx]
        
        img_name = self.data.iloc[real_idx, 0]
        label = int(self.data.iloc[real_idx, 1]) if self.species == "cat" else int(self.data.iloc[real_idx, 1]) - 10
        
        orig_img_path = os.path.join(self.classes_dir if self.is_folder_mode else self.img_dir, img_name)
        cached_name = img_name.replace(os.sep, "_") if self.is_folder_mode else img_name
        cached_img_path = os.path.join(self.cache_dir, cached_name)
        
        if os.path.exists(cached_img_path):
            crop_image = Image.open(cached_img_path).convert("RGB")
        else:
            try:
                crop_np, _, _ = self.detector.detect_largest_animal(orig_img_path)
                crop_image = Image.fromarray(crop_np.cpu().numpy() if hasattr(crop_np, 'cpu') else crop_np).convert("RGB")
                crop_image.save(cached_img_path)
            except:
                crop_image = Image.open(orig_img_path).convert("RGB")
        
        # Augmentierungs-Funktionen anwenden
        if aug_type == 'mirror':
            crop_image = crop_image.transpose(Image.FLIP_LEFT_RIGHT)
        elif aug_type == 'blur':
            crop_image = crop_image.filter(ImageFilter.GaussianBlur(radius=1.5))
        elif aug_type == 'cropmix':
            # NEU: CropMix Implementation (Multi-Scale Blending des Bildes mit sich selbst)
            w, h = crop_image.size
            # Wähle einen zufälligen Bildausschnitt (60% bis 90% der Originalgröße)
            crop_frac = random.uniform(0.6, 0.9)
            new_w, new_h = int(w * crop_frac), int(h * crop_frac)
            x = random.randint(0, w - new_w)
            y = random.randint(0, h - new_h)
            
            # Ausschnitt ausschneiden und wieder auf Originalgröße skalieren
            cropped = crop_image.crop((x, y, x + new_w, y + new_h))
            cropped_resized = cropped.resize((w, h), Image.BILINEAR)
            
            # Original und skalierten Ausschnitt linear miteinander verschmelzen (Alpha 0.4 bis 0.6)
            alpha = random.uniform(0.4, 0.6)
            crop_image = Image.blend(crop_image, cropped_resized, alpha)
        
        return self.transform(crop_image), label

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs, device, species, exp_name):
    save_path = f"{species}_scratch_{exp_name}.pth"
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Train {species} Epoch {epoch}", file=sys.stdout, leave=False)
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                outputs = model.backbone(inputs)
                loss = criterion(outputs, labels)
                
            if scaler: 
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else: 
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        class_correct = [0] * NUM_CLASSES
        class_total = [0] * NUM_CLASSES
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Val {species} Epoch {epoch}", file=sys.stdout, leave=False)
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                    outputs = model.backbone(inputs)
                    loss = criterion(outputs, labels)
                    
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1

        epoch_train_loss = train_loss / train_total if train_total > 0 else 0.0
        epoch_train_acc = 100. * train_correct / train_total if train_total > 0 else 0.0
        epoch_val_loss = val_loss / val_total if val_total > 0 else 0.0
        epoch_val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
        
        per_class_acc = {}
        target_classes = CAT_CLASSES if species == "cat" else DOG_CLASSES
        for i in range(NUM_CLASSES):
            if class_total[i] > 0:
                per_class_acc[target_classes[i]] = 100. * class_correct[i] / class_total[i]
            else:
                per_class_acc[target_classes[i]] = 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = (f"Epoch [{epoch:03d}/{num_epochs}] | Species: {species.upper()} | LR: {current_lr:.6f} | "
                   f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:.2f}% | "
                   f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:.2f}%")
        print(log_msg, flush=True)

        scheduler.step()
        
        torch.save({
            "epoch": epoch, "model_state": model.state_dict(), "val_acc": epoch_val_acc,
            "val_loss": epoch_val_loss, "lr": current_lr, "per_class_acc": per_class_acc
        }, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3); parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--exp_name', type=str, default='default'); parser.add_argument('--data_dir', type=str, default='images')
    parser.add_argument('--classes_dir', type=str, default='./classes')
    
    parser.add_argument('--mirror', action='store_true', help='Aktiviert das Spiegeln zur Datenvervielfachung')
    parser.add_argument('--blur', action='store_true', help='Aktiviert Weichzeichnen zur Datenvervielfachung')
    # NEU: CropMix Flag registrieren
    parser.add_argument('--cropmix', action='store_true', help='Aktiviert CropMix (Multi-Scale Blending) zur Datenvervielfachung')
    # NEU: Class Imbalance Flag registrieren
    parser.add_argument('--balance_weights', action='store_true', help='Aktiviert die Klassen-Gewichtung für ungleichmäßig verteilte Datensätze')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for mode in [{"name": "CSV", "dir": None}]:
        for species in ["cat", "dog"]:
            
            detector = AnimalDetector(weights_path='yolov6s.pt', device=device)
            base_dataset = CroppedAnimalDataset(os.path.join(args.data_dir, "labels.csv"), args.data_dir, species, detector, transform, mode["dir"])
            
            print(f"[INFO] Prüfe/Erstelle Bild-Cache für {species} ({mode['name']})...")
            with torch.no_grad():
                for i in tqdm(range(len(base_dataset)), desc=f"Caching {species}"):
                    _ = base_dataset[i]
            
            del detector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            all_indices = list(range(len(base_dataset)))
            random.shuffle(all_indices)
            split_point = int(0.8 * len(all_indices))
            train_idx = all_indices[:split_point]
            val_idx = all_indices[split_point:]
            
            # NEU: args.cropmix an das Train-Dataset übergeben
            train_dataset = CroppedAnimalDataset(
                os.path.join(args.data_dir, "labels.csv"), args.data_dir, species, None, transform, mode["dir"],
                indices=train_idx, mirror=args.mirror, blur=args.blur, cropmix=args.cropmix
            )
            val_dataset = CroppedAnimalDataset(
                os.path.join(args.data_dir, "labels.csv"), args.data_dir, species, None, transform, mode["dir"],
                indices=val_idx, mirror=False, blur=False
            )
            
            print(f"[INFO] {species.upper()} - Originale Trainingsbilder: {len(train_idx)} -> Erweitert auf: {len(train_dataset)}")
            print(f"[INFO] {species.upper()} - Validierungsbilder (rein): {len(val_dataset)}")

            weights_file = f"{species}_scratch_{args.exp_name}.pth"
            model = AnimalClassifier(weights_path=weights_file if os.path.exists(weights_file) else None, num_classes=NUM_CLASSES, device=device)
            
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8)

            # Class Weights (Imbalance Handling) - nur wenn --balance_weights gesetzt ist
            if args.balance_weights:
                train_labels = []
                for idx in train_idx:
                    raw_label = int(base_dataset.data.iloc[idx, 1])
                    label = raw_label if species == "cat" else raw_label - 10
                    train_labels.append(label)

                class_counts = Counter(train_labels)
                total_train_samples = len(train_labels)

                weights = []
                for i in range(NUM_CLASSES):
                    count = class_counts.get(i, 0)
                    if count > 0:
                        w = total_train_samples / (NUM_CLASSES * count)
                    else:
                        w = 1.0
                    weights.append(w)

                class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
                formatted_weights = [f"{w:.2f}" for w in weights]
                print(f"[INFO] {species.upper()} Class Weights (balance_weights aktiv): {formatted_weights}")

                criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            else:
                print(f"[INFO] {species.upper()} - Kein Class Balancing (--balance_weights nicht gesetzt), nutze Standard CrossEntropyLoss.")
                criterion = nn.CrossEntropyLoss()

            train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, torch.amp.GradScaler() if device == "cuda" else None, args.epochs, device, species, args.exp_name)

            print(f"[INFO] Bereinige GPU-Speicher nach Phase: {mode['name']} - {species}...")
            del model, optimizer, scheduler, train_loader, val_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()