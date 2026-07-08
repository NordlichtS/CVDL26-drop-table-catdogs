import os
import time
import random
import numpy as np
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from collections import Counter
from PIL import Image

# Importe aus deinem Projekt
from Final_Project.detector import AnimalDetector
from animalClassifier import AnimalClassifier

# Konstanten
CAT_CLASSES = ["Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair", "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat"]
DOG_CLASSES = ["Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed", "Golden_Retriever", "German_Shepherd", "Siberian_Husky", "Dalmatian", "Rottweiler"]
NUM_CLASSES = 10

class CroppedAnimalDataset(Dataset):
    def __init__(self, csv_file, img_dir, species, detector, transform=None, classes_dir=None):
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

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        label = int(self.data.iloc[idx, 1]) if self.species == "cat" else int(self.data.iloc[idx, 1]) - 10
        
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
        
        return self.transform(crop_image), label

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler, num_epochs, device, species, exp_name):
    save_path = f"{species}_scratch_{exp_name}.pth"
    for epoch in range(1, num_epochs + 1):
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"{species} Epoch {epoch}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                loss = criterion(model.backbone(inputs), labels)
            if scaler: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else: loss.backward(); optimizer.step()
        scheduler.step()
        torch.save({"model_state": model.state_dict()}, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3); parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--exp_name', type=str, default='default'); parser.add_argument('--data_dir', type=str, default='images')
    parser.add_argument('--classes_dir', type=str, default='./classes')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = AnimalDetector(weights_path='yolov6s.pt', device=device)
    
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Trainings-Phasen
    for mode in [{"name": "CSV", "dir": None}, {"name": "FOLDER", "dir": args.classes_dir}]:
        for species in ["cat", "dog"]:
            full_dataset = CroppedAnimalDataset(os.path.join(args.data_dir, "labels.csv"), args.data_dir, species, detector, transform, mode["dir"])
            
            # Splitting & Loader
            indices = list(range(len(full_dataset))); random.shuffle(indices)
            train_idx, val_idx = indices[:int(0.8*len(indices))], indices[int(0.8*len(indices)):]
            
            # Initialisiere Modell (lädt automatisch .pth falls vorhanden)
            weights_file = f"{species}_scratch_{args.exp_name}.pth"
            model = AnimalClassifier(weights_path=weights_file if os.path.exists(weights_file) else None, num_classes=NUM_CLASSES, device=device)
            
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            
            train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=16, shuffle=True)
            val_loader = DataLoader(torch.utils.data.Subset(full_dataset, val_idx), batch_size=16)
            
            train_model(model, train_loader, val_loader, nn.CrossEntropyLoss(), optimizer, scheduler, torch.amp.GradScaler() if device == "cuda" else None, args.epochs, device, species, args.exp_name)