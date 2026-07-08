#!/usr/bin/env python3
"""
Pet breed classifier — EfficientNetV2-S, Kaggle 2x GPU edition.

- nn.DataParallel across all visible GPUs (T4 x2 on Kaggle)
- BATCH_SIZE=64 (32 per GPU), num_workers=4
- Warmup (2 epochs) -> cosine annealing
- Fixed CLASSES id order, stratified split, WeightedRandomSampler
- Per-epoch report in logs: loss, acc, macro F1, per-class F1, worst classes, LR, time
- Checkpoint unwraps DataParallel (no 'module.' prefix hell)
"""

import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import datasets, transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image
from tqdm import tqdm

# ----------------------------- Config ---------------------------------------
DATASET_PATH = "/kaggle/input/datasets/vaniakazakov/test-animals-rec/classes"
OUT_DIR = "/kaggle/working"
BATCH_SIZE = 64          # split across GPUs by DataParallel (32 per T4)
TRAIN_SPLIT = 0.8
SEED = 42
IMG_SIZE = 384
NUM_WORKERS = 4

FROM_SCRATCH = False
EPOCHS = 80 if FROM_SCRATCH else 15
LR = 1e-3 if FROM_SCRATCH else 1e-4
WARMUP_EPOCHS = 5 if FROM_SCRATCH else 2

CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat",
    "Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed",
    "Golden_Retriever", "German_Shepherd", "Siberian_Husky", "Dalmatian", "Rottweiler",
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_GPUS = torch.cuda.device_count() if DEVICE == "cuda" else 0
print(f"Device: {DEVICE} | GPUs: {N_GPUS}")
if N_GPUS > 1:
    for i in range(N_GPUS):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# ----------------------------- Data -----------------------------------------
base = datasets.ImageFolder(DATASET_PATH)
assert set(base.classes) == set(CLASSES), (
    f"Folder/class mismatch:\nfolders: {base.classes}\nexpected: {CLASSES}"
)
remap = {base.class_to_idx[c]: CLASS_TO_ID[c] for c in CLASSES}
all_samples = [(path, remap[t]) for path, t in base.samples]

rng = random.Random(SEED)
by_class = defaultdict(list)
for s in all_samples:
    by_class[s[1]].append(s)

train_samples, test_samples = [], []
for cls, items in by_class.items():
    rng.shuffle(items)
    n_train = int(TRAIN_SPLIT * len(items))
    train_samples.extend(items[:n_train])
    test_samples.extend(items[n_train:])
rng.shuffle(train_samples)

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
    transforms.ToTensor(),
    NORM,
    transforms.RandomErasing(p=0.25),
])
test_tf = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    NORM,
])


class PetDataset(Dataset):
    def __init__(self, samples, tf):
        self.samples = samples
        self.tf = tf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.tf(img), label


train_ds = PetDataset(train_samples, train_tf)
test_ds = PetDataset(test_samples, test_tf)

counts = Counter(lbl for _, lbl in train_samples)
wcls = {c: 1.0 / n for c, n in counts.items()}
sample_weights = [wcls[lbl] for _, lbl in train_samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          persistent_workers=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True,
                         persistent_workers=True)

print(f"Train: {len(train_ds)} | Test: {len(test_ds)} | Batches/epoch: {len(train_loader)}")
print("Per-class counts:", {CLASSES[c]: n for c, n in sorted(counts.items())})

# ----------------------------- Model ----------------------------------------
weights = None if FROM_SCRATCH else EfficientNet_V2_S_Weights.DEFAULT
model = efficientnet_v2_s(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

if N_GPUS > 1:
    model = nn.DataParallel(model)
    print(f"DataParallel enabled on {N_GPUS} GPUs ({BATCH_SIZE // N_GPUS} imgs/GPU)")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
                                          total_iters=WARMUP_EPOCHS),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=EPOCHS - WARMUP_EPOCHS),
    ],
    milestones=[WARMUP_EPOCHS],
)
scaler = torch.amp.GradScaler() if DEVICE == "cuda" else None

print(f"Mode: {'FROM SCRATCH' if FROM_SCRATCH else 'pretrained'} | "
      f"lr={LR} (warmup {WARMUP_EPOCHS} ep) | epochs={EPOCHS} | batch={BATCH_SIZE}")


# ----------------------------- Metrics --------------------------------------
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
    return float(np.mean(f1s)), f1s, cm


def unwrap(m):
    return m.module if isinstance(m, nn.DataParallel) else m


# ----------------------------- Train loop -----------------------------------
best_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    cur_lr = optimizer.param_groups[0]["lr"]

    # --- train ---
    model.train()
    correct = total = 0
    running_loss = 0.0
    n_batches = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=scaler is not None):
            out = model(imgs)
            loss = criterion(out, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()
        n_batches += 1
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{running_loss / n_batches:.3f}",
                         acc=f"{100 * correct / total:.2f}%")
    train_loss = running_loss / n_batches
    train_acc = correct / total
    scheduler.step()

    # --- eval ---
    model.eval()
    all_preds, all_labels = [], []
    test_loss = 0.0
    n_test_batches = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing", leave=False):
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels_d = labels.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=scaler is not None):
                out = model(imgs)
                test_loss += criterion(out, labels_d).item()
            n_test_batches += 1
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(labels.tolist())
    test_loss /= n_test_batches
    test_acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1, per_class_f1, cm = macro_f1_and_cm(all_preds, all_labels)
    dt = time.time() - t0

    # --- per-epoch report in logs ---
    ranked = sorted(zip(CLASSES, per_class_f1), key=lambda x: x[1])
    worst = ", ".join(f"{n}={v:.2f}" for n, v in ranked[:3])
    best3 = ", ".join(f"{n}={v:.2f}" for n, v in ranked[-3:])
    print(f"\n===== EPOCH {epoch}/{EPOCHS} REPORT =====")
    print(f"  lr           : {cur_lr:.2e}")
    print(f"  train        : loss {train_loss:.4f} | acc {train_acc * 100:.2f}%")
    print(f"  test         : loss {test_loss:.4f} | acc {test_acc * 100:.2f}% | macro F1 {f1:.4f}")
    print(f"  worst classes: {worst}")
    print(f"  best classes : {best3}")
    print(f"  time         : {dt:.0f}s")

    if f1 > best_f1:
        best_f1 = f1
        torch.save({
            "model_state": unwrap(model).state_dict(),   # no 'module.' prefix
            "classes": CLASSES,
            "epoch": epoch,
            "macro_f1": f1,
            "test_acc": test_acc,
            "from_scratch": FROM_SCRATCH,
            "img_size": IMG_SIZE,
        }, f"{OUT_DIR}/best_model.pth")
        np.savetxt(f"{OUT_DIR}/confusion_matrix.csv", cm, fmt="%d", delimiter=",",
                   header=",".join(CLASSES), comments="")
        print(f"  >> new best, checkpoint saved (macro F1 {f1:.4f})")
    print("=" * 34)

print(f"\nFinished. Best macro F1: {best_f1:.4f}")