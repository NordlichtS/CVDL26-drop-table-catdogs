import torch
import os
from animalClassifier import AnimalClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 10

pth_files = [
    "cat_scratch_lr:0.0001 ep:100.pth",
    "cat_scratch_lr:0.0003 ep:100.pth",
    "cat_scratch_lr:0.00003 ep:100.pth",
    "cat_scratch_lr:0.00005 ep:100.pth",
    "cat_scratch.pth",
    "dog_scratch_lr:0.0001 ep:100.pth",
    "dog_scratch_lr:0.0003 ep:100.pth",
    "dog_scratch_lr:0.00003 ep:100.pth",
    "dog_scratch_lr:0.00005 ep:100.pth"
]

print(f"{'DATEINAME':<40} | {'STATUS':<15} | {'VAL ACCURACY':<12}")
print("-" * 75)

for file in pth_files:
    if not os.path.exists(file):
        print(f"{file:<40} | NICHT GEFUNDEN  | -")
        continue
        
    try:
        # Gewichte laden
        checkpoint = torch.load(file, map_location=device, weights_only=True)
        
        # Falls es ein Dictionary ist, holen wir den State, sonst ist es direkt das State-Dict
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
            
        # Überprüfen, ob Metriken mitgespeichert wurden
        val_acc_str = "-"
        if isinstance(checkpoint, dict) and "val_acc" in checkpoint:
            val_acc_str = f"{checkpoint['val_acc']:.2f}%"
            
        # Zähle wie viele Layer-Gewichte in der Datei stecken
        num_tensors = len(state_dict.keys())
        
        print(f"{file:<40} | {num_tensors:>3} Tensoren OK | {val_acc_str:<12}")
            
    except Exception as e:
        print(f"{file:<40} | DEFEKT/CRASHED  | -")