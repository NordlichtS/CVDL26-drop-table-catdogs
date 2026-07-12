import torch
import os
from animalClassifier import AnimalClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 10

# 1. Hier den Ordnerpfad angeben
weights_dir = "./"
#weight_data/data_on30000pictures_third_run"

# 2. Automatisch alle Dateien finden, die auf .pth enden
if os.path.exists(weights_dir):
    pth_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
    pth_files.sort()  # Sortiert die Liste alphabetisch für eine schöne Übersicht
else:
    print(f"[ERROR] Der Ordner '{weights_dir}' wurde nicht gefunden!")
    pth_files = []

print(f"{'DATEINAME':<40} | {'STATUS':<15} | {'VAL ACCURACY':<12}")
print("-" * 75)

for file in pth_files:
    # 3. Den vollen Pfad für das Laden und Prüfen zusammenbauen
    full_path = os.path.join(weights_dir, file)
    
    if not os.path.exists(full_path):
        print(f"{file:<40} | NICHT GEFUNDEN  | -")
        print("-" * 75)
        continue
        
    try:
        # Hier überall 'full_path' statt 'file' nutzen:
        checkpoint = torch.load(full_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint
            
        val_acc_str = "-"
        if isinstance(checkpoint, dict) and "val_acc" in checkpoint:
            val_acc_str = f"{checkpoint['val_acc']:.2f}%"
            
        num_tensors = len(state_dict.keys())
        
        print(f"{file:<40} | {num_tensors:>3} Tensoren OK | {val_acc_str:<12}")
        
        if isinstance(checkpoint, dict) and "per_class_acc" in checkpoint:
            print("    └─> Genauigkeit nach Rasse:")
            for class_name, acc in checkpoint["per_class_acc"].items():
                print(f"        • {class_name:<20}: {acc:.2f}%")
        
        print("-" * 75)
            
    except Exception as e:
        print(f"{file:<40} | DEFEKT/CRASHED  | -")
        print(f"[ERROR]: {e}")
        print("-" * 75)