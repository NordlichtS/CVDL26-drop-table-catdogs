import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# 1. Zielordner definieren und erstellen, falls er noch nicht existiert
output_folder = os.path.join("Epoch", "loss_Graphs")
os.makedirs(output_folder, exist_ok=True)

# 2. Alle Logdateien im aktuellen Ordner suchen, die mit "training_output_" anfangen
# In Zeile 11:
log_files = [f for f in os.listdir('./TrainingLogs') if f.startswith('training_output_') and f.endswith('.log')]

if not log_files:
    print("Keine Logdateien mit dem Präfix 'training_output_' im aktuellen Ordner gefunden.")
    exit()

# Regex-Muster zum Extrahieren der Epochen-Metriken
pattern = re.compile(
    r"Epoch\s+\[(\d+)/\d+\]\s+\|\s+Species:\s+([A-Z_]+)\s+\|\s+LR:\s+[0-9.e-]+\s+\|\s+Train Loss:\s+([0-9.]+)\s+-\s+Acc:\s+([0-9.]+)%\s+\|\s+Val Loss:\s+([0-9.]+)\s+-\s+Acc:\s+([0-9.]+)%"
)

# Styling-Konfiguration für die Graphen (Präsentations-Stil)
plt.style.use('seaborn-v0_8-whitegrid')
colors = {
    'train_loss': '#1f77b4',  # Blau
    'val_loss': '#ff7f0e',    # Orange
    'train_acc': '#2ca02c',   # Grün
    'val_acc': '#d62728'      # Rot
}

print(f"Gefundene Logdateien: {log_files}\nStarte Verarbeitung...")

for log_file in log_files:
    # Dateiinhalt einlesen

    with open(os.path.join('./TrainingLogs', log_file), 'r', encoding='utf-8') as f:
        content = f.read()
    
    matches = pattern.findall(content)
    if not matches:
        print(f" -> Überspringe {log_file}: Keine passenden Trainingsmetriken gefunden.")
        continue
        
    # Daten nach Tierklasse (Species) gruppieren
    runs = {}
    for m in matches:
        epoch = int(m[0])
        species = m[1]
        train_loss = float(m[2])
        train_acc = float(m[3])
        val_loss = float(m[4])
        val_acc = float(m[5])
        
        if species not in runs:
            runs[species] = []
        runs[species].append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
    base_name = os.path.splitext(log_file)[0]
    
    for species, data in runs.items():
        df = pd.DataFrame(data)
        
        # Subplots erstellen (1 Zeile, 2 Spalten)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # --- 1. LOSS PLOT ---
        ax1.plot(df['epoch'], df['train_loss'], label="Train Loss", color=colors['train_loss'], linewidth=2.5, marker='o', markersize=4)
        ax1.plot(df['epoch'], df['val_loss'], label="Validation Loss", color=colors['val_loss'], linewidth=2.5, marker='s', markersize=4)
        
        # Beste Epoche ermitteln (Höchste Val Accuracy)
        best_val_idx = df['val_acc'].idxmax()
        best_epoch = df.loc[best_val_idx, 'epoch']
        best_val_loss = df.loc[best_val_idx, 'val_loss']
        best_val_acc = df.loc[best_val_idx, 'val_acc']
        
        # Beste Epoche als vertikale Linie und goldener Stern einzeichnen (Präsentations-Tipp 2)
        ax1.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax1.plot(best_epoch, best_val_loss, marker='*', markersize=12, color='gold', markeredgecolor='black', label='Best Checkpoint')
        
        ax1.set_title(f"Model Loss ({species})", fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel("Epoch", fontsize=11, labelpad=8)
        ax1.set_ylabel("Loss", fontsize=11, labelpad=8)
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        
        # --- 2. ACCURACY PLOT ---
        ax2.plot(df['epoch'], df['train_acc'], label="Train Accuracy", color=colors['train_acc'], linewidth=2.5, marker='o', markersize=4)
        ax2.plot(df['epoch'], df['val_acc'], label="Validation Accuracy", color=colors['val_acc'], linewidth=2.5, marker='s', markersize=4)
        
        ax2.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax2.plot(best_epoch, best_val_acc, marker='*', markersize=12, color='gold', markeredgecolor='black', label=f'Max Acc: {best_val_acc:.2f}%')
        
        ax2.set_title(f"Model Accuracy ({species})", fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel("Epoch", fontsize=11, labelpad=8)
        ax2.set_ylabel("Accuracy (%)", fontsize=11, labelpad=8)
        ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        
        plt.suptitle(f"Training Curve for Class: {species} (from {log_file})", fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Dynamisch im Zielordner speichern
        out_filename = os.path.join(output_folder, f"{base_name}_{species.lower()}.png")
        plt.savefig(out_filename, dpi=300, bbox_inches='tight')
        print(f" -> Erfolgreich gespeichert: {out_filename}")
        plt.close()

print("\nFertig! Alle Graphen wurden im Ordner 'Epoch/loss_Graphs' abgelegt.")