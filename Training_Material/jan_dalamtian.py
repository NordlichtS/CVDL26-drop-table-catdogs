import os
import re
import glob
import csv
import shutil

# Exakt dieselbe globale Klassen-Liste
CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat",
    "Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed",
    "Golden_Retriever", "German_Shepherd", "Siberian_Husky", "Dalmatian", "Rottweiler"
]

def get_next_counter(target_folder):
    """
    Sucht nach bereits existierenden fortlaufenden Dateien (z.B. 05565.jpg) 
    und gibt die nächsthöhere freie Nummer zurück.
    """
    if not os.path.exists(target_folder):
        return 0
    
    existing_files = glob.glob(os.path.join(target_folder, "*.jpg"))
    max_counter = -1
    
    for f in existing_files:
        basename = os.path.basename(f)
        match = re.match(r"^(\d{5})\.jpg$", basename)
        if match:
            num = int(match.group(1))
            if num > max_counter:
                max_counter = num
                
    return max_counter + 1 if max_counter != -1 else 0

def process_local_darmatians(source_folder="/home/souschen/Downloads/dals", target_folder="images"):
    # 1. Überprüfen, ob der Quellordner existiert
    if not os.path.exists(source_folder):
        print(f"🛑 Fehler: Der Ordner '{source_folder}' wurde nicht gefunden!")
        return

    # 2. Checkpoint ermitteln (schützt alle bisherigen Bilder)
    counter = get_next_counter(target_folder)
    print(f"Analysiere Zielordner '{target_folder}'...")
    print(f"Nächster freier Bild-Index: {counter:05d}.jpg")
    
    os.makedirs(target_folder, exist_ok=True)
    csv_pfad = os.path.join(target_folder, "labels.csv")
    csv_exists = os.path.exists(csv_pfad) and os.path.getsize(csv_pfad) > 0
    
    # Dalmatian hat in der CLASSES-Liste den Index 18
    label_id = CLASSES.index("Dalmatian")
    print(f"Verwende Label-ID {label_id} für Dalmatian.")

    copied_count = 0

    # 3. CSV im Append-Modus öffnen
    with open(csv_pfad, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not csv_exists:
            writer.writerow(["filename", "label"])
            csv_file.flush()
            
        print(f"\n--- Starte Kopieren der Dalmatiner-Bilder aus '{source_folder}' ---")
        
        # Alle Dateien im Quellordner sortiert durchgehen
        for filename in sorted(os.listdir(source_folder)):
            # Nur gängige Bildformate mitnehmen
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            new_filename = f"{counter:05d}.jpg"
            src_path = os.path.join(source_folder, filename)
            dest_path = os.path.join(target_folder, new_filename)
            
            try:
                shutil.copy(src_path, dest_path)
                writer.writerow([new_filename, label_id])
                counter += 1
                copied_count += 1
                if copied_count % 50 == 0:
                    print(f"  -> {copied_count} Bilder verarbeitet...")
            except Exception as e:
                print(f"   x Fehler bei Datei {filename}: {e}")
                
        csv_file.flush()

    print(f"\nFertig! {copied_count} Dalmatiner-Bilder erfolgreich integriert.")
    print(f"Die labels.csv wurde erweitert. Nächster freier Index ist: {counter:05d}.jpg")

if __name__ == "__main__":
    process_local_darmatians()