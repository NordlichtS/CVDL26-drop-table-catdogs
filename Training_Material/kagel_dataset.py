import os
import re
import glob
import csv
import shutil
import kagglehub

# Exakt dieselbe globale Klassen-Liste (Katzen 0-9, Hunde 10-19)
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

def process_kaggle_cats(target_folder="images"):
    print("Greife auf das Cat-Breeds Dataset von Kaggle zu...")
    raw_dataset_path = kagglehub.dataset_download("nikolasgegenava/cat-breeds")
    
    # 1. Checkpoint ermitteln
    counter = get_next_counter(target_folder)
    print(f"\nAnalysiere Zielordner '{target_folder}'...")
    print(f"Nächster freier Bild-Index: {counter:05d}.jpg")
    
    os.makedirs(target_folder, exist_ok=True)
    csv_pfad = os.path.join(target_folder, "labels.csv")
    csv_exists = os.path.exists(csv_pfad) and os.path.getsize(csv_pfad) > 0
    
    # 2. Struktur-Suche: Wir finden heraus, wo im Kaggle-Cache die eigentlichen Rassen-Ordner stecken
    source_images_dir = None
    
    # Wir wandern durch das Dataset und suchen nach Ordnern, die eine unserer Klassen enthalten könnten
    classes_lower_flat = [c.lower().replace('_', '').replace('-', '') for c in CLASSES[:10]]
    
    print("\nScanne Dataset-Struktur nach Katzenrassen...")
    for root, dirs, files in os.walk(raw_dataset_path):
        # Wenn wir einen Ordner finden, der wie eine unserer Katzenrassen heißt, haben wir die richtige Ebene gefunden!
        for d in dirs:
            if d.lower().replace('_', '').replace('-', '').replace(' ', '') in classes_lower_flat:
                source_images_dir = root
                break
        if source_images_dir:
            break

    if not source_images_dir:
        print("🛑 Fehler: Konnte keine passenden Rassen-Ordner im heruntergeladenen Dataset finden!")
        print("Inhalt des Kaggle-Ordners zur Diagnose:")
        for root, dirs, files in os.walk(raw_dataset_path):
            print(f" Ordner: {root}")
            if dirs: print(f"   Unterordner: {dirs}")
            if files: print(f"   Dateien (Auszug): {files[:5]}")
            break
        return

    print(f"-> Rassen-Ordner erfolgreich lokalisiert in: {source_images_dir}")

    copied_count = 0
    classes_lower_normalized = [c.lower().replace('_', '').replace('-', '').strip() for c in CLASSES]

    # 3. CSV im Append-Modus öffnen
    with open(csv_pfad, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not csv_exists:
            writer.writerow(["filename", "label"])
            csv_file.flush()
            
        print("\n--- Starte Sortierung und Integration der Katzenrassen ---")
        
        for folder_name in sorted(os.listdir(source_images_dir)):
            folder_path = os.path.join(source_images_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            folder_normalized = folder_name.lower().replace('_', '').replace('-', '').replace(' ', '').strip()
            
            if folder_normalized in classes_lower_normalized:
                label_id = classes_lower_normalized.index(folder_normalized)
                class_name = CLASSES[label_id]
                print(f"✓ Match gefunden! Verarbeite Rasse: {class_name} (Label {label_id})...")
                
                for filename in sorted(os.listdir(folder_path)):
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                        
                    new_filename = f"{counter:05d}.jpg"
                    src_path = os.path.join(folder_path, filename)
                    dest_path = os.path.join(target_folder, new_filename)
                    
                    try:
                        shutil.copy(src_path, dest_path)
                        writer.writerow([new_filename, label_id])
                        counter += 1
                        copied_count += 1
                    except Exception as e:
                        print(f"   x Fehler bei Datei {filename}: {e}")
                        
        csv_file.flush()

    print(f"\nFertig! {copied_count} Katzenbilder erfolgreich in '{target_folder}' integriert.")
    print(f"Die labels.csv wurde erweitert. Nächster freier Index ist: {counter:05d}.jpg")

if __name__ == "__main__":
    process_kaggle_cats(target_folder="images")