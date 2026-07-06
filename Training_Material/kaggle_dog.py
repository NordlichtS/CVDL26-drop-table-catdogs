import os
import re
import glob
import csv
import requests
import tarfile
import shutil

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

def download_and_extract_stanford(url, archive_path, extract_path):
    """
    Lädt das Stanford-Dogs-Dataset von der offiziellen Uni-URL herunter, falls nicht vorhanden.
    """
    if os.path.exists(os.path.join(extract_path, "Images")):
        print("Stanford-Dataset ist bereits lokal entpackt vorhanden. Überspringe Download.")
        return True

    print("Stanford Dogs Dataset wird direkt von Stanford geladen (ca. 760 MB)...")
    print("Das kann einen Moment dauern...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print("Download beendet. Entpacke das Archiv...")
            
            # Stanford nutzt ein unkomprimiertes .tar-Archiv
            with tarfile.open(archive_path, "r:") as tar:
                tar.extractall(path=extract_path)
            
            print("Entpacken erfolgreich abgeschlossen.")
            
            if os.path.exists(archive_path):
                os.remove(archive_path)
            return True
        else:
            print(f"Fehler beim Download. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"Ein Fehler ist beim Herunterladen aufgetreten: {e}")
        return False

def process_stanford_dogs(target_folder="images"):
    # Die offizielle, unbeschränkte Direkt-URL der Stanford University
    stanford_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    archive_file = "stanford_images.tar"
    extract_folder = "stanford_dataset_raw"
    
    # 1. Checkpoint ermitteln
    counter = get_next_counter(target_folder)
    print(f"Analysiere Zielordner '{target_folder}'...")
    print(f"Nächster freier Bild-Index: {counter:05d}.jpg")
    
    # 2. Stanford Download & Entpacken (ohne jegliche 403 Kaggle-Sperren)
    if not download_and_extract_stanford(stanford_url, archive_file, extract_folder):
        print("Pipeline abgebrochen.")
        return

    stanford_images_dir = os.path.join(extract_folder, "Images")
    
    os.makedirs(target_folder, exist_ok=True)
    csv_pfad = os.path.join(target_folder, "labels.csv")
    csv_exists = os.path.exists(csv_pfad) and os.path.getsize(csv_pfad) > 0
    
    copied_count = 0
    # Komplette normalisierte Liste für den Index-Abgleich (Unterstriche und Leerzeichen weg)
    classes_lower_normalized = [c.lower().replace('_', '').replace('-', '').strip() for c in CLASSES]

    # 3. CSV im Append-Modus öffnen
    with open(csv_pfad, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not csv_exists:
            writer.writerow(["filename", "label"])
            csv_file.flush()
            
        print("\n--- Starte Sortierung und Integration der Hunderassen ---")
        
        for folder_name in sorted(os.listdir(stanford_images_dir)):
            folder_path = os.path.join(stanford_images_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
                
            # Stanford-Präfix abschneiden (z.B. n02110341-dalmatian -> dalmatian)
            breed_pure = folder_name.split('-', 1)[1] if '-' in folder_name else folder_name
            folder_normalized = breed_pure.lower().replace('_', '').replace('-', '').replace(' ', '').strip()
            
            if folder_normalized in classes_lower_normalized:
                label_id = classes_lower_normalized.index(folder_normalized)
                class_name = CLASSES[label_id]
                print(f"✓ Match gefunden! Verarbeite Rasse: {class_name} (Label {label_id})...")
                
                for filename in sorted(os.listdir(folder_path)):
                    if not filename.lower().endswith(('.jpg', '.jpeg')):
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

    print(f"\nFertig! {copied_count} Hundebilder erfolgreich in '{target_folder}' integriert.")
    print(f"Die labels.csv wurde erweitert. Nächster freier Index ist: {counter:05d}.jpg")

if __name__ == "__main__":
    process_stanford_dogs(target_folder="images")