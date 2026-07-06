import os
import re
import glob
import csv
import requests
import tarfile
import shutil

# Exakt dieselbe Klassenliste wie in deiner API / deinem Oxford-Loader
CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat",
    "Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed",
    "Golden_Retriever", "German_Shepherd", "Siberian_Husky", "Dalmatian", "Rottweiler"
]

def get_next_counter(target_folder):
    """
    Sucht nach bereits existierenden fortlaufenden Dateien (z.B. 00250.jpg) 
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
    Lädt das Stanford-Dogs-Dataset automatisch herunter und entpackt es (ca. 760 MB).
    """
    if os.path.exists(os.path.join(extract_path, "Images")):
        print("Stanford-Dataset ist bereits lokal entpackt vorhanden. Überspringe Download.")
        return True

    print("Stanford Dogs Dataset wird heruntergeladen (ca. 760 MB)...")
    print("Das kann einen Moment dauern...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print("Download beendet. Entpacke das Archiv...")
            
            # Stanford nutzt ein unkomprimiertes .tar-Archiv, daher "r:" statt "r:gz"
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

def process_stanford_and_local(target_folder="images", local_cats_folder="local_cats"):
    # Offizielle URL für die Stanford-Dogs-Bilder
    stanford_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    archive_file = "stanford_images.tar"
    extract_folder = "stanford_dataset_raw"
    
    # 1. Höchsten existierenden Counter im Zielordner ermitteln (schützt Oxford-Bilder!)
    counter = get_next_counter(target_folder)
    print(f"Analysiere Zielordner '{target_folder}'...")
    print(f"Nächster freier Bild-Index: {counter:05d}.jpg")
    
    # 2. Stanford Download & Entpacken
    if not download_and_extract_stanford(stanford_url, archive_file, extract_folder):
        print("Pipeline abgebrochen.")
        return

    stanford_images_dir = os.path.join(extract_folder, "Images")
    
    os.makedirs(target_folder, exist_ok=True)
    csv_pfad = os.path.join(target_folder, "labels.csv")
    csv_exists = os.path.exists(csv_pfad) and os.path.getsize(csv_pfad) > 0
    
    copied_count = 0

    # 3. CSV im Append-Modus ('a') öffnen – wir hängen uns einfach hinten dran
    with open(csv_pfad, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        if not csv_exists:
            writer.writerow(["filename", "label"])
            
        print("\n--- Schritt 1: Integriere fehlende Hunderassen aus Stanford ---")
        
        # Stanford-Ordner haben das Format: n02113712-Golden_retriever
        for folder_name in sorted(os.listdir(stanford_images_dir)):
            folder_path = os.path.join(stanford_images_dir, folder_name)
            if not os.path.isdir(folder_path) or '-' not in folder_name:
                continue
            
            # Extrahiere den reinen Rassenamen nach dem Bindestrich und kleinschreiben
            breed_part = folder_name.split('-', 1)[1].lower()
            
            # NORMALISIERUNG: Unterstriche und Bindestriche zu Leerzeichen machen
            breed_part_normalized = breed_part.replace('_', ' ').replace('-', ' ').strip()
            
            # Auch deine CLASSES-Suchliste für den Vergleich genau so normalisieren
            classes_lower_normalized = [c.lower().replace('_', ' ').replace('-', ' ').strip() for c in CLASSES]
            
            if breed_part_normalized in classes_lower_normalized:
                label_id = classes_lower_normalized.index(breed_part_normalized)
                print(f"✓ Match gefunden! Verarbeite Stanford-Rasse: {CLASSES[label_id]} (Label {label_id})...")
                
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
                        print(f"   x Fehler bei {filename}: {e}")

        print("\n--- Schritt 2: Prüfe auf lokale Katzenbilder (Tabby / Tiger_Cat) ---")
        for cat_class in ["Tabby", "Tiger_Cat"]:
            cat_folder_path = os.path.join(local_cats_folder, cat_class)
            label_id = CLASSES.index(cat_class)
            
            if os.path.exists(cat_folder_path) and os.path.isdir(cat_folder_path):
                print(f"Lokaler Ordner für '{cat_class}' gefunden. Integriere Bilder...")
                for filename in sorted(os.listdir(cat_folder_path)):
                    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        continue
                        
                    new_filename = f"{counter:05d}.jpg"
                    src_path = os.path.join(cat_folder_path, filename)
                    dest_path = os.path.join(target_folder, new_filename)
                    
                    try:
                        shutil.copy(src_path, dest_path)
                        writer.writerow([new_filename, label_id])
                        counter += 1
                        copied_count += 1
                    except Exception as e:
                        print(f"   x Fehler bei lokalem Bild {filename}: {e}")
            else:
                print(f"ℹ️ Kein lokaler Ordner für '{cat_class}' unter '{cat_folder_path}' gefunden. (Übersprungen)")

    print(f"\nFertig! {copied_count} zusätzliche Bilder erfolgreich integriert.")
    print(f"Die labels.csv wurde erweitert. Nächster freier Index ist: {counter:05d}.jpg")

if __name__ == "__main__":
    process_stanford_and_local(target_folder="images", local_cats_folder="local_cats")