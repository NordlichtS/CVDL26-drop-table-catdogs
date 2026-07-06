import os
import re
import glob
import csv
import requests
import tarfile
import shutil

# Deine definierte Klassen-Liste (exakt dieselbe Struktur wie in der API)
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

def download_and_extract_oxford(url, archive_path, extract_path):
    """
    Lädt das Oxford-Dataset automatisch herunter und entpackt es, 
    falls es nicht schon lokal vorhanden ist.
    """
    if os.path.exists(os.path.join(extract_path, "images")):
        print("Oxford-Dataset ist bereits lokal entpackt vorhanden. Überspringe Download.")
        return True

    print("Oxford-IIIT Pet Dataset wird von der Universität Oxford heruntergeladen (ca. 770 MB)...")
    print("Das kann je nach Internetverbindung ein paar Minuten dauern.")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(archive_path, "wb") as f:
                # 1 MB Blöcke für schnelles Schreiben auf die Festplatte
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            print("Download beendet. Entpacke das Archiv...")
            
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_path)
            
            print("Entpacken erfolgreich abgeschlossen.")
            
            # Aufräumen der temporären .tar.gz Datei
            if os.path.exists(archive_path):
                os.remove(archive_path)
            return True
        else:
            print(f"Fehler beim Download. Server antwortete mit Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"Ein Fehler ist beim Herunterladen aufgetreten: {e}")
        return False

def process_oxford_dataset(target_folder="images"):
    # Offizielle Download-URL des Oxford Datasets
    oxford_url = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    archive_file = "oxford_images.tar.gz"
    extract_folder = "oxford_dataset_raw"
    
    # 1. Höchsten existierenden Counter im Zielordner ermitteln
    counter = get_next_counter(target_folder)
    print(f"Analysiere Zielordner '{target_folder}'...")
    print(f"Nächster freier Bild-Index: {counter:05d}.jpg")
    
    # 2. Datensatz automatisch herunterladen und entpacken
    if not download_and_extract_oxford(oxford_url, archive_file, extract_folder):
        print("Pipeline abgebrochen.")
        return

    oxford_images_dir = os.path.join(extract_folder, "images")
    if not os.path.exists(oxford_images_dir):
        print(f"Fehler: Der entpackte Bildordner wurde unter {oxford_images_dir} nicht gefunden.")
        return

    # Zielordner erstellen, falls er noch nicht existiert
    os.makedirs(target_folder, exist_ok=True)
    csv_pfad = os.path.join(target_folder, "labels.csv")
    
    # Prüfen, ob eine labels.csv bereits existiert und befüllt ist
    csv_exists = os.path.exists(csv_pfad) and os.path.getsize(csv_pfad) > 0
    
    classes_lower = [c.lower() for c in CLASSES]
    copied_count = 0

    # 3. CSV im Append-Modus ('a') öffnen, damit bestehende Daten geschützt sind
    with open(csv_pfad, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Den Header (filename,label) nur schreiben, wenn die Datei komplett leer/neu ist
        if not csv_exists:
            writer.writerow(["filename", "label"])
            
        print("\nVerarbeite Oxford-Dataset und integriere passende Rassen...")
        
        # Sortierte Dateiliste durchgehen für eine feste Reihenfolge
        for filename in sorted(os.listdir(oxford_images_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg')):
                continue
                
            # Oxford-Format ist: Rasse_Nummer.jpg (z.B. Abyssinian_12.jpg oder yorkshire_terrier_3.jpg)
            if '_' not in filename:
                continue
                
            # Trenne den Rassenamen von der Nummer am Ende (trennt am letzten '_')
            name_part, _ = os.path.splitext(filename)[0].rsplit('_', 1)
            
            # Groß-/Kleinschreibung ignorieren beim Abgleich mit deiner Klassenliste
            if name_part.lower() in classes_lower:
                label_id = classes_lower.index(name_part.lower())
                
                # Neuen Dateinamen basierend auf dem aktuellen Counter bestimmen
                new_filename = f"{counter:05d}.jpg"
                src_image_path = os.path.join(oxford_images_dir, filename)
                dest_image_path = os.path.join(target_folder, new_filename)
                
                try:
                    # Bild in deinen Hauptordner kopieren
                    shutil.copy(src_image_path, dest_image_path)
                    # Zeile an die bestehende CSV anhängen
                    writer.writerow([new_filename, label_id])
                    
                    print(f"  ✓ Oxford: {filename} -> {new_filename} (Label {label_id}: {CLASSES[label_id]})")
                    
                    counter += 1
                    copied_count += 1
                except Exception as e:
                    print(f"  x Fehler beim Kopieren von {filename}: {e}")

    print(f"\nFertig! {copied_count} passende Bilder aus dem Oxford-Dataset integriert.")
    print(f"Die labels.csv wurde aktualisiert. Nächster freier Index ist: {counter:05d}.jpg")

if __name__ == "__main__":
    # Startet den Prozess standardmäßig für den Ordner "images"
    process_oxford_dataset(target_folder="images")