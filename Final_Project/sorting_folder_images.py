import os
import glob
import shutil

# --- Konfiguration ---
images_dir = 'images'
classes_dir = 'classes'
csv_file = os.path.join(images_dir, 'labels.csv')

# Exakte Klassenzuordnung, wie von dir vorgegeben
CLASSES = [
    "Abyssinian",         #  0
    "Bengal",             #  1
    "Birman",             #  2
    "Bombay",             #  3
    "British_Shorthair",  #  4
    "Maine_Coon",         #  5
    "Ragdoll",            #  6
    "Sphynx",             #  7
    "Tabby",              #  8
    "Tiger_Cat",          #  9
    "Beagle",             # 10
    "Pug",                # 11
    "Boxer",              # 12
    "Shiba_Inu",          # 13
    "Samoyed",            # 14
    "Golden_Retriever",   # 15
    "German_Shepherd",    # 16
    "Siberian_Husky",     # 17
    "Dalmatian",          # 18
    "Rottweiler",         # 19
]

def get_highest_id(image_folder):
    """Sucht im images-Ordner nach der höchsten vergebenen Nummer."""
    existing_images = glob.glob(os.path.join(image_folder, '*.jpg'))
    max_id = -1
    for img_path in existing_images:
        basename = os.path.basename(img_path)
        name_without_ext = os.path.splitext(basename)[0]
        try:
            num = int(name_without_ext)
            if num > max_id:
                max_id = num
        except ValueError:
            pass 
    return max_id

def main():
    # 1. Letzte ID herausfinden
    last_id = get_highest_id(images_dir)
    current_id = last_id + 1 if last_id >= 0 else 0
    print(f"Letzte gefundene ID: {last_id}. Starte neue Bilder bei: {current_id:05d}.jpg")

    # 2. Dictionary aus der vorgegebenen Liste erstellen für schnelles Nachschlagen
    # Das sieht dann so aus: {'Abyssinian': 0, 'Bengal': 1, ...}
    class_to_id = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

    # 3. Bilder verschieben und CSV aktualisieren
    moved_count = 0
    
    with open(csv_file, 'a', encoding='utf-8') as f:
        
        # Wir durchlaufen die Ordner in `classes`
        # os.listdir(classes_dir) gibt uns die Ordnernamen (Abyssinian, Beagle, etc.)
        for folder_name in os.listdir(classes_dir):
            cls_dir_path = os.path.join(classes_dir, folder_name)
            
            # Prüfen, ob es wirklich ein Ordner ist und ob der Name in unserer Liste steht
            if os.path.isdir(cls_dir_path):
                if folder_name not in class_to_id:
                    print(f"WARNUNG: Ordner '{folder_name}' gefunden, aber er steht nicht in der CLASSES-Liste. Wird übersprungen.")
                    continue
                
                # Die korrekte ID aus dem Dictionary holen (0 bis 19)
                cls_id = class_to_id[folder_name]
                
                # Alle Dateien im jeweiligen Rassen-Ordner durchgehen
                for filename in os.listdir(cls_dir_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        old_path = os.path.join(cls_dir_path, filename)
                        
                        # Neuen Dateinamen generieren (z.B. 08634.jpg)
                        new_filename = f"{current_id:05d}.jpg"
                        new_path = os.path.join(images_dir, new_filename)
                        
                        # Bild verschieben
                        shutil.move(old_path, new_path)
                        
                        # Eintrag in die CSV schreiben (Format: filename,label)
                        f.write(f"{new_filename},{cls_id}\n")
                        
                        current_id += 1
                        moved_count += 1

    print(f"\nFertig! Es wurden erfolgreich {moved_count} Bilder verschoben und in '{csv_file}' eingetragen.")

if __name__ == "__main__":
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    main()