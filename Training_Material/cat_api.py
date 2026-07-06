import yaml
import requests
import os
import csv
import time

# Deine definierte Klassen-Liste 
CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat",
    "Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed",
    "Golden_Retriever", "German_Shepherd", "Siberian_Husky", "Dalmatian", "Rottweiler"
]

def process_cat_api(api_keys_file="api_keys.yaml", target_folder="images"):
    # 1. Config laden
    skript_ordner = os.path.dirname(os.path.abspath(__file__))
    yaml_pfad = os.path.join(skript_ordner, api_keys_file)
    
    with open(yaml_pfad, "r") as stream:
        config = yaml.safe_load(stream)

    headers = {"x-api-key": config["cat_api"]["x_api_key"]}
    base_url = "https://api.thecatapi.com/v1"
    
    os.makedirs(target_folder, exist_ok=True)
    csv_pfad = os.path.join(target_folder, "labels.csv")
    
    # 2. Checkpoint-Logik: Höchste existierende Bildnummer im Ordner finden
    start_counter = 0
    if os.path.exists(target_folder):
        for file_name in os.listdir(target_folder):
            if file_name.lower().endswith(".jpg"):
                name_part = os.path.splitext(file_name)[0]
                if name_part.isdigit():
                    file_num = int(name_part)
                    if file_num >= start_counter:
                        start_counter = file_num + 1
                        
    if start_counter > 0:
        print(f"[Checkpoint] Bestehende Bilder gefunden! Setze Nummerierung bei {start_counter:05d}.jpg fort.")
    else:
        print("[Checkpoint] Ordner ist leer. Starte neu bei 00000.jpg.")
        
    counter = start_counter
    seen_image_ids = set()
    csv_exists = os.path.exists(csv_pfad) and os.path.getsize(csv_pfad) > 0

    # 3. Rassen von der Cat API holen
    print("\nHole Rassen-Informationen von der cat_api...")
    breeds_resp = requests.get(f"{base_url}/breeds", headers=headers)
    api_breeds = breeds_resp.json()
    
    breed_mapping = {}
    for api_breed in api_breeds:
        normalized_name = api_breed["name"].replace(" ", "_")
        if normalized_name in CLASSES:
            breed_mapping[normalized_name] = api_breed["id"]

    print(f"Gefundene Katzen-Rassen: {list(breed_mapping.keys())}")

    # 4. Bilder laden (Append-Modus für die CSV!)
    with open(csv_pfad, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Header nur bei ganz neuer Datei
        if not csv_exists:
            writer.writerow(["filename", "label"])
            csv_file.flush()
            
        for class_name, api_breed_id in breed_mapping.items():
            label_id = CLASSES.index(class_name)
            print(f"\n--- Starte Download für {class_name} (Unser Label: {label_id}) ---")
            
            page = 0
            while True:
                # Das pure Setup, das bei der Cat API vorhin einwandfrei funktioniert hat
                params = {"breed_ids": api_breed_id, "page": page, "limit": 25}
                
                # Kurze Verschnaufpause für die API
                time.sleep(0.5) 
                
                try:
                    response = requests.get(f"{base_url}/images/search", headers=headers, params=params, timeout=10)
                    
                    if response.status_code == 429:
                        print("  ! Rate-Limit getroffen. Warte 5 Sekunden...")
                        time.sleep(5)
                        continue
                        
                    if response.status_code != 200:
                        print(f"Fehler: {response.status_code}")
                        break
                        
                    data = response.json()
                except Exception as e:
                    print(f"API Fehler: {e}")
                    break
                    
                if not data:
                    break
                    
                new_images_found = False
                for entity in data:
                    # Duplikat-Schutz
                    if entity['id'] not in seen_image_ids:
                        seen_image_ids.add(entity['id'])
                        new_images_found = True
                        
                        filename = f"{counter:05d}.jpg"
                        bild_pfad = os.path.join(target_folder, filename)
                        
                        try:
                            img_resp = requests.get(entity["url"], timeout=10)
                            if img_resp.status_code == 200:
                                with open(bild_pfad, "wb") as f:
                                    f.write(img_resp.content)
                                
                                writer.writerow([filename, label_id])
                                csv_file.flush()
                                
                                print(f"  ✓ {filename} (Label {label_id})")
                                counter += 1
                        except Exception as e:
                            print(f"  Fehler beim Bild-Download: {e}")
                
                # Wenn die Cat API anfängt die gleichen Bilder zu wiederholen -> Rasse beenden
                if not new_images_found:
                    print(f"  -> Pool für {class_name} erschöpft (Keine neuen Bilder mehr).")
                    break
                    
                page += 1

    print(f"\nFertig! Letzte Bildnummer im Ordner: {(counter - 1):05d}.jpg")

if __name__ == "__main__":
    process_cat_api()