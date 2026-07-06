import yaml
import requests
import os
import csv
import time

# Exakt dieselbe globale Klassen-Liste (Katzen 0-9, Hunde 10-19)
CLASSES = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat",
    "Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed",
    "Golden_Retriever", "German_Shepherd", "Siberian_Husky", "Dalmatian", "Rottweiler"
]

def process_dog_api(api_keys_file="api_keys.yaml", target_folder="images"):
    # 1. Config laden
    skript_ordner = os.path.dirname(os.path.abspath(__file__))
    yaml_pfad = os.path.join(skript_ordner, api_keys_file)
    
    with open(yaml_pfad, "r") as stream:
        config = yaml.safe_load(stream)

    headers = {"x-api-key": config["dog_api"]["x_api_key"]}
    base_url = "https://api.thedogapi.com/v1"
    
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

    # 3. Rassen von der Dog API holen
    print("\nHole Rassen-Informationen von der dog_api...")
    try:
        breeds_resp = requests.get(f"{base_url}/breeds", headers=headers, timeout=10)
        api_breeds = breeds_resp.json()
    except Exception as e:
        print(f"Fehler beim Laden der Rassenliste: {e}")
        return
    
    breed_mapping = {}
    for api_breed in api_breeds:
        normalized_name = api_breed["name"].replace(" ", "_")
        if normalized_name in CLASSES:
            breed_mapping[normalized_name] = api_breed["id"]

    print(f"Gefundene Hunde-Rassen für den Download: {list(breed_mapping.keys())}")

    # 4. Bilder laden (Append-Modus 'a')
    with open(csv_pfad, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not csv_exists:
            writer.writerow(["filename", "label"])
            csv_file.flush()
            
        for class_name, api_breed_id in breed_mapping.items():
            label_id = CLASSES.index(class_name)
            print(f"\n--- Starte Download für {class_name} (Unser Label: {label_id}) ---")
            
            page = 0
            images_downloaded_for_breed = 0
            max_images_per_breed = 50
            consecutive_empty_pages = 0
            
            while images_downloaded_for_breed < max_images_per_breed and page < 15:
                # Parameter optimiert: Kleines Limit zwingt das API-Backend, den Filter zu beachten!
                params = {
                    "breed_ids": str(api_breed_id), 
                    "breed_id": str(api_breed_id),  # Fallback für manche API-Versionen
                    "limit": 20,                     
                    "page": page,
                    "has_breeds": 1                 # Erzwingt das Mitliefern der Rassen-Metadaten für den Check
                }
                
                time.sleep(0.6) # Schutz vor Rate-Limits
                
                try:
                    response = requests.get(f"{base_url}/images/search", headers=headers, params=params, timeout=15)
                    if response.status_code != 200:
                        print(f"  API-Fehler {response.status_code}. Überspringe Seite.")
                        break
                    data = response.json()
                except Exception as e:
                    print(f"  Verbindungsfehler: {e}")
                    break
                    
                if not data:
                    break
                    
                valid_images_on_page = 0
                for entity in data:
                    if images_downloaded_for_breed >= max_images_per_breed:
                        break
                        
                    # STRENGE KONTROLLE: Prüfen, ob die gewünschte Rasse wirklich im Bild steckt
                    entity_breeds = entity.get("breeds", [])
                    breed_matches = any(str(b.get("id")) == str(api_breed_id) for b in entity_breeds)
                    
                    if not breed_matches:
                        # API hat uns ein falsches Bild untergeschoben -> KNALLHART ÜBERSPRINGEN!
                        continue
                    
                    # Duplikat-Schutz
                    if entity['id'] not in seen_image_ids:
                        seen_image_ids.add(entity['id'])
                        
                        filename = f"{counter:05d}.jpg"
                        bild_pfad = os.path.join(target_folder, filename)
                        
                        try:
                            img_resp = requests.get(entity["url"], timeout=10)
                            if img_resp.status_code == 200:
                                with open(bild_pfad, "wb") as f:
                                    f.write(img_resp.content)
                                
                                writer.writerow([filename, label_id])
                                csv_file.flush()
                                
                                print(f"  ✓ {filename} (Echter {class_name} verifiziert!)")
                                counter += 1
                                images_downloaded_for_breed += 1
                                valid_images_on_page += 1
                        except Exception as e:
                            print(f"  Fehler beim Download von {entity['url']}: {e}")
                
                if valid_images_on_page == 0:
                    consecutive_empty_pages += 1
                    if consecutive_empty_pages > 3:
                        print(f"  -> Pool für {class_name} erschöpft (keine echten Bilder mehr auf den Folgeseiten).")
                        break
                else:
                    consecutive_empty_pages = 0
                    
                page += 1
            
            print(f"  -> {class_name} beendet. {images_downloaded_for_breed} verifizierte Bilder gesichert.")

    print(f"\nFertig! Letzte Bildnummer im Ordner: {(counter - 1):05d}.jpg")

if __name__ == "__main__":
    process_dog_api()