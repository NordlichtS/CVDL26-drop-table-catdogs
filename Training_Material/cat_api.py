"""
Description for the api key acknowledment:
App which determines if on the picture is a cat and defines the breed of the cat. 
We use the dataset to train our machine learning and computer vision model for a project at university, 
which we then want to extend to further animals and later on extend this learning algorithm to identify if this specific animal 
was already seen once and categorize it. 
This will help wildforst and nationalparks to learn about their animals and paths they have taken at scale with the help of tourist pictures, 
which they can upload at a local datacenter / cloud. 
The rangers work should be reduced and automated so they can overview the paths and animals at scale 
"""

import yaml
import requests
import os
import typing 
import json
import piexif
from piexif import helper

#yaml and keys
def api_call_for_pictures(api_keys_file, name_of_save_file):
    skript_ordner = os.path.dirname(os.path.abspath(__file__))
    yaml_pfad = os.path.join(skript_ordner, api_keys_file)
    stream = open(yaml_pfad, "r")
    dictionary = yaml.safe_load(stream)
    page = 0

    headers = {"x-api-key": dictionary["cat_api"]["x_api_key"]}
    url = dictionary["cat_api"]["url"]
    
    #api call
    api_daten = [1]
    proceed = api_daten != []
    while proceed:
        params = {
            "has_breeds": 1,
            "page": page,
            "limit": 25,
        }
        response= requests.get(url, headers= headers, params=params)
        if response.status_code == 200:
            api_daten = response.json()
            if not api_daten:
                proceed = False
                break
            
            for entity in api_daten:
                save_pictures(entity, name_of_save_file)
            print(api_daten)
        else:
            print(f"Failed to grab data: {response.status_code}")
    
        page += 1

def save_pictures(api_daten: dict, ordner_name: str):
    ziel_ordner = ordner_name
    os.makedirs(ziel_ordner, exist_ok=True)

    if "breeds" in api_daten and api_daten["breeds"]:
        rasse_name = api_daten["breeds"][0]["name"].replace(" ", "_")
    else:
        rasse_name = "Unbekannt"

    dateiname_basis = f"katze_{rasse_name}_{api_daten['id']}"
    bild_pfad = os.path.join(ziel_ordner, f"{dateiname_basis}.jpg")
    bild_download_url = api_daten["url"]

    try:
        bild_resp = requests.get(bild_download_url, stream=True)
        if bild_resp.status_code == 200:
            with open(bild_pfad, "wb") as datei:
                for chunk in bild_resp.iter_content(1024):
                    datei.write(chunk)
            metadata_string = json.dumps(api_daten, ensure_ascii=False)

            user_comment = piexif.helper.UserComment.load(
                metadata_string, encoding="ascii"
            )

            exif_dict = {"Exif": {piexif.ExifIFD.UserComment: user_comment}}

            exif_bytes = piexif.dump(exif_dict)

            piexif.insert(exif_bytes, bild_pfad)

            print(
                f"✓ Bild mit eingebetteten EXIF-Metadaten gespeichert: {bild_pfad}"
            )

    except Exception as e:
        print(f"Fehler bei {bild_download_url}: {e}")


api_call_for_pictures(api_keys_file="api_keys.yaml", name_of_save_file="cat_pictures")

