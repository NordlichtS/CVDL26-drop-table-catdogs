#!/bin/bash

# 1. In dein Projektverzeichnis wechseln
cd /home/st/sternberg/CVDL26-drop-table-catdogs

# 2. Virtuelle Umgebung aktivieren
source .venv/bin/activate

# Schleife von 1 bis 20 starten
for i in {1..20}
do
    echo "--------------------------------------------------"
    echo "[Fortschritt] Generiere Bild $i von 20..."
    
    # 3. Ein ZUFÄLLIGES Bild auswählen
    RANDOM_IMAGE=$(ls ./images/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | shuf -n 1)

    if [ -z "$RANDOM_IMAGE" ]; then
        echo "[FEHLER] Keine Bilder im Ordner ./images gefunden!"
        exit 1
    fi

    # 4. Den Befehl ausführen
    python3 ./Final_Project/visualize_cam.py \
        --checkpoint "./cat_scratch_lr:0.000001 ep:100 30000DATA.pth" \
        --image "$RANDOM_IMAGE" \
        --species cat
done

echo "--------------------------------------------------"
echo "[FERTIG] 20 Heatmaps wurden erfolgreich erstellt!"