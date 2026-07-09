#!/bin/bash
#SBATCH --job-name=catdogs_training      # Name des Jobs in der Queue
#SBATCH --output=training_output_%j.log  # Hier landet der Terminal-Output (%j = Job-ID)
#SBATCH --error=training_error_%j.log   # Hier landen Fehlermeldungen
#SBATCH --partition=NvidiaAll           # Nutzt die Nvidia-Grafikkarten-Pools der LMU
#SBATCH --time=12:00:00                 # Maximale Laufzeit (Format: HH:MM:SS)

# 1. In das richtige Projektverzeichnis wechseln
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

cd /home/st/sternberg/CVDL26-drop-table-catdogs 

source .venv/bin/activate

# 2. Das Skript starten und den Pfad zu den hochgeladenen Bildern übergeben
#Möglichkeiten
# standard --mirror --blur
python3 -u Final_Project/train.py --data_dir ./images --lr 0.000005 --epochs 200 --exp_name "lr:0.000005 ep:200 30000DATA_blur" --blur
