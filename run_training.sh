#!/bin/bash
#SBATCH --job-name=catdogs_training         # Name des Jobs in der Queue
#SBATCH --output=training_output_%A_%a.log  # %A = Gesamt-Job-ID, %a = Array-Index (1-7)
#SBATCH --error=training_error_%A_%a.log    # Getrennte Fehler-Logs für jeden Run
#SBATCH --partition=NvidiaAll               # Nutzt die Nvidia-Grafikkarten-Pools der LMU
#SBATCH --array=1-7                         # Startet 7 Jobs parallel (Limit der LMU: max 15)
#SBATCH --time=48:00:00                     # Erhöht auf 2 Tage, um Time-Limits abzufangen!

# 1. In das richtige Projektverzeichnis wechseln
export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

cd /home/st/sternberg/CVDL26-drop-table-catdogs

source .venv/bin/activate

# 2. Hyperparameter je nach Slurm-Array-ID zuweisen
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    LR="0.00001"
    EPOCHS="100"
    FLAGS="--balance_weights"
    EXP_NAME="lr:0.00001 ep:100 30000DATA__balance_weights"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    LR="0.000001"
    EPOCHS="100"
    FLAGS="--balance_weights"
    EXP_NAME="lr:0.000001 ep:100 30000DATA__balance_weights"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    LR="0.0000001"
    EPOCHS="100"
    FLAGS="--balance_weights"
    EXP_NAME="lr:0.0000001 ep:100 30000DATA__balance_weights"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    LR="0.000001"
    EPOCHS="200"
    FLAGS="--balance_weights"
    EXP_NAME="lr:0.000001 ep:200 30000DATA__balance_weight"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    LR="0.000001"
    EPOCHS="1000"
    FLAGS="--mirror --blur --balance_weights"
    EXP_NAME="lr:0.000001 ep:1000 30000DATA_mirror_blurr_balance_weight"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
    LR="0.000001"
    EPOCHS="10000"
    FLAGS="--mirror --blur --cropmix --balance_weights" # <--- NEU: Hier testen wir alle 3 kombiniert!
    EXP_NAME="lr:0.000001 ep:10000 30000DATA_mirror_blur_cropmix_balance_weight"
fi

if [ $SLURM_ARRAY_TASK_ID -eq 7 ]; then
    LR="0.000005"
    EPOCHS="200"
    FLAGS="--cropmix --balance_weights"                 # <--- NEU: Hier testen wir CropMix exklusiv!
    EXP_NAME="lr:0.000005 ep:200 30000DATA_cropmix_balance_weight"
fi

# 3. Den eigentlichen Befehl ausführen
python3 -u Final_Project/train.py \
    --data_dir ./images \
    --lr $LR \
    --epochs $EPOCHS \
    --exp_name "$EXP_NAME" \
    $FLAGS