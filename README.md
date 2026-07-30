# CVDL26 Drop Table Cat Dogs

Fine-grained animal recognition project for the CVDL SEP challenge.

## Overview

The current submission is built as a two-stage pipeline:

1. `Final_Project/detector.py` runs a YOLOv6s detector and selects the largest cat or dog in the image.
2. `Final_Project/animalClassifier.py` runs an EfficientNetV2-S breed classifier on the cropped animal.
3. `AnimalRecognitionChallenge/inference.py` wraps both stages for the official evaluation interface and returns a class index in `0..19` or `-1` for reject.

The challenge label order is fixed in `AnimalRecognitionChallenge/inference.py`:

- `0..9` = cat breeds
- `10..19` = dog breeds
- `-1` = no target animal / reject

## Technology Stack

- Python
- PyTorch and TorchVision
- YOLOv6 via `torch.hub`
- EfficientNetV2-S for breed classification
- OpenCV, NumPy, Pillow
- pandas, scikit-learn, tqdm
- Grad-CAM for explainable AI visualizations
- Slurm batch scripts for training on the LMU cluster

## Repository Layout

```text
CVDL26-drop-table-catdogs/
|-- AnimalRecognitionChallenge/
|   `-- inference.py              # Official evaluation entry point
|-- Final_Project/
|   |-- detector.py               # YOLOv6 crop selection
|   |-- animalClassifier.py       # EfficientNetV2-S breed classifier
|   |-- train.py                  # Current crop-based training pipeline
|   |-- traineffnet.py            # Alternate ImageFolder training baseline
|   |-- evaluate.py               # Checkpoint inspection helper
|   |-- compare_classifier.py     # ResNet18 diagnostic comparator
|   |-- visualize_cam.py          # Grad-CAM heatmap generation
|   |-- xAI_GradCam.py            # Legacy / experimental XAI helper
|   |-- sorting_folder_images.py  # Merge class folders into flat images/
|   `-- download_model_detector.py # Download YOLOv6s weights
|-- Training_Material/
|   |-- cat_api.py                # TheCatAPI data collection helper
|   |-- kagel_dataset.py          # Kaggle cat-breed collection helper
|   |-- kaggle_dog.py             # Stanford dogs collection helper
|   |-- oxford_loader.py          # Oxford-IIIT Pet collection helper
|   |-- extended_dogs_loader.py   # Stanford + local Dalmatian integration
|   `-- jan_dalamtian.py          # Local Dalmatian import helper
|-- notebooks/
|   |-- 01_train_combined.ipynb
|   `-- 02_train_species_split.ipynb
|-- Final_Report/                # Final report LaTeX project
|-- Preliminary_Report/          # Early report LaTeX project
|-- LaTeXAuthor Guidelines for CVDL SEP Report/
|-- run_training.sh              # Slurm training launcher
|-- run_visualization_cam.sh     # Batch Grad-CAM launcher
|-- requirements.txt
|-- yolov6s.pt                   # Detector weights
`-- README.md
```

## How the Project Works

### Inference Flow

`AnimalRecognitionChallenge/inference.py` is the submission-facing script.

```text
input image
  -> YOLOv6 detector
  -> largest detected cat/dog crop
  -> 224x224 resize
  -> EfficientNetV2-S breed classifier
  -> predicted breed index or -1
```

The detector and classifier are separated so the classifier sees a standardized crop instead of the full background.

### Training Flow

The active training path is `Final_Project/train.py`.

- It builds a cropped-animal dataset from `images/labels.csv`.
- It caches detector crops to avoid repeated YOLOv6 calls.
- It supports optional augmentation flags:
  - `--mirror`
  - `--blur`
  - `--cropmix`
- It supports optional class reweighting with `--balance_weights`.
- It trains one classifier for cats and one classifier for dogs.
- It saves checkpoints as `cat_scratch_<exp>.pth` and `dog_scratch_<exp>.pth`.

`Final_Project/traineffnet.py` is an earlier alternate experiment that trains a single 20-class classifier from an `ImageFolder` layout with weighted sampling and ImageNet initialization.

### Data Preparation

The helper scripts in `Training_Material/` document how the dataset was assembled.

- `cat_api.py` was used for early TheCatAPI collection.
- `kagel_dataset.py`, `kaggle_dog.py`, and `oxford_loader.py` cover the main curated sources used later in the project.
- `extended_dogs_loader.py` adds missing dog breeds and local Dalmatian images.
- `jan_dalamtian.py` handles the Dalmatian-only local import.

The report materials describe a project history that started with API-based collection, then moved toward larger curated sources and additional breed-specific patches to improve coverage. The current repository keeps the loaders and merge helpers used for that workflow.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure the data folder expected by the training scripts exists:

```text
images/
|-- labels.csv
|-- 00000.jpg
|-- 00001.jpg
`-- ...
```

4. If you want to use the folder-based helpers, also prepare a `classes/` directory with one folder per breed.

## Running Training

The provided Slurm launcher is `run_training.sh`.

```bash
sbatch run_training.sh
```

Notes:

- The script is written for the LMU cluster and contains a hardcoded project path.
- The current array jobs sweep several learning-rate, epoch, and augmentation combinations.
- Adjust the working directory and virtual environment path if you run it elsewhere.

To run the training script directly:

```bash
python Final_Project/train.py --data_dir ./images --lr 1e-6 --epochs 100 --exp_name demo --balance_weights
```

Add `--mirror`, `--blur`, or `--cropmix` as needed.

## Running Inference

The challenge entry point is:

```bash
python AnimalRecognitionChallenge/inference.py --image-folder <path-to-test-images>
```

Expected local files:

- `yolov6s.pt` for the detector
- `cat_scratch.pth` and `dog_scratch.pth` for the breed classifiers

If `yolov6s.pt` is missing, `Final_Project/download_model_detector.py` can download it from the official YOLOv6 release.

## XAI / Visualization

`Final_Project/visualize_cam.py` generates Grad-CAM heatmaps for a chosen image and checkpoint.

Example:

```bash
python Final_Project/visualize_cam.py --checkpoint <path-to-checkpoint> --image <image-path> --species cat
```

The batch helper `run_visualization_cam.sh` repeatedly generates heatmaps for random images.

## Notes For The Final Report

- `Final_Report/` contains the LaTeX source for the final paper.
- `Preliminary_Report/` keeps the earlier report version.
- The final scientific report should describe the full pipeline, the experiments, the XAI analysis, and the team contribution appendix.

## Remaining Cleanup Before Final Packaging

- Remove or clearly quarantine legacy and debug code paths in the submission-facing scripts, especially any leftover comparison or placeholder branches.
- Keep only the final training and inference paths in the shipped archive, or label backup scripts as archival so they are not mistaken for the primary workflow.
- Verify the final submission still runs from a clean checkout with the documented weight files and no hidden local-path assumptions.
- Keep a source URL list for any images collected from the internet, and record the exact preprocessing commands used to build `images/`.
