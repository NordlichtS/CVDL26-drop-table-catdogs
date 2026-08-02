Here is the complete, fully merged **`README.md`** translated into English and with all emojis removed:

---

# CVDL26: Drop Table Cat Dogs

> **Fine-Grained Animal Recognition System (Cat and Dog Breed Recognition Cascade)** > *Developed for the CVDL SEP Challenge (LMU Munich)*

---

## Overview

This project implements a two-stage deep learning pipeline (cascade) for fine-grained recognition and classification of cat and dog breeds:

1. **Detection:** [`Final_Project/detector.py`](https://www.google.com/search?q=./Final_Project/detector.py) utilizes a **YOLOv6** model to locate the largest animal (cat or dog) in the image and extract a precise crop.
2. **Classification:** [`Final_Project/animalClassifier.py`](https://www.google.com/search?q=./Final_Project/animalClassifier.py) processes the cropped animal image with an **EfficientNetV2-S** model to predict the exact breed.
3. **Inference Pipeline:** [`AnimalRecognitionChallenge/inference.py`](https://www.google.com/search?q=./AnimalRecognitionChallenge/inference.py) combines both stages for the official evaluation interface and returns a class index.

By decoupling the detector and classifier, the classification model receives a standardized, centered animal crop rather than the entire, potentially distracting background.

### Label Mapping

The mapping of label IDs is fixed in `AnimalRecognitionChallenge/inference.py`:

| Label ID | Category | Description |
| --- | --- | --- |
| `0 .. 9` | **Cats** | Breed classes `0` to `9` |
| `10 .. 19` | **Dogs** | Breed classes `10` to `19` |
| `-1` | **Reject** | No target animal detected / Rejection |

---

## Technology Stack

* **Language:** Python
* **Deep Learning Frameworks:** PyTorch, TorchVision
* **Detection:** YOLOv6 via `torch.hub`
* **Classification:** EfficientNetV2-S
* **Image Processing & Data:** OpenCV, NumPy, Pillow, pandas, scikit-learn, tqdm
* **Explainable AI (xAI):** Grad-CAM
* **HPC / Cluster:** Slurm Batch Scripts (LMU Cluster)

---

## Repository Structure

```text
CVDL26-drop-table-catdogs/
├── AnimalRecognitionChallenge/
│   └── inference.py              # Official evaluation entry point
├── Final_Project/
│   ├── detector.py               # YOLOv6 bounding box selection
│   ├── animalClassifier.py       # EfficientNetV2-S breed classifier
│   ├── train.py                  # Active crop-based training pipeline
│   ├── traineffnet.py            # Alternate ImageFolder baseline trainer
│   ├── evaluate.py               # Helper script for checkpoint inspection
│   ├── compare_classifier.py     # Diagnostic comparison with ResNet18
│   ├── visualize_cam.py          # Grad-CAM heatmap generation
│   ├── xAI_GradCam.py            # Experimental / legacy XAI helper
│   ├── generate_xAI_panel.py     # Batch/panel generation for Grad-CAM
│   ├── sorting_folder_images.py  # Merge class folders into flat images/
│   └── download_model_detector.py # Download script for YOLOv6s weights
├── Training_Material/
│   ├── cat_api.py                # Data collection via TheCatAPI
│   ├── kagel_dataset.py          # Kaggle cat-breeds downloader
│   ├── kaggle_dog.py             # Stanford dogs downloader
│   ├── oxford_loader.py          # Oxford-IIIT Pet Dataset integration
│   ├── extended_dogs_loader.py   # Stanford + Dalmatian integration
│   └── jan_dalamtian.py          # Local Dalmatian importer
├── notebooks/
│   ├── 01_train_combined.ipynb
│   └── 02_train_species_split.ipynb
├── Final_Report/                 # LaTeX project: Final report
├── Preliminary_Report/           # LaTeX project: Intermediate report
├── LaTeXAuthor Guidelines.../    # LMU guidelines for reports
├── run_training.sh               # Slurm training launcher
├── run_visualization_cam.sh      # Slurm Grad-CAM batch launcher
├── requirements.txt              # Python dependencies
├── yolov6s.pt                    # Detector weights
└── README.md

```

---

## How the Project Works

### Inference Flow

```text
[ Input Image ]
       │
       ▼
[ YOLOv6 Detector ]  ──►  (Selects largest detected animal)
       │
       ▼
[ Crop & Resize (224x224) ]
       │
       ▼
[ EfficientNetV2-S ]
       │
       ▼
[ Output: Breed Index (0-19) or -1 ]

```

### Training Flow

The active training pipeline is controlled via **`Final_Project/train.py`**:

* Constructs datasets from cropped animal images based on `images/labels.csv`.
* Caches YOLOv6 crops to avoid repeated detector calls.
* **Augmentations:** Supports optional flag parameters such as `--mirror`, `--blur`, and `--cropmix`.
* **Class Reweighting:** Flag `--balance_weights` compensates for class imbalances.
* Trains **two separate classifiers** (one specifically for cats, one for dogs).
* Saves checkpoints as `cat_scratch_<exp>.pth` and `dog_scratch_<exp>.pth`.

> *Alternate Baseline:* `Final_Project/traineffnet.py` represents an earlier experiment that trains a single 20-class classifier from an `ImageFolder` structure with weighted sampling and ImageNet pretraining.

---

## Data Preparation & Loaders

The helper scripts in `Training_Material/` document the step-by-step assembly of the training dataset:

* **`cat_api.py`:** Early data collection via *TheCatAPI*.
* **`oxford_loader.py`:** Integration of the *Oxford-IIIT Pet Dataset*.
* **`kagel_dataset.py`:** Integration of the *Kaggle Cat-Breeds Dataset* via `kagglehub`.
* **`kaggle_dog.py` & `extended_dogs_loader.py`:** Integration of the *Stanford Dogs Dataset* along with local additions for missing breeds.
* **`jan_dalamtian.py`:** Imports specific local Dalmatian image data.

> **History:** The project started with API-based downloads and expanded to larger curated sources with breed-specific additions. All loaders utilize a `get_next_counter` logic for sequential file naming (`00000.jpg`, `00001.jpg`, ...) to prevent data loss.

---

## Optional and Disabled Experiments

The repository retains experimental alternatives that are **disabled** in the submitted inference path:

* **Secondary-box zero masking:** Zero-masking of secondary bounding boxes (commented out in `Final_Project/detector.py:131-146`). Can be re-enabled manually for comparisons, but masks all non-selected boxes without overlap protection. The submitted path uses an unmasked exact-box crop instead.
* **Unified 20-class classifier:** `Final_Project/traineffnet.py` is a separate experiment with pretraining. It must not be confused with the species-routed submission model.
* **Pretrained comparator:** `Final_Project/compare_classifier.py` implements a ResNet18 diagnostic comparator (integration in `inference.py` is commented out).
* **Confusion matrix output:** The local inference harness imports the metric, but the output print statement is commented out.
* **Grad-CAM utilities:** Provided via `generate_xAI_panel.py` and `visualize_cam.py` for offline diagnostics without affecting predictions.

> *Note:* Adaptive box padding, a 40% masking bypass rule, rotation augmentation, perceptual hash groupings, and a train-validation gap stopping rule are not implemented in the active code.

---

## Reproducibility Status & Limitations

* Detector weights (`yolov6s.pt`) are included, but the required classifier checkpoints for cats and dogs are absent from the standard checkout.
* The exact final training corpus, source URL manifests, split manifests, raw training logs, prediction files, and end-to-end benchmark logs are not included in the repository.
* The active split uses unseeded `random.shuffle`: repeated training runs will therefore use slightly different data distributions.

---

## Setup & Installation

1. **Create & activate a virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```


3. **Prepare directory structure:**
Ensure the training data is structured as follows:
```text
images/
├── labels.csv
├── 00000.jpg
├── 00001.jpg
└── ...

```


*Optional:* If using folder-based helper scripts, prepare a `classes/` directory containing one subfolder per breed.

---

## Execution

### Start Training

**Directly via Python:**

```bash
python Final_Project/train.py \
    --data_dir ./images \
    --lr 1e-6 \
    --epochs 100 \
    --exp_name demo \
    --balance_weights \
    --mirror --blur --cropmix

```

**On the LMU Cluster (Slurm):**

```bash
sbatch run_training.sh

```

*Notes regarding the Slurm script:*

* The script is configured for the LMU cluster and contains project-specific paths.
* Executes array jobs sweeping across various learning rates, epochs, and augmentation combinations.

### Run Inference

Run the evaluation script on a folder containing test images:

```bash
python AnimalRecognitionChallenge/inference.py --image-folder <path-to-test-images>

```

**Required files in the execution directory:**

* `yolov6s.pt` (Detector weights; can be downloaded via `Final_Project/download_model_detector.py`)
* `cat_scratch.pth` & `dog_scratch.pth` (Classifier weights)

---

## xAI & Visualization (Grad-CAM)

Generate Grad-CAM heatmaps to explain model decisions:

```bash
python Final_Project/visualize_cam.py \
    --checkpoint <path-to-checkpoint> \
    --image <image-path> \
    --species cat

```

For automated batch processing, the Slurm script `run_visualization_cam.sh` is available.

---
