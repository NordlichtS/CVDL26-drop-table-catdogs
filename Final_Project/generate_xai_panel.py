import os
import argparse
import csv
import random  # <-- NEU: Für die Zufallsauswahl
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Versuche AnimalClassifier & CLASSES aus animalClassifier.py zu laden
try:
    from animalClassifier import AnimalClassifier, CLASSES
    print("[INFO] 'CLASSES' erfolgreich aus animalClassifier.py importiert.")
except ImportError:
    from animalClassifier import AnimalClassifier
    CLASSES = [
        "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
        "Maine_Coon", "Ragdoll", "Sphynx", "Tabby", "Tiger_Cat",
        "Beagle", "Pug", "Boxer", "Shiba_Inu", "Samoyed",
        "Golden_Retriever", "German_Shepherd", "Siberian_Husky", "Dalmatian", "Rottweiler"
    ]

def detect_num_classes_from_checkpoint(checkpoint_path):
    """Ermittelt die Ausgabeklassen (10 oder 20) aus der .pth Datei."""
    if not os.path.exists(checkpoint_path):
        return 10
    
    try:
        cp = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = cp["model_state"] if isinstance(cp, dict) and "model_state" in cp else cp
        for k, v in sd.items():
            if "classifier" in k and "weight" in k:
                return v.shape[0]
    except Exception:
        pass
    
    return 10

def get_ground_truth(csv_path, image_name, classes_list):
    """Liest das Label aus labels.csv für ein bestimmtes Bild aus."""
    if not os.path.exists(csv_path):
        alt_path = os.path.join("images", os.path.basename(csv_path))
        if os.path.exists(alt_path):
            csv_path = alt_path
        else:
            return "Unknown"

    base_name = os.path.basename(image_name)
    name_no_ext = os.path.splitext(base_name)[0]
    clean_id = name_no_ext.lstrip("0") or "0"

    with open(csv_path, mode='r', encoding='utf-8') as f:
        first_line = f.readline()
        f.seek(0)
        
        delimiter = ';' if ';' in first_line else (',' if ',' in first_line else '\t')
        reader = csv.reader(f, delimiter=delimiter)

        for line_num, row in enumerate(reader):
            if not row or len(row) < 2:
                continue

            col_0 = row[0].strip().strip('"\'')
            col_1 = row[1].strip().strip('"\'')

            if line_num == 0 and ("filename" in col_0.lower() or "image" in col_0.lower() or "id" in col_0.lower() or "label" in col_0.lower()):
                continue

            csv_file_name = os.path.basename(col_0)
            csv_file_no_ext = os.path.splitext(csv_file_name)[0]
            csv_clean_id = csv_file_no_ext.lstrip("0") or "0"

            if base_name == col_0 or name_no_ext == csv_file_no_ext or clean_id == csv_clean_id:
                try:
                    class_idx = int(col_1)
                    if 0 <= class_idx < len(classes_list):
                        class_name = classes_list[class_idx]
                        print(f"[INFO] CSV-Treffer: ID '{col_0}' -> Index {class_idx} ({class_name})")
                        return class_name
                    else:
                        return f"Class_{class_idx}"
                except ValueError:
                    return col_1

    return "Unknown"

def draw_header_banner(img_bgr, text):
    """Fügt eine schwarze Titelleiste oberhalb des Bildes ein."""
    h, w, _ = img_bgr.shape
    banner = np.zeros((35, w, 3), dtype=np.uint8)
    cv2.putText(
        banner, text, (8, 23), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
    )
    return np.vstack((banner, img_bgr))

def main():
    parser = argparse.ArgumentParser(description="Erzeugt Live-GradCAM 3-Panel-Visualisierung")
    
    parser.add_argument('--checkpoint', type=str, default='dog_scratch.pth', help='Pfad zur .pth Datei')
    parser.add_argument('--images_dir', type=str, default='images', help='Ordner mit den Originalbildern')
    # Default ist jetzt None -> dadurch wird automatisch ein Zufallsbild gewählt:
    parser.add_argument('--image', type=str, default=None, help='Dateiname (leer lassen für Zufallsbild)')
    parser.add_argument('--csv', type=str, default='labels.csv', help='Pfad zur labels.csv')
    parser.add_argument('--output_dir', type=str, default='CAM_Heat_Map', help='Zielordner')
    
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. BILD-PFAD ERMITTELN (ZUFALLS-BILD WÄHLEN)
    image_path = None
    
    # Falls gezielt ein Bild übergeben wurde:
    if args.image:
        if os.path.exists(args.image):
            image_path = args.image
        elif os.path.exists(os.path.join(args.images_dir, args.image)):
            image_path = os.path.join(args.images_dir, args.image)
            
    # Falls kein Bild übergeben wurde (oder Pfad nicht existiert): Wähle zufällig eins aus images/
    if image_path is None:
        if os.path.exists(args.images_dir):
            valid_imgs = [f for f in os.listdir(args.images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if valid_imgs:
                selected_img = random.choice(valid_imgs)
                image_path = os.path.join(args.images_dir, selected_img)
                print(f"[INFO] Zufällig ausgewähltes Bild: '{selected_img}'")
            else:
                raise FileNotFoundError(f"Keine Bilder im Ordner '{args.images_dir}' gefunden!")
        else:
            raise FileNotFoundError(f"Ordner '{args.images_dir}' existiert nicht!")

    gt_label = get_ground_truth(args.csv, os.path.basename(image_path), CLASSES)
    print(f"[1/5] Bild geladen: {image_path} | Ground Truth: {gt_label}")

    # 2. AUTOMATISCHE ERKENNUNG DER KLASSENANZAHL
    num_classes = detect_num_classes_from_checkpoint(args.checkpoint)

    if num_classes == 10:
        if "dog" in args.checkpoint.lower():
            active_classes = CLASSES[10:]
        else:
            active_classes = CLASSES[:10]
    else:
        active_classes = CLASSES

    print(f"[2/5] Initialisiere AnimalClassifier (num_classes={num_classes}) mit '{args.checkpoint}'...")
    model = AnimalClassifier(weights_path=args.checkpoint, num_classes=num_classes, device=device)
    model.eval()

    target_layers = [model.backbone.features[-1]]

    # 3. BILD TRANSFORMIEREN
    orig_pil = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(orig_pil).unsqueeze(0).to(device)
    rgb_np = np.float32(orig_pil.resize((224, 224))) / 255.0

    # 4. INFERENZ & GRAD-CAM
    print("[3/5] Berechne Inferenz & Grad-CAM...")
    with torch.no_grad():
        logits = model.backbone(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_class_idx = logits.argmax(dim=-1).item()
        confidence = probs[0, pred_class_idx].item() * 100

    pred_name = active_classes[pred_class_idx] if pred_class_idx < len(active_classes) else f"Class_{pred_class_idx}"

    cam = GradCAM(model=model.backbone, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_class_idx)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    # 5. DIE 3 PANELS ERSTELLEN
    print("[4/5] Erstelle 3-Panel Grid...")
    
    orig_bgr = cv2.cvtColor((rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    panel_a = draw_header_banner(orig_bgr, f"GT: {gt_label}")

    heatmap_bgr = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    panel_b = draw_header_banner(heatmap_bgr, "Grad-CAM Heatmap")

    overlay_rgb = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    panel_c = draw_header_banner(overlay_bgr, f"Pred: {pred_name} ({confidence:.1f}%)")

    grid_result = np.hstack((panel_a, panel_b, panel_c))

    # 6. SPEICHERN
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = f"panel_{os.path.basename(image_path)}"
    output_path = os.path.join(args.output_dir, out_file)
    
    cv2.imwrite(output_path, grid_result)
    print(f"[5/5] ERFOLG! Ergebnis gespeichert unter: {output_path}")

if __name__ == "__main__":
    main()