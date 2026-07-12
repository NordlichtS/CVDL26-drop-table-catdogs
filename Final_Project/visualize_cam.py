import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Dein Modell importieren
from animalClassifier import AnimalClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Pfad zur .pth Datei')
    parser.add_argument('--image', type=str, required=True, help='Pfad zum Testbild')
    parser.add_argument('--species', type=str, default='cat', choices=['cat', 'dog'])
    parser.add_argument('--output', type=str, default='gradcam_result.jpg')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Modell laden und Gewichte wiederherstellen
    # Da wir NUM_CLASSES=10 haben
    model = AnimalClassifier(num_classes=10, device=device)
    
    print(f"Lade Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 2. Das Ziel-Layer für Grad-CAM definieren
    # WICHTIG: Grad-CAM braucht das ALLERLETZTE Convolutional Layer deines Backbones.
    # Richtig für dein EfficientNet-V2-S:
    target_layers = [model.backbone.features[-1]]

    # 3. Bild laden und für das Modell vorbereiten
    orig_img = Image.open(args.image).convert('RGB')
    
    # Für das Modell (Tensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    # Für die spätere Anzeige (im Bereich 0.0 - 1.0 als float32-numpy)
    rgb_img = orig_img.resize((224, 224))
    rgb_img = np.float32(rgb_img) / 255.0

    # 4. Grad-CAM initialisieren
    cam = GradCAM(model=model.backbone, target_layers=target_layers)
    # Wenn du wissen willst, worauf das Netz für eine bestimmte Klasse geschaut hat:
    # (Wir lassen es standardmäßig die Klasse mit der höchsten Vorhersage wählen)
    # 1. Kurz ohne Gradienten den höchsten Score ermitteln
    with torch.no_grad():
        output = model.backbone(input_tensor)
        pred_class = output.argmax(dim=-1).item()  # .item() erzwingt einen echten Python-Int!

    # 2. Den echten Integer an Grad-CAM übergeben
    targets = [ClassifierOutputTarget(pred_class)]
    print(f"[INFO] Generiere Grad-CAM für vorhergesagte Klasse Index: {pred_class}")

    # Heatmap generieren
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # Batch-Dimension entfernen

    # 5. Heatmap über das Originalbild legen
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 6. Ergebnis speichern (NEU: Automatischer Ordner und dynamischer Name)
    output_dir = "CAM_Heat_Map"
    os.makedirs(output_dir, exist_ok=True)

    counter = 1
    while True:
        filename = f"{counter:05d}.jpg"
        output_path = os.path.join(output_dir, filename)
        if not os.path.exists(output_path):
            break
        counter += 1

    # OpenCV nutzt standardmäßig BGR, daher konvertieren wir zu RGB zurück
    cv2.imwrite(output_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    print(f"[INFO] Ergebnis erfolgreich gespeichert unter: {output_path}")



if __name__ == "__main__":
    main()