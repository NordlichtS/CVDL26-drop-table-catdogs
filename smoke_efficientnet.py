"""CPU-only smoke test for the local EfficientNetV2-S implementation."""

import torch

from Final_Project.models.efficientnetv2_s import EfficientNetV2S
from AnimalRecognitionChallenge.inference import predict_efficientnet_crop


def main() -> None:
    """Run a minimal forward pass and print output shapes."""

    device = "cpu"
    model = EfficientNetV2S(num_classes=20, device=device)

    # Match the resized crop layout used by the project: C x H x W.
    crop_tensor = torch.rand(3, 224, 224, device=device)
    probs, predicted_label, feature_map = predict_efficientnet_crop(crop_tensor, model, device=device)

    top_probs, top_indices = torch.topk(probs[0], k=3)
    print(f"crop_tensor.shape = {tuple(crop_tensor.shape)}")
    print(f"probs.shape = {tuple(probs.shape)}")
    print(f"predicted_label = {predicted_label}")
    print(f"feature_map.shape = {tuple(feature_map.shape) if feature_map is not None else None}")
    print("top-3 probabilities:")
    for rank, (probability, index) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        print(f"  {rank}: class={index}, prob={probability:.4f}")


if __name__ == "__main__":
    main()