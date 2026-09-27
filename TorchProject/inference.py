"""
Entry point: classify a single Fashion-MNIST-style image using a
trained checkpoint.

Usage:
    python inference.py --image path/to/image.png
    python inference.py --image path/to/image.png --checkpoint checkpoints/best_fashion_mnist.pth
"""

import argparse

import torch
from PIL import Image

from src.dataset import get_eval_transform, CLASS_NAMES
from src.model import FashionCNN
from src.train import get_device
from src.utils import load_checkpoint


def load_image(image_path: str) -> torch.Tensor:
    """
    Load an image from disk and apply the SAME preprocessing used for
    validation/test data during training (grayscale conversion, resize
    to 28x28, normalize with the training mean/std). Using a different
    preprocessing here would feed the model data unlike anything it was
    trained on, silently producing meaningless predictions.
    """
    image = Image.open(image_path).convert("L")   # "L" = single-channel grayscale
    image = image.resize((28, 28))

    transform = get_eval_transform()
    tensor = transform(image)          # shape: (1, 28, 28)
    tensor = tensor.unsqueeze(0)       # add batch dimension -> (1, 1, 28, 28)
    return tensor


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device) -> tuple[str, float]:
    """
    Runs a single forward pass and returns the predicted class name
    and its confidence (softmax probability).

    model.eval() and torch.no_grad() matter here for the same reasons
    they matter during validation: Dropout/BatchNorm must behave
    deterministically, and no gradient graph is needed since we're not
    calling backward().
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)

    confidence, predicted_idx = probabilities.max(dim=1)
    predicted_class = CLASS_NAMES[predicted_idx.item()]

    return predicted_class, confidence.item()


def main():
    parser = argparse.ArgumentParser(description="Classify a single image with FashionCNN")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_fashion_mnist.pth")
    args = parser.parse_args()

    device = get_device()

    model = FashionCNN()
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    image_tensor = load_image(args.image)
    predicted_class, confidence = predict(model, image_tensor, device)

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    main()