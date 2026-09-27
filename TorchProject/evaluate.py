"""
Entry point: load the best checkpoint and run full test-set evaluation.

Usage:
    python evaluate.py
    python evaluate.py --checkpoint checkpoints/best_fashion_mnist.pth
"""

import argparse

import torch

from src.dataset import get_dataloaders
from src.model import FashionCNN
from src.train import get_device
from src.utils import load_checkpoint
from src.evaluate import evaluate_full, print_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate FashionCNN on the test set")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_fashion_mnist.pth",
        help="Path to the checkpoint to evaluate"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    _, _, test_loader = get_dataloaders()

    model = FashionCNN()
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    metrics = evaluate_full(model, test_loader, device)
    print_evaluation_report(metrics)


if __name__ == "__main__":
    main()