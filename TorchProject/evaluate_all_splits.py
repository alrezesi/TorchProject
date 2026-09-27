"""
One-off utility: compute Accuracy, Precision, Recall, F1, and confusion
matrix for the train, validation, and test splits using an already
trained checkpoint. Not part of the core CLI (train.py/evaluate.py) —
useful for reporting/README purposes.

Usage:
    python evaluate_all_splits.py --checkpoint checkpoints/best_fashion_mnist.pth
"""

import argparse

from src.dataset import get_dataloaders
from src.model import FashionCNN
from src.train import get_device
from src.utils import load_checkpoint
from src.evaluate import evaluate_full, print_evaluation_report


def main():
    parser = argparse.ArgumentParser(description="Evaluate FashionCNN on train/val/test splits")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_fashion_mnist.pth")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}\n")

    train_loader, val_loader, test_loader = get_dataloaders()

    model = FashionCNN()
    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    splits = {
        "TRAIN (note: includes augmentation, so slightly pessimistic vs. training-time reporting)": train_loader,
        "VALIDATION": val_loader,
        "TEST": test_loader,
    }

    for split_name, loader in splits.items():
        print("=" * 70)
        print(f"Split: {split_name}")
        print("=" * 70)
        metrics = evaluate_full(model, loader, device)
        print_evaluation_report(metrics)
        print()


if __name__ == "__main__":
    main()