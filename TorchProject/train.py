"""
Entry point: train FashionCNN from scratch (or resume from a checkpoint)
and save loss/accuracy plots at the end.

Usage:
    python train.py
    python train.py --epochs 50 --resume checkpoints/best_fashion_mnist.pth
"""

import argparse

from src.dataset import get_dataloaders
from src.model import FashionCNN
from src.train import run_training, get_device
from src.visualize import plot_loss_curves, plot_metric_curve


def main():
    parser = argparse.ArgumentParser(description="Train FashionCNN on Fashion-MNIST")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_fashion_mnist.pth")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint to resume training from")
    args = parser.parse_args()

    device = get_device()
    train_loader, val_loader, test_loader = get_dataloaders()
    model = FashionCNN()

    model, history = run_training(
        model, train_loader, val_loader,
        num_epochs=args.epochs, lr=args.lr,
        checkpoint_path=args.checkpoint,
        device=device,
        resume_from=args.resume,
    )

    plot_loss_curves(history)
    plot_metric_curve(history)


if __name__ == "__main__":
    main()