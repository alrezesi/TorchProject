"""
Plot generation for training history: loss curves and metric-vs-epoch,
saved as PNG files under reports/ for the README and for review.
"""

import matplotlib.pyplot as plt


def plot_loss_curves(history: dict, save_path: str = "reports/loss_curve.png") -> None:
    """
    Plots train vs validation loss across epochs on the same axes.
    A widening gap between the two lines (train dropping, val flat or
    rising) is the visual signature of overfitting.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()   # frees memory — important if plotting in a loop or notebook
    print(f"Saved loss curve to {save_path}")


def plot_metric_curve(history: dict, save_path: str = "reports/accuracy_curve.png") -> None:
    """
    Plots validation accuracy across epochs, to see whether the model
    is still improving, has plateaued, or is oscillating (a sign the
    learning rate may be too high).
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Train Accuracy")
    plt.plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved accuracy curve to {save_path}")