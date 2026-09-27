"""
Training loop for FashionCNN on Fashion-MNIST.

Includes: AdamW optimizer with weight decay, label smoothing,
cosine-annealing LR schedule, and best-checkpoint saving.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import save_checkpoint, load_checkpoint


def get_device() -> torch.device:
    """Detect the best available device. Falls back to CPU automatically."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> tuple[float, float]:
    """Runs one training epoch. Returns (avg_loss, accuracy)."""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, dataloader, criterion, device) -> tuple[float, float]:
    """Runs evaluation without updating weights. Returns (avg_loss, accuracy)."""
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def run_training(model, train_loader, val_loader, num_epochs: int = 30,
                  lr: float = 1e-3, weight_decay: float = 1e-4,
                  label_smoothing: float = 0.05,
                  checkpoint_path: str = "checkpoints/best_fashion_mnist.pth",
                  last_checkpoint_path: str = "checkpoints/last_fashion_mnist.pth",
                  device: torch.device = None,
                  resume_from: str = None):
    """
    checkpoint_path: saved ONLY when validation accuracy improves — this
                     is the checkpoint to use for evaluation/inference.
    last_checkpoint_path: saved EVERY epoch, regardless of improvement —
                     this is the checkpoint to resume training from, so
                     a crash never loses more than the current epoch.
    """
    if device is None:
        device = get_device()

    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    start_epoch = 1
    best_accuracy = 0.0
    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "epoch_time": [],
    }

    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from, model, optimizer, scheduler, device=device)
        start_epoch = checkpoint["epoch"] + 1
        best_accuracy = checkpoint["best_metric"]
        if "history" in checkpoint and checkpoint["history"] is not None:
            history = checkpoint["history"]
        print(f"Resumed from epoch {checkpoint['epoch']} (best_acc so far: {best_accuracy * 100:.2f}%)")

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    Path(last_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        epoch_time = time.time() - start_time

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epoch_time"].append(epoch_time)

        print(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}% "
            f"time={epoch_time:.1f}s"
        )

        # Always save the latest state — this is what resume should load,
        # so no epoch's progress is ever lost even if val_acc didn't improve.
        save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy,
                         last_checkpoint_path, history=history)

        # Additionally save a separate "best" checkpoint only on improvement —
        # this is the one used for evaluation/inference.
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy,
                             checkpoint_path, history=history)
            print(f"  -> New best model saved (val_acc={best_accuracy * 100:.2f}%)")

    print(f"\nBest validation accuracy: {best_accuracy * 100:.2f}%")
    return model, history


if __name__ == "__main__":
    from src.dataset import get_dataloaders
    from src.model import FashionCNN

    train_loader, val_loader, test_loader = get_dataloaders()
    model = FashionCNN()

    # Quick smoke test first: 2 epochs to confirm everything runs on CPU
    # before committing to a long 30-epoch run.
    run_training(model, train_loader, val_loader, num_epochs=2)