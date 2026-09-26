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
                  device: torch.device = None):

    if device is None:
        device = get_device()

    print(f"Using device: {device}")
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Step the scheduler once per epoch (not per batch) for CosineAnnealingLR
        scheduler.step()

        epoch_time = time.time() - start_time

        print(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% "
            f"time={epoch_time:.1f}s"
        )

        # Save the best model based on validation accuracy
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved (val_acc={best_accuracy*100:.2f}%)")

    print(f"\nBest validation accuracy: {best_accuracy*100:.2f}%")
    return model


if __name__ == "__main__":
    from src.dataset import get_dataloaders
    from src.model import FashionCNN

    train_loader, val_loader, test_loader = get_dataloaders()
    model = FashionCNN()

    # Quick smoke test first: 2 epochs to confirm everything runs on CPU
    # before committing to a long 30-epoch run.
    run_training(model, train_loader, val_loader, num_epochs=2)