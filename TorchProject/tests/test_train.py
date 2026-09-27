"""
Tests for the manual training loop (train_one_epoch, evaluate) using a
tiny synthetic dataset so each test runs in well under a second.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.model import FashionCNN
from src.train import train_one_epoch, evaluate, get_device


def _make_dummy_loader(num_samples: int = 32, batch_size: int = 8):
    images = torch.randn(num_samples, 1, 28, 28)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size)


def test_get_device_returns_valid_device():
    device = get_device()
    assert device.type in ("cpu", "cuda")


def test_train_one_epoch_returns_loss_and_accuracy_in_valid_range():
    model = FashionCNN()
    loader = _make_dummy_loader()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    loss, accuracy = train_one_epoch(model, loader, criterion, optimizer, device)

    assert loss >= 0, "Loss cannot be negative"
    assert 0.0 <= accuracy <= 1.0, "Accuracy must be a fraction between 0 and 1"


def test_evaluate_does_not_update_weights():
    """
    evaluate() must never change the model's weights — it should be a
    pure read-only pass (no backward, no optimizer.step()).
    """
    model = FashionCNN()
    loader = _make_dummy_loader()
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    param_before = next(model.parameters()).clone()
    evaluate(model, loader, criterion, device)
    param_after = next(model.parameters())

    assert torch.allclose(param_before, param_after), \
        "evaluate() must not modify model weights"


def test_evaluate_sets_model_to_eval_mode():
    """After evaluate() runs, the model should be left in eval mode."""
    model = FashionCNN()
    model.train()  # start in train mode on purpose
    loader = _make_dummy_loader()
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cpu")

    evaluate(model, loader, criterion, device)

    assert model.training is False, "evaluate() should leave the model in eval mode"


def test_train_one_epoch_sets_model_to_train_mode():
    """After train_one_epoch() runs, the model should be left in train mode."""
    model = FashionCNN()
    model.eval()  # start in eval mode on purpose
    loader = _make_dummy_loader()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    device = torch.device("cpu")

    train_one_epoch(model, loader, criterion, optimizer, device)

    assert model.training is True, "train_one_epoch() should leave the model in train mode"