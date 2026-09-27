"""
Tests for checkpoint save/load and training resume behavior.
Uses a tiny synthetic dataset so this runs in seconds, not minutes.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.model import FashionCNN
from src.train import run_training
from src.utils import save_checkpoint, load_checkpoint


def _make_dummy_loaders():
    images = torch.randn(32, 1, 28, 28)
    labels = torch.randint(0, 10, (32,))
    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=8)
    return loader, loader


def test_last_checkpoint_always_reflects_latest_epoch(tmp_path):
    """
    The 'last' checkpoint must be updated every epoch, regardless of
    whether validation accuracy improved — this is what makes resume
    safe even during a streak of non-improving epochs.
    """
    train_loader, val_loader = _make_dummy_loaders()
    best_path = str(tmp_path / "best.pth")
    last_path = str(tmp_path / "last.pth")

    model = FashionCNN()
    run_training(model, train_loader, val_loader, num_epochs=2,
                 checkpoint_path=best_path, last_checkpoint_path=last_path,
                 device=torch.device("cpu"))

    last_checkpoint = torch.load(last_path, map_location="cpu")
    assert last_checkpoint["epoch"] == 2, \
        "The 'last' checkpoint must always reflect the final epoch trained"


def test_resume_continues_from_last_checkpoint(tmp_path):
    train_loader, val_loader = _make_dummy_loaders()
    best_path = str(tmp_path / "best.pth")
    last_path = str(tmp_path / "last.pth")

    model1 = FashionCNN()
    run_training(model1, train_loader, val_loader, num_epochs=2,
                 checkpoint_path=best_path, last_checkpoint_path=last_path,
                 device=torch.device("cpu"))

    model2 = FashionCNN()
    _, history = run_training(model2, train_loader, val_loader, num_epochs=4,
                               checkpoint_path=best_path, last_checkpoint_path=last_path,
                               device=torch.device("cpu"), resume_from=last_path)

    assert history["epoch"] == [1, 2, 3, 4], \
        "History should contain all 4 epochs, not just the ones since resume"


def test_load_checkpoint_restores_weights_exactly(tmp_path):
    train_loader, val_loader = _make_dummy_loaders()
    best_path = str(tmp_path / "best.pth")
    last_path = str(tmp_path / "last.pth")

    model1 = FashionCNN()
    run_training(model1, train_loader, val_loader, num_epochs=1,
                 checkpoint_path=best_path, last_checkpoint_path=last_path,
                 device=torch.device("cpu"))

    model2 = FashionCNN()
    load_checkpoint(last_path, model2, device=torch.device("cpu"))

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2)


def test_save_and_load_checkpoint_restores_optimizer_state(tmp_path):
    checkpoint_path = str(tmp_path / "ckpt.pth")

    model = FashionCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    dummy_input = torch.randn(4, 1, 28, 28)
    loss = model(dummy_input).sum()
    loss.backward()
    optimizer.step()

    save_checkpoint(model, optimizer, scheduler=None, epoch=1,
                     best_metric=0.5, checkpoint_path=checkpoint_path)

    new_model = FashionCNN()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)
    checkpoint = load_checkpoint(checkpoint_path, new_model, new_optimizer,
                                  device=torch.device("cpu"))

    assert checkpoint["epoch"] == 1
    assert checkpoint["best_metric"] == 0.5
    assert len(new_optimizer.state_dict()["state"]) == len(optimizer.state_dict()["state"])