"""
Tests for FashionCNN's architecture: shape correctness and basic
sanity checks that don't require any real data.
"""

import torch

from src.model import FashionCNN, count_parameters


def test_output_shape_matches_num_classes():
    """A batch of 32 images should produce 32 rows of 10-class logits."""
    model = FashionCNN(num_classes=10)
    dummy_batch = torch.randn(32, 1, 28, 28)

    output = model(dummy_batch)

    assert output.shape == (32, 10)


def test_output_shape_with_batch_size_one():
    """
    Batch size 1 is a common edge case (e.g. single-image inference).
    BatchNorm1d in the classifier head can fail on batch_size=1 during
    training mode (it needs more than one sample to compute a batch
    statistic), so this is checked in eval mode.
    """
    model = FashionCNN(num_classes=10)
    model.eval()
    dummy_image = torch.randn(1, 1, 28, 28)

    with torch.no_grad():
        output = model(dummy_image)

    assert output.shape == (1, 10)


def test_model_has_trainable_parameters():
    """Sanity check: the model should have a nonzero, reasonable parameter count."""
    model = FashionCNN()
    param_count = count_parameters(model)

    assert param_count > 0


def test_model_output_changes_after_one_training_step():
    """
    After one optimizer step on a batch with a nonzero loss, the model's
    weights should have changed. If they don't, something is broken in
    the forward/backward/step wiring (e.g. gradients not flowing).
    """
    model = FashionCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))

    # Snapshot one weight tensor before the update.
    param_before = next(model.parameters()).clone()

    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    param_after = next(model.parameters())

    assert not torch.allclose(param_before, param_after), \
        "Model weights did not change after a training step"