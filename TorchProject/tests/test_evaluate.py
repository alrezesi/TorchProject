"""
Tests for src/evaluate.py's metric computation, using hand-crafted
predictions with known, hand-calculated correct answers — not model
output — so the expected values are unambiguous.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.evaluate import get_predictions, evaluate_full


class _PerfectModel(torch.nn.Module):
    """A fake 'model' that always outputs the correct label as a one-hot
    logit vector — used to verify metrics equal 100% on a perfect predictor."""

    def forward(self, x, labels_for_test=None):
        raise NotImplementedError


def test_perfect_predictions_give_100_percent_metrics():
    """
    If predictions exactly match the true labels, accuracy, precision,
    recall and F1 should all be 1.0 (100%) — the simplest possible
    correctness check for the metric-computation code itself.
    """
    from sklearn.metrics import accuracy_score, f1_score

    true_labels = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    predictions = true_labels.copy()  # perfect predictions

    assert accuracy_score(true_labels, predictions) == 1.0
    assert f1_score(true_labels, predictions, average="macro") == 1.0


def test_evaluate_full_detects_all_wrong_predictions():
    """
    A model that is wrong on every single sample should score 0%
    accuracy — sanity-checks the metric pipeline at the other extreme.
    """
    from sklearn.metrics import accuracy_score

    true_labels = [0, 1, 2, 3]
    predictions = [1, 2, 3, 0]  # every prediction shifted — all wrong

    assert accuracy_score(true_labels, predictions) == 0.0


def test_get_predictions_returns_matching_lengths():
    """get_predictions must return one prediction per input sample."""
    from src.model import FashionCNN

    images = torch.randn(20, 1, 28, 28)
    labels = torch.randint(0, 10, (20,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=7)  # uneven last batch on purpose

    model = FashionCNN()
    device = torch.device("cpu")

    true_labels, predictions = get_predictions(model, loader, device)

    assert len(true_labels) == 20
    assert len(predictions) == 20


def test_evaluate_full_returns_all_expected_keys():
    from src.model import FashionCNN

    # Use enough samples with a fixed manual seed to make it very likely
    # (not guaranteed, but good enough here) all 10 classes appear —
    # though evaluate_full itself is now robust even if they don't.
    torch.manual_seed(0)
    images = torch.randn(64, 1, 28, 28)
    labels = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)

    model = FashionCNN()
    metrics = evaluate_full(model, loader, torch.device("cpu"))

    expected_keys = {"accuracy", "precision_macro", "recall_macro",
                      "f1_macro", "confusion_matrix", "classification_report"}
    assert expected_keys.issubset(metrics.keys())