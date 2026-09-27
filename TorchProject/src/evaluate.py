"""
Full test-set evaluation: accuracy, precision, recall, F1, and a
confusion matrix — beyond the running accuracy/loss already computed
during training/validation in src/train.py.
"""

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.dataset import CLASS_NAMES


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """
    Run the model over the full dataloader and collect all predictions
    and true labels as flat lists (needed by sklearn's metric functions,
    which expect 1D arrays, not per-batch tensors).
    """
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

    return all_labels, all_preds


def evaluate_full(model, test_loader, device) -> dict:
    """
    Computes accuracy, macro precision/recall/F1, and the confusion
    matrix on the test set. Returns a dict so the caller (evaluate.py,
    or a report generator) can use the numbers however it needs.
    """
    true_labels, predictions = get_predictions(model, test_loader, device)

    # macro average: each class contributes equally to the final score,
    # regardless of how many test samples it has. This matters more on
    # imbalanced datasets; Fashion-MNIST is balanced, so macro and
    # weighted averages come out nearly identical here.
    all_labels = list(range(len(CLASS_NAMES)))  # explicitly force all 10 classes

    metrics = {
        "accuracy": accuracy_score(true_labels, predictions),
        "precision_macro": precision_score(true_labels, predictions, labels=all_labels,
                                           average="macro", zero_division=0),
        "recall_macro": recall_score(true_labels, predictions, labels=all_labels,
                                     average="macro", zero_division=0),
        "f1_macro": f1_score(true_labels, predictions, labels=all_labels,
                             average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(true_labels, predictions, labels=all_labels),
        "classification_report": classification_report(
            true_labels, predictions, labels=all_labels,
            target_names=CLASS_NAMES, zero_division=0,
        ),
    }
    return metrics


def print_evaluation_report(metrics: dict) -> None:
    """Prints a human-readable summary of the evaluation results."""
    print(f"Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision (macro): {metrics['precision_macro'] * 100:.2f}%")
    print(f"Recall (macro):    {metrics['recall_macro'] * 100:.2f}%")
    print(f"F1-score (macro):  {metrics['f1_macro'] * 100:.2f}%")
    print("\nPer-class report:")
    print(metrics["classification_report"])
    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])