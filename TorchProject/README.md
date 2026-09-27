# 🧥 Fashion-MNIST Classification with PyTorch

A from-scratch PyTorch project implementing a full deep learning
pipeline — custom CNN, manual training loop, checkpointing with
resume support, and complete evaluation — without using any
pre-built trainer (no Hugging Face Trainer, no PyTorch Lightning).

## 🎯 Problem

Multi-class image classification: given a 28x28 grayscale image of a
clothing item, predict which of 10 categories it belongs to
(T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker,
Bag, Ankle boot).

## 📦 Dataset

**Fashion-MNIST** — 60,000 training images, 10,000 test images, 10
balanced classes (6,000 images/class in the original training set).

- Training set is split into **54,000 train / 6,000 validation**
  (10% held out, stratified by index, `seed=42`) for model selection
  during training.
- Test set (10,000 images) is never seen during training — used only
  for final, unbiased evaluation.
- Data augmentation (`RandomCrop(28, padding=2)`, `RandomHorizontalFlip`)
  is applied to the training split only; validation and test use only
  normalization, so they reflect the true data distribution.

## 🏗️ Architecture — `FashionCNN`

A 3-block convolutional network:

| Block | Layers | Channels | Spatial size |
|---|---|---|---|
| Input | — | 1 | 28x28 |
| Block 1 | Conv-BN-ReLU x2 -> MaxPool(2) -> Dropout(0.20) | 1 -> 64 | 28x28 -> 14x14 |
| Block 2 | Conv-BN-ReLU x2 -> MaxPool(2) -> Dropout(0.25) | 64 -> 128 | 14x14 -> 7x7 |
| Block 3 | Conv-BN-ReLU x2 -> MaxPool(2) -> Dropout(0.30) | 128 -> 256 | 7x7 -> 3x3 |
| Classifier | Flatten -> Linear(2304, 512) -> ReLU -> BatchNorm1d -> Dropout(0.40) -> Linear(512, 10) | — | 2304 -> 10 |

All convolutions use `kernel_size=3, padding=1`. Output is raw
(unnormalized) logits — softmax is applied only where needed
(inference), since `CrossEntropyLoss` already applies `log_softmax`
internally during training.

**Total trainable parameters:** `<PARAM_COUNT>` (run `python -m src.model` to see this)

## ⚙️ Training configuration

| Setting | Value |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| LR schedule | CosineAnnealingLR |
| Loss | CrossEntropyLoss (label_smoothing=0.05) |
| Batch size | 128 (train), 256 (eval) |
| Epochs run | 12 |
| Best checkpoint | epoch 11 (val_acc=94.18%) |
| Device | CPU (CUDA auto-detected and used automatically if available) |

Training saves two checkpoints:
- 🏆 `checkpoints/best_fashion_mnist.pth` — saved only when validation
  accuracy improves; used for evaluation/inference.
- 💾 `checkpoints/last_fashion_mnist.pth` — saved every epoch
  unconditionally; used to safely resume training even after a run of
  epochs with no improvement.

Both checkpoints store the full training state (model weights,
optimizer state, scheduler state, epoch, best metric, and full
per-epoch history), not just model weights — so resuming continues
exactly where training left off.

## 📊 Results (best checkpoint, epoch 11)

| Split | Accuracy | Precision (macro) | Recall (macro) | F1-score (macro) |
|---|---|---|---|---|
| Train* | 95.01% | 95.00% | 95.00% | 95.00% |
| Validation | 94.18% | 94.22% | 94.27% | 94.24% |
| **Test** | **93.58%** | **93.59%** | **93.58%** | **93.58%** |

\* Train metrics are computed with the same augmentation used during
training (random crop/flip), so they are slightly lower than a
no-augmentation pass would show — and close to validation/test,
indicating the model is **not overfitting** ✅.

### 🔍 Per-class performance (test set)

The hardest class by far is **Shirt** (precision 0.81, recall 0.81),
which the confusion matrix shows is most often confused with
**T-shirt/top** and **Coat** — visually similar upper-body garments.
Trouser, Sandal, and Bag are the easiest classes (precision/recall
≥ 0.97), being visually distinct from the rest.

Full classification report and confusion matrix per split are
available by running `evaluate_all_splits.py`.

See `reports/loss_curve.png` (Train vs Validation Loss) and
`reports/accuracy_curve.png` (Accuracy vs Epoch) for training curves.

## 📁 Project structure

| File / Folder | Purpose |
|---|---|
| `train.py` | CLI entry point — training |
| `evaluate.py` | CLI entry point — test-set evaluation |
| `inference.py` | CLI entry point — single-image prediction |
| `src/dataset.py` | Data loading, transforms, splitting |
| `src/model.py` | FashionCNN architecture |
| `src/train.py` | Training loop, validation, device management |
| `src/evaluate.py` | Metrics: accuracy / precision / recall / F1 / confusion matrix |
| `src/utils.py` | Checkpoint save / load |
| `src/visualization.py` | Loss / accuracy plot generation |
| `checkpoints/` | Saved model checkpoints (best + last) |
| `reports/` | Generated plots |
| `tests/` | Unit tests (pytest) |


## 🚀 How to train

```bash
python train.py --epochs 12
```

Resume from a checkpoint (e.g. to train additional epochs):

```bash
python train.py --epochs 13 --resume checkpoints/last_fashion_mnist.pth
```

## 📈 How to evaluate

```bash
python evaluate.py --checkpoint checkpoints/best_fashion_mnist.pth
```

## 🔮 How to run inference

```bash
python inference.py --image path/to/image.png --checkpoint checkpoints/best_fashion_mnist.pth
```

The input image is converted to grayscale and resized to 28x28
automatically; preprocessing exactly matches what validation/test
data receives during training (no augmentation).

## ✅ Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests cover model shape correctness, the training loop's train/eval
mode switching, checkpoint save/load/resume behavior (including
optimizer state and full history restoration), and metric computation
correctness.