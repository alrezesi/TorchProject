# Fashion-MNIST Classification with PyTorch

## Problem
Multi-class image classification: given a 28x28 grayscale image of a
clothing item, predict which of 10 categories it belongs to
(T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker,
Bag, Ankle boot).

## Dataset
Fashion-MNIST — 60,000 training images, 10,000 test images, 10
balanced classes. 10% of the training set is held out as a validation
split (stratified by index, seed=42) for early stopping / model
selection during training.

## Architecture
A 3-block CNN (FashionCNN):
- Block 1: 1 -> 64 channels
- Block 2: 64 -> 128 channels
- Block 3: 128 -> 256 channels

Each block: Conv2d -> BatchNorm2d -> ReLU (x2) -> MaxPool2d -> Dropout.
Classifier head: Flatten -> Linear(2304, 512) -> ReLU -> BatchNorm1d ->
Dropout -> Linear(512, 10).

Total trainable parameters: <عدد واقعی از count_parameters() رو اینجا بذارید>

## Training configuration
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- LR schedule: CosineAnnealingLR
- Loss: CrossEntropyLoss with label smoothing (0.05)
- Batch size: 128 (train), 256 (eval)
- Epochs: <عدد واقعی>
- Data augmentation (train only): RandomCrop(28, padding=2), RandomHorizontalFlip

## Metrics (on the held-out test set)
| Metric | Value |
|---|---|
| Accuracy |
| Precision (macro) | 
| Recall (macro) |
| F1-score (macro) |

See `reports/loss_curve.png` and `reports/accuracy_curve.png` for
training curves, and the confusion matrix printed by `evaluate.py`.

## CPU/GPU support
Training and inference auto-detect CUDA and fall back to CPU
(`get_device()` in `src/train.py`). Developed and tested on CPU only.

## How to train
    python train.py --epochs 30
    python train.py --epochs 50 --resume checkpoints/best_fashion_mnist.pth

## How to evaluate
    python evaluate.py --checkpoint checkpoints/best_fashion_mnist.pth

## How to run inference
    python inference.py --image path/to/image.png