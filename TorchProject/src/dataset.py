"""
Data pipeline for Fashion-MNIST:
- Loading
- Preprocessing (normalize + train-only augmentation)
- Train / Validation / Test split
- DataLoader creation
"""

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def get_train_transform():
    """
    Augmentation is applied ONLY to training data.
    RandomCrop + padding and RandomHorizontalFlip create slightly varied
    versions of each image, which helps the model generalize better.
    """
    return transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    ])


def get_eval_transform():
    """
    No augmentation for validation/test — we want to evaluate on
    the real, unmodified data distribution.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
    ])


def get_datasets(data_dir: str = "./data", val_ratio: float = 0.1, seed: int = 42):
    """
    Returns: train_dataset, val_dataset, test_dataset
    Train and validation use different transforms even though they come
    from the same underlying 60,000 images, so we load the base dataset
    twice (once augmented, once not) and split by matching indices.
    """
    # Two views of the same 60k images, with different transforms
    full_train_augmented = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=get_train_transform()
    )
    full_train_plain = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=get_eval_transform()
    )
    test_dataset = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=get_eval_transform()
    )

    val_size = int(len(full_train_augmented) * val_ratio)
    train_size = len(full_train_augmented) - val_size

    # Split indices once, then apply the same split to both dataset views
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        range(len(full_train_augmented)), [train_size, val_size], generator=generator
    )
    train_indices, val_indices = list(train_subset), list(val_subset)

    train_dataset = Subset(full_train_augmented, train_indices)  # augmented
    val_dataset = Subset(full_train_plain, val_indices)          # not augmented

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(data_dir: str = "./data", train_batch_size: int = 128,
                     eval_batch_size: int = 256, val_ratio: float = 0.1,
                     num_workers: int = 2, seed: int = 42):
    """
    Returns: train_loader, val_loader, test_loader
    """
    train_dataset, val_dataset, test_dataset = get_datasets(data_dir, val_ratio, seed)

    # pin_memory only helps when a GPU is available; harmless to gate it either way
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")