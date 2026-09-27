"""
One-off utility: export a few test-set images as PNG files, so we have
real images to test inference.py with. Not part of the core pipeline —
safe to delete after use.
"""

from torchvision import datasets
import os

os.makedirs("sample_images", exist_ok=True)

test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True)

CLASS_NAMES = [
    "T-shirt-top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle-boot",
]

for i in range(5):
    image, label = test_dataset[i]
    class_name = CLASS_NAMES[label]
    image.save(f"sample_images/sample_{i}_label_{label}_{class_name}.png")
    print(f"Saved sample_{i}_label_{label}_{class_name}.png")

print("\nDone. Images saved to sample_images/")