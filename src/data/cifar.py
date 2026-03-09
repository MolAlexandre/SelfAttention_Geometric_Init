"""
CIFAR-10 dataset loading with standard augmentations.
"""

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD  = (0.2470, 0.2435, 0.2616)


class CifarDatasetManager:
    """Download (once) and serve CIFAR-10 train / val DataLoaders."""

    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir   = cache_dir
        self.cifar_cache = os.path.join(cache_dir, "cifar10")

        self.transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])
        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STD),
        ])

    def create_dataloaders(self, batch_size: int, num_workers: int = 1):
        already_downloaded = os.path.exists(
            os.path.join(self.cifar_cache, "cifar-10-batches-py")
        )
        if not already_downloaded:
            print("Téléchargement de CIFAR-10...")
            os.makedirs(self.cifar_cache, exist_ok=True)

        train_ds = datasets.CIFAR10(
            root=self.cifar_cache, train=True,
            download=not already_downloaded, transform=self.transform_train,
        )
        val_ds = datasets.CIFAR10(
            root=self.cifar_cache, train=False,
            download=not already_downloaded, transform=self.transform_val,
        )

        pin = torch.cuda.is_available()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin)

        print(f"✓ CIFAR-10 — {len(train_ds)} train | {len(val_ds)} val")
        print(f"✓ {len(train_loader)} train batches | {len(val_loader)} val batches")
        return train_loader, val_loader
