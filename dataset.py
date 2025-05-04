from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
import os
import torch
import numpy as np
from PIL import Image

# Define transformations for different datasets
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# For MNIST, we'll convert to 3 channels to match CNN input requirements
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert single channel to 3 channels
])


def get_dataset_and_transform(dataset_name, data_root, train=True):
    """Get the appropriate dataset and transform based on dataset name"""
    if dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=data_root, train=train, download=True, transform=cifar_transform
        ), cifar_transform
    elif dataset_name == "cifar100":
        return datasets.CIFAR100(
            root=data_root, train=train, download=True, transform=cifar_transform
        ), cifar_transform
    elif dataset_name == "mnist":
        return datasets.MNIST(
            root=data_root, train=train, download=True, transform=mnist_transform
        ), mnist_transform
    else:
        # Default to CIFAR-10 if dataset not recognized
        print(f"Warning: Dataset '{dataset_name}' not recognized, using CIFAR-10 instead.")
        return datasets.CIFAR10(
            root=data_root, train=train, download=True, transform=cifar_transform
        ), cifar_transform


def get_dataloader(data_root, batch_size, dataset_name="cifar10", train=True):
    """Get dataloader for the specified dataset"""
    dataset, _ = get_dataset_and_transform(dataset_name, data_root, train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train)
    return dataloader
