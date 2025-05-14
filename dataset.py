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

# 添加需要224x224尺寸的模型转换
large_input_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小为224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# For MNIST, we'll convert to 3 channels to match CNN input requirements
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert single channel to 3 channels
])

# MNIST转换为224x224，用于需要大尺寸输入的模型
mnist_large_input_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整大小为224x224
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert single channel to 3 channels
])


def get_dataset_and_transform(dataset_name, data_root, train=True, model_type=None):
    """Get the appropriate dataset and transform based on dataset name and model type"""
    
    # 需要大尺寸输入的模型列表
    large_input_models = ["inception", "efficientnet", "vgg11", "vgg13", "vgg16", "vgg19"]
    
    # 根据模型类型选择适当的转换
    if model_type in large_input_models or (model_type is not None and any(model_type.startswith(prefix) for prefix in ["efficientnet", "vgg"])):
        # 对于需要大尺寸输入的模型，使用224x224的图像尺寸
        if dataset_name == "cifar10":
            return datasets.CIFAR10(
                root=data_root, train=train, download=True, transform=large_input_transform
            ), large_input_transform
        elif dataset_name == "cifar100":
            return datasets.CIFAR100(
                root=data_root, train=train, download=True, transform=large_input_transform
            ), large_input_transform
        elif dataset_name == "mnist":
            return datasets.MNIST(
                root=data_root, train=train, download=True, transform=mnist_large_input_transform
            ), mnist_large_input_transform
    
    # 对其他模型使用原始转换
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


def get_dataloader(data_root, batch_size, dataset_name="cifar10", train=True, model_type=None):
    """Get dataloader for the specified dataset"""
    dataset, _ = get_dataset_and_transform(dataset_name, data_root, train, model_type)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train)
    return dataloader
