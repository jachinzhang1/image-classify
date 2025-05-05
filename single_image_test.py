import torch
import os
from typing import Optional
from PIL import Image
from torchvision import transforms
from configs import Configs
from model.attention_cnn import AttentionCNN
from model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202
from model.autoencoder import Autoencoder
from dataset import cifar_transform, mnist_transform

# Default class dictionary for CIFAR-10
class_dict = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

# Class dictionaries for other datasets
dataset_classes = {
    "cifar10": {
        0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
        5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
    },
    "cifar100": {},  # Will be dynamically loaded if needed
    "mnist": {
        0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
        5: "5", 6: "6", 7: "7", 8: "8", 9: "9"
    },
}


def get_class_names(dataset_name, data_root, num_classes):
    """Get class names for the specified dataset"""
    if dataset_name in dataset_classes and dataset_classes[dataset_name]:
        return dataset_classes[dataset_name]

    # If the dataset is custom or not pre-defined, try to discover class names from the directory structure
    try:
        classes = {}
        if dataset_name == "custom":
            # Try to find class folders in the data directory
            class_dirs = [d for d in os.listdir(data_root)
                          if os.path.isdir(os.path.join(data_root, d))]
            class_dirs.sort()
            for i, class_name in enumerate(class_dirs):
                if i < num_classes:
                    classes[i] = class_name

        # If no classes found, use numerical indices
        if not classes:
            classes = {i: str(i) for i in range(num_classes)}

        # Cache the classes
        dataset_classes[dataset_name] = classes
        return classes

    except Exception as e:
        print(f"Error getting class names: {str(e)}")
        # Fallback to numerical indices
        return {i: str(i) for i in range(num_classes)}


def get_transform_for_dataset(dataset_name):
    """Get the appropriate transform for the dataset"""
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        return cifar_transform
    elif dataset_name == "mnist":
        return mnist_transform
    else:
        # Default to CIFAR transform for unrecognized datasets
        print(
            f"Warning: Transform for '{dataset_name}' not found, using CIFAR transform.")
        return cifar_transform


def classify_image(cfg: Configs, image_path: str):
    device = cfg.device

    # Get the appropriate transform for the dataset
    transform = get_transform_for_dataset(cfg.selected_dataset)

    # Load and preprocess image
    try:
        image = Image.open(image_path)

        # For MNIST, convert to grayscale first
        if cfg.selected_dataset == "mnist":
            image = image.convert("L")  # Convert to grayscale
            image = transforms.Resize((28, 28))(image)
        else:  # CIFAR and others need RGB
            image = image.convert("RGB")
            image = transforms.Resize((32, 32))(image)

        # Apply the transform and prepare tensor (transform will handle channel duplication for MNIST)
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        raise Exception(f"Error preprocessing image: {str(e)}")

    # Check if checkpoint exists
    if not os.path.exists(cfg.test_config["ckpt_path"]):
        raise FileNotFoundError(
            f"Checkpoint file {cfg.test_config['ckpt_path']} not found.")

    # Load model
    if cfg.selected_model == "attention_cnn":
        model = AttentionCNN(num_classes=cfg.num_classes).to(device)
    # Handle all ResNet variants
    elif cfg.selected_model.startswith("resnet"):
        # Create the appropriate ResNet variant based on selected model
        if cfg.selected_model == "resnet18":
            model = ResNet18(num_classes=cfg.num_classes).to(device)
        elif cfg.selected_model == "resnet34":
            model = ResNet34(num_classes=cfg.num_classes).to(device)
        elif cfg.selected_model == "resnet50":
            model = ResNet50(num_classes=cfg.num_classes).to(device)
        elif cfg.selected_model == "resnet101":
            model = ResNet101(num_classes=cfg.num_classes).to(device)
        elif cfg.selected_model == "resnet152":
            model = ResNet152(num_classes=cfg.num_classes).to(device)
        elif cfg.selected_model == "resnet20":
            model = ResNet20(num_classes=cfg.num_classes).to(device)
        elif cfg.selected_model == "resnet32":
            model = ResNet32(num_classes=cfg.num_classes).to(device)
        elif cfg.selected_model == "resnet44":
            model = ResNet44(num_classes=cfg.num_classes).to(device)
        elif cfg.selected_model == "resnet56":
            model = ResNet56(num_classes=cfg.num_classes).to(device)
        elif cfg.selected_model == "resnet110":
            model = ResNet110(num_classes=cfg.num_classes).to(device)
        elif cfg.selected_model == "resnet1202":
            model = ResNet1202(num_classes=cfg.num_classes).to(device)
        else:
            raise ValueError(f"Unknown ResNet variant: {cfg.selected_model}")
    elif cfg.selected_model == "Autoencoder":
        model = Autoencoder(num_classes=cfg.num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model type: {cfg.selected_model}")

    try:
        model.load_state_dict(torch.load(
            cfg.test_config["ckpt_path"], map_location=device))
        model.eval()

        # Classify
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output.data, 1)

        # Get class names for the dataset
        classes = get_class_names(
            cfg.selected_dataset, cfg.data_root, cfg.num_classes)
        class_idx = predicted.item()

        # Update global class_dict for the UI to use
        global class_dict
        class_dict = classes

        return class_idx
    except Exception as e:
        raise Exception(f"Error during classification: {str(e)}")


if __name__ == "__main__":
    config = Configs(config_path="options.yaml")
    image_path = input("Enter path to image: ")

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
    else:
        try:
            print(f"Dataset: {config.selected_dataset}")
            print(f"Model: {config.selected_model}")
            print(f"Checkpoint: {config.test_config['ckpt_path']}")

            class_idx = classify_image(config, image_path)
            classes = get_class_names(
                config.selected_dataset, config.data_root, config.num_classes)
            print(
                f"Predicted class: {classes[class_idx]} (index: {class_idx})")
        except Exception as e:
            print(f"Error: {str(e)}")
