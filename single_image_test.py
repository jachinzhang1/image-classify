import torch
import os
from typing import Optional
from PIL import Image
from torchvision import transforms
from configs import Configs
from utils import get_model
from dataset import cifar_transform, mnist_transform, large_input_transform, mnist_large_input_transform
from constants import class_dict, dataset_classes


def get_class_names(dataset_name, data_root, num_classes):
    """Get class names for the specified dataset"""
    if dataset_name in dataset_classes and dataset_classes[dataset_name]:
        return dataset_classes[dataset_name]

    # If the dataset is custom or not pre-defined, try to discover class names from the directory structure
    try:
        classes = {}
        if dataset_name == "custom":
            # Try to find class folders in the data directory
            class_dirs = [
                d
                for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            ]
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


def get_transform_for_dataset(dataset_name, model_type=None):
    """Get the appropriate transform for the dataset and model type"""
    # 需要大尺寸输入的模型列表
    large_input_models = ["inception", "efficientnet", "vgg11", "vgg13", "vgg16", "vgg19"]
    
    # 对于需要大尺寸输入的模型使用224x224的尺寸
    if model_type in large_input_models or (model_type is not None and any(model_type.startswith(prefix) for prefix in ["efficientnet", "vgg"])):
        if dataset_name == "mnist":
            return mnist_large_input_transform
        else:  # cifar10, cifar100等
            return large_input_transform
    
    # 对于其他模型使用原始转换
    if dataset_name == "cifar10" or dataset_name == "cifar100":
        return cifar_transform
    elif dataset_name == "mnist":
        return mnist_transform
    else:
        # Default to CIFAR transform for unrecognized datasets
        print(
            f"Warning: Transform for '{dataset_name}' not found, using CIFAR transform."
        )
        return cifar_transform


def classify_image(cfg: Configs, image_path: str):
    device = cfg.device

    # Get the appropriate transform for the dataset and model
    transform = get_transform_for_dataset(cfg.selected_dataset, cfg.selected_model)

    # 需要大尺寸输入的模型列表
    large_input_models = ["inception", "efficientnet", "vgg11", "vgg13", "vgg16", "vgg19"]
    uses_large_input = (cfg.selected_model in large_input_models or 
                       any(cfg.selected_model.startswith(prefix) for prefix in ["efficientnet", "vgg"]))

    # Load and preprocess image
    try:
        image = Image.open(image_path)

        # 根据模型类型和数据集类型调整大小
        if uses_large_input:
            # 需要大尺寸输入的模型
            if cfg.selected_dataset == "mnist":
                image = image.convert("L")  # Convert to grayscale
            else:
                image = image.convert("RGB")
            image = transforms.Resize((224, 224))(image)
        else:
            # 其他模型使用原始尺寸
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
            f"Checkpoint file {cfg.test_config['ckpt_path']} not found."
        )

    model = get_model(
        selected_model=cfg.selected_model, num_classes=cfg.num_classes, device=device
    )

    try:
        model.load_state_dict(
            torch.load(cfg.test_config["ckpt_path"], map_location=device)
        )
        model.eval()

        # Classify
        with torch.no_grad():
            output = model(image_tensor)
            
            # 处理Inception模型在评估模式下的输出
            if isinstance(output, tuple):
                output = output[0]  # 使用主输出
                
            _, predicted = torch.max(output.data, 1)

        # Get class names for the dataset
        classes = get_class_names(cfg.selected_dataset, cfg.data_root, cfg.num_classes)
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
                config.selected_dataset, config.data_root, config.num_classes
            )
            print(f"Predicted class: {classes[class_idx]} (index: {class_idx})")
        except Exception as e:
            print(f"Error: {str(e)}")
