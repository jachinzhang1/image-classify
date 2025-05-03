import torch
from typing import Optional
from PIL import Image
from torchvision import transforms
from configs import Configs
from model.attention_cnn import AttentionCNN
from model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202

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


def classify_image(cfg: Configs, image_path: str):
    device = cfg.device

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),  # Adjust based on model input size
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Load model
    assert cfg.selected_model in cfg.model_types
    
    if cfg.selected_model == "attention_cnn":
        model = AttentionCNN(num_classes=cfg.num_classes).to(device)
    elif cfg.selected_model == "resnet18":
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
        raise ValueError("Unsupported model type.")
    
    model.load_state_dict(torch.load(cfg.test_config["ckpt_path"], map_location=device))
    model.eval()

    # Classify
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)

    return predicted.item()  # Return class index


if __name__ == "__main__":
    config = Configs(config_path="options.yaml")
    image_path = "images/airplane-2.png"  # Replace with your image path
    class_idx = classify_image(config, image_path)
    print(f"Predicted class: {class_dict[class_idx]}")
