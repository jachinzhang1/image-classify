import torch
import torch.utils
import torch.utils.data
from model.attention_cnn import AttentionCNN
from model.resnet import (
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet20,
    ResNet32,
    ResNet44,
    ResNet56,
    ResNet110,
    ResNet1202,
)
from model.autoencoder import Autoencoder
from model.alexnet import AlexNet
from model.vgg import VGG11, VGG13, VGG16, VGG19
from model.mobilenet import MobileNet
from model.densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet264
from model.inception import Inception
from model.efficientnet import EfficientNet
from typing import Any


def pretrain_autoencoder(
    model: Autoencoder,
    training_config: dict[str, Any],
    dataloader: torch.utils.data.DataLoader,
    device,
    controller,
    progress_callback,
):
    pretrain_epochs = training_config.get("pretrain_epochs", 5)
    pretrain_lr = training_config.get("pretrain_lr", 0.001)
    print(f"Starting autoencoder pre-training ({pretrain_epochs} epochs)...")
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=pretrain_lr)

    pretrain_iterations = pretrain_epochs * len(dataloader)
    model.train_autoencoder(
        dataloader=dataloader,
        optimizer=pretrain_optimizer,
        epochs=pretrain_epochs,
        device=device,
        progress_callback=progress_callback,
        total_iterations=pretrain_iterations,
        controller=controller,
    )
    return model


def get_model(
    selected_model: str,
    num_classes: int = 10,
    device=torch.device,
    training_config=None,
    dataloader=None,
    controller=None,
    progress_callback=None,
) -> torch.nn.Module:
    if selected_model == "attention_cnn":
        model = AttentionCNN(num_classes).to(device)
    elif selected_model.startswith("resnet"):
        if selected_model.endswith("18"):
            model = ResNet18(num_classes).to(device)
        elif selected_model.endswith("34"):
            model = ResNet34(num_classes).to(device)
        elif selected_model.endswith("50"):
            model = ResNet50(num_classes).to(device)
        elif selected_model.endswith("101"):
            model = ResNet101(num_classes).to(device)
        elif selected_model.endswith("152"):
            model = ResNet152(num_classes).to(device)
        elif selected_model.endswith("20"):
            model = ResNet20(num_classes).to(device)
        elif selected_model.endswith("32"):
            model = ResNet32(num_classes).to(device)
        elif selected_model.endswith("44"):
            model = ResNet44(num_classes).to(device)
        elif selected_model.endswith("56"):
            model = ResNet56(num_classes).to(device)
        elif selected_model.endswith("110"):
            model = ResNet110(num_classes).to(device)
        elif selected_model.endswith("1202"):
            model = ResNet1202(num_classes).to(device)
        else:
            raise ValueError(f"Unknown ResNet variant: {selected_model}")
    elif selected_model == "Autoencoder":
        model = Autoencoder(num_classes).to(device)
        if training_config is not None:
            model = pretrain_autoencoder(
                model=model,
                training_config=training_config,
                dataloader=dataloader,
                device=device,
                controller=controller,
                progress_callback=progress_callback,
            )
    elif selected_model == "AlexNet":
        model = AlexNet(num_classes).to(device)
    elif selected_model.startswith("vgg"):
        if selected_model.endswith("11"):
            model = VGG11(num_classes).to(device)
        elif selected_model.endswith("13"):
            model = VGG13(num_classes).to(device)
        elif selected_model.endswith("16"):
            model = VGG16(num_classes).to(device)
        elif selected_model.endswith("19"):
            model = VGG19(num_classes).to(device)
        else:
            raise ValueError(f"Unknown VGG variant: {selected_model}")
    elif selected_model == "mobilenet":
        model = MobileNet(num_classes).to(device)
    elif selected_model.startswith("densenet"):
        if selected_model.endswith("121"):
            model = DenseNet121(num_classes).to(device)
        elif selected_model.endswith("169"):
            model = DenseNet169(num_classes).to(device)
        elif selected_model.endswith("201"):
            model = DenseNet201(num_classes).to(device)
        elif selected_model.endswith("264"):
            model = DenseNet264(num_classes).to(device)
        else:
            raise ValueError(f"Unknown DenseNet variant: {selected_model}")
    elif selected_model == "inception":
        model = Inception(num_classes).to(device)
    elif selected_model.startswith("efficientnet"):
        if selected_model.endswith("b0"):
            model = EfficientNet(num_classes, variant='b0').to(device)
        else:
            raise ValueError(f"Unknown EfficientNet variant: {selected_model}")
    else:
        raise ValueError(f"Unsupported model type: {selected_model}")

    return model
