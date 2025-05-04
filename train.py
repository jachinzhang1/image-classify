import os
import torch
import numpy as np
import torch.nn as nn
from time import time
from typing import Optional
from pprint import pprint
from datetime import datetime
from configs import Configs
from dataset import get_dataloader
from model.attention_cnn import AttentionCNN
from model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202


def main(
    cfg: Configs,
    controller: Optional[object] = None,
    progress_callback: Optional[object] = None,
):
    device = cfg.device
    # Use the selected dataset from config
    dataloader = get_dataloader(
        cfg.data_root, 
        cfg.training_config["batch_size"], 
        dataset_name=cfg.selected_dataset,
        train=True
    )

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

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training_config["lr"])
    model.train()
    n_epochs = cfg.training_config["n_epochs"]
    total_iterations = n_epochs * len(dataloader)

    # Create output directory with hierarchical structure: dataset/model_type
    output_base = os.path.join(
        cfg.training_config["output_dir"],
        cfg.selected_dataset,   # First level: dataset name
        cfg.selected_model      # Second level: model type
    )
    os.makedirs(output_base, exist_ok=True)

    # Early stopping parameters
    patience = cfg.training_config.get("early_stopping_patience", 10)  # Get from config or use default
    wait = 0       # Counter for patience
    best_acc = 0.0
    best_epoch = 0
    best_state_dict = None

    # Training loop
    for epoch in range(n_epochs):
        if controller and getattr(controller, "should_stop", False):
            print("Training cancelled by user.")
            return

        tic = time()
        epoch_loss_list = []
        correct, total_samples = 0, 0
        for i, (images, labels) in enumerate(dataloader):
            if controller and getattr(controller, "should_stop", False):
                print("Training cancelled by user.")
                return

            if progress_callback:
                progress_callback.emit(
                    epoch * len(dataloader) + i +
                    1, total_iterations, epoch + 1
                )

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_list.append(loss.item())
            correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

        toc = time()
        epoch_acc = correct / total_samples
        epoch_loss = np.mean(epoch_loss_list)
        print(
            f"Epoch {epoch+1}/{n_epochs} Loss: {epoch_loss:.3f} Acc: {epoch_acc:.2f} Time: {toc-tic:.1f}s"
        )
        
        # Check if this is the best model so far
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            best_state_dict = model.state_dict().copy()
            
            # No need to save best.pth, we'll save it at the end
            print(f"Found new best model with accuracy: {best_acc:.4f} at epoch {epoch+1}")
            
            # Reset patience counter
            wait = 0
        else:
            # Increment patience counter
            wait += 1
            print(f"No improvement for {wait} epochs. Best accuracy: {best_acc:.4f} at epoch {best_epoch+1}")
            
            # Check if we should stop early
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
                break

    # If we have a best model (which we should), load it back
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"Loaded best model from epoch {best_epoch+1} with accuracy {best_acc:.4f}")

    # Save the best model in a timestamped directory
    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(output_base, curr_time)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_state_dict, os.path.join(save_dir, "model.pth"))
    print(f"Saved best model to {save_dir}/model.pth")
    
    # Also save a copy with standard name for easy testing
    standard_path = os.path.join(output_base, "latest.pth")
    torch.save(best_state_dict, standard_path)
    print(f"Saved copy of best model to {standard_path}")
    
    # Always ensure we return the best model
    return model, best_acc, best_epoch


if __name__ == "__main__":
    config = Configs(config_path="options.yaml")
    print("Training with configuration:")
    pprint(config.to_dict())
    model, best_acc, best_epoch = main(cfg=config)
    print(f"Best model: Epoch {best_epoch+1}, Accuracy: {best_acc:.4f}")
