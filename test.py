import torch
import os
from typing import Optional
from configs import Configs
from dataset import get_dataloader
from model.attention_cnn import AttentionCNN
from model.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet1202
from model.autoencoder import Autoencoder

def main(
    cfg: Configs,
    controller: Optional[object] = None,
    progress_callback: Optional[object] = None,
    accuracy_callback: Optional[object] = None,
):
    device = cfg.device
    
    # Use the selected dataset from config
    dataloader = get_dataloader(
        cfg.data_root, 
        cfg.test_config["batch_size"], 
        dataset_name=cfg.selected_dataset,
        train=False
    )

    # Check if checkpoint exists
    ckpt_path = cfg.test_config["ckpt_path"]
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file {ckpt_path} not found.")
        if accuracy_callback:
            accuracy_callback.emit(0.0)
        return

    # Load model based on selected model type
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
        error_msg = f"Unsupported model type: {cfg.selected_model}"
        print(error_msg)
        raise ValueError(error_msg)

    try:
        # Load the model weights
        model.load_state_dict(torch.load(
            cfg.test_config["ckpt_path"], map_location=device))
        model.eval()
        
        # Perform evaluation
        correct, total_samples = 0, 0
        class_correct = [0] * cfg.num_classes
        class_total = [0] * cfg.num_classes
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(dataloader):
                if controller and getattr(controller, "should_stop", False):
                    print("Testing cancelled by user.")
                    return

                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                pred = outputs.argmax(1)
                
                # Overall accuracy
                correct += (pred == labels).sum().item()
                total_samples += labels.size(0)
                
                # Per-class accuracy
                for j in range(labels.size(0)):
                    label = labels[j].item()
                    class_correct[label] += (pred[j] == label).item()
                    class_total[label] += 1

                if progress_callback:
                    progress_callback.emit(i + 1, len(dataloader))

        # Calculate and report accuracy
        acc = correct / total_samples
        if accuracy_callback:
            accuracy_callback.emit(acc)
            
        print(f"Overall Accuracy: {acc:.4f}")
        
        # Print per-class accuracy
        print("\nPer-class accuracy:")
        for i in range(cfg.num_classes):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                print(f"  Class {i}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
            else:
                print(f"  Class {i}: No samples")
                
    except Exception as e:
        error_msg = f"Error during testing: {str(e)}"
        print(error_msg)
        if accuracy_callback:
            accuracy_callback.emit(0.0)
        raise Exception(error_msg)


if __name__ == "__main__":
    config = Configs(config_path="options.yaml")
    print("Testing with configuration:")
    print(f"- Dataset: {config.selected_dataset}")
    print(f"- Model: {config.selected_model}")
    print(f"- Checkpoint: {config.test_config['ckpt_path']}")
    main(cfg=config)
