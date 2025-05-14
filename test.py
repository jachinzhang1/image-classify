import torch
import os
from typing import Optional
from configs import Configs
from dataset import get_dataloader
from utils import get_model


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
        train=False,
        model_type=cfg.selected_model,
    )

    # Check if checkpoint exists
    ckpt_path = cfg.test_config["ckpt_path"]
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint file {ckpt_path} not found.")
        if accuracy_callback:
            accuracy_callback.emit(0.0)
        return

    model = get_model(
        selected_model=cfg.selected_model, num_classes=cfg.num_classes, device=device
    )

    try:
        # Load the model weights
        model.load_state_dict(
            torch.load(cfg.test_config["ckpt_path"], map_location=device)
        )
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
                
                # 处理Inception模型在评估模式下的输出
                # 在评估模式下，Inception通常只返回主输出，但为了安全起见，我们仍进行检查
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 使用主输出
                    
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
                print(
                    f"  Class {i}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})"
                )
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
