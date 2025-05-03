import torch
from typing import Optional
from configs import Configs
from dataset import get_dataloader
from model.attention_cnn import AttentionCNN


def main(
    cfg: Configs,
    controller: Optional[object] = None,
    progress_callback: Optional[object] = None,
    accuracy_callback: Optional[object] = None,
):
    device = cfg.device
    dataloader = get_dataloader(
        cfg.data_root, cfg.test_config["batch_size"], train=False
    )

    if cfg.selected_model == "attention_cnn":
        model = AttentionCNN(num_classes=cfg.num_classes).to(device)
    else:
        raise ValueError("Unsupported model type.")

    model.load_state_dict(torch.load(cfg.test_config["ckpt_path"], map_location=device))
    model.eval()
    correct, total_samples = 0, 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if controller and getattr(controller, "should_stop", False):
                print("Testing cancelled by user.")
                return

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

            if progress_callback:
                progress_callback.emit(i + 1, len(dataloader))

    acc = correct / total_samples
    if accuracy_callback:
        accuracy_callback.emit(acc)
    print(f"Acc: {acc:.4f}")


if __name__ == "__main__":
    config = Configs(config_path="options.yaml")
    main(cfg=config)
