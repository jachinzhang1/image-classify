import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from configs import Configs
from dataset import get_dataloader
from model.attention_cnn import AttentionCNN


def main(cfg: Configs):
    device = cfg.device
    dataloader = get_dataloader(
        cfg.data_root, cfg.test_config["batch_size"], train=False
    )
    assert cfg.selected_model in cfg.model_types
    if cfg.selected_model == "attention_cnn":
        model = AttentionCNN(num_classes=cfg.num_classes).to(device)
    else:
        raise ValueError("Unsupported model type.")

    model.load_state_dict(torch.load(cfg.test_config["ckpt_path"], map_location=device))
    model.eval()
    correct, total_samples = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

    acc = correct / total_samples
    print(f"Acc: {acc:.4f}")


if __name__ == "__main__":
    config = Configs(config_path="options.yaml")
    main(cfg=config)
