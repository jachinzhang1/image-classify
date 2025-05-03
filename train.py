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


def main(
    cfg: Configs,
    controller: Optional[object] = None,
    progress_callback: Optional[object] = None,
):
    device = cfg.device
    dataloader = get_dataloader(
        cfg.data_root, cfg.training_config["batch_size"], train=True
    )

    if cfg.selected_model == "attention_cnn":
        model = AttentionCNN(num_classes=cfg.num_classes).to(device)
    else:
        raise ValueError("Unsupported model type.")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training_config["lr"])
    model.train()
    n_epochs = cfg.training_config["n_epochs"]
    total_iterations = n_epochs * len(dataloader)

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
                    epoch * len(dataloader) + i + 1, total_iterations, epoch + 1
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

    curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = os.path.join(
        cfg.training_config["output_dir"], cfg.selected_model, curr_time
    )
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))


if __name__ == "__main__":
    config = Configs(config_path="options.yaml")
    pprint(config.to_dict())
    main(cfg=config)
