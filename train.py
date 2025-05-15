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
from utils import get_model


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
        train=True,
        model_type=cfg.selected_model,
    )

    model = get_model(
        selected_model=cfg.selected_model,
        num_classes=cfg.num_classes,
        device=device,
        training_config=cfg.training_config,
        dataloader=dataloader,
        controller=controller,
        progress_callback=progress_callback,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training_config["lr"])
    model.train()
    n_epochs = cfg.training_config["n_epochs"]
    total_iterations = n_epochs * len(dataloader)

    # Create output directory with hierarchical structure: dataset/model_type
    # Check if the output_dir already contains dataset and model name
    # to avoid duplicate nesting
    output_dir = cfg.training_config["output_dir"]
    output_path_parts = output_dir.replace("\\", "/").split("/")

    # If the output directory already has the dataset and model as last components, use it directly
    if (
        len(output_path_parts) >= 2
        and output_path_parts[-2] == cfg.selected_dataset
        and output_path_parts[-1] == cfg.selected_model
    ):
        output_base = output_dir
    else:
        # Create hierarchical path with dataset and model type
        output_base = os.path.join(
            output_dir,
            cfg.selected_dataset,  # First level: dataset name
            cfg.selected_model,  # Second level: model type
        )

    os.makedirs(output_base, exist_ok=True)

    # Early stopping parameters
    # Get from config or use default
    patience = cfg.training_config.get("early_stopping_patience", 10)
    wait = 0  # Counter for patience
    best_acc = 0.0
    best_epoch = 0
    best_state_dict = None

    # checkpoints
    save_checkpoints = cfg.training_config.get("save_checkpoints", True)
    checkpoints_epochs = cfg.training_config.get("checkpoints_epochs", []) if save_checkpoints else []

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
                    epoch * len(dataloader) + i + 1, total_iterations, epoch + 1
                )

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # 处理Inception模型的多输出情况
            if cfg.selected_model == "inception" and isinstance(outputs, tuple):
                # 在训练模式下，Inception模型会返回(主输出, 辅助输出1, 辅助输出2)
                # 计算所有输出的损失并加权
                main_output = outputs[0]
                aux_outputs = outputs[1:]
                
                # 主输出的损失
                loss = criterion(main_output, labels)
                
                # 添加辅助分类器的损失，权重为0.3
                for aux_output in aux_outputs:
                    loss += 0.3 * criterion(aux_output, labels)
                
                # 使用主输出计算准确率
                correct += (main_output.argmax(1) == labels).sum().item()
            else:
                # 对于其他模型，直接计算损失
                loss = criterion(outputs, labels)
                correct += (outputs.argmax(1) == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_list.append(loss.item())
            total_samples += labels.size(0)

        toc = time()
        epoch_acc = correct / total_samples
        epoch_loss = np.mean(epoch_loss_list)
        print(
            f"Epoch {epoch+1}/{n_epochs} Loss: {epoch_loss:.3f} Acc: {epoch_acc:.2%} Time: {toc-tic:.1f}s"
        )
        
        # 保存特定epoch的模型

        if save_checkpoints and  epoch + 1 in checkpoints_epochs: 
            print("Saving checkpoint...")
            checkpoint_dir = os.path.join(output_base, f"epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")
            
        # Check if this is the best model so far
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            best_state_dict = model.state_dict().copy()

            # No need to save best.pth, we'll save it at the end
            print(
                f"Found new best model with accuracy: {best_acc:.4f} at epoch {epoch+1}"
            )

            # Reset patience counter
            wait = 0
        else:
            # Increment patience counter
            wait += 1
            print(
                f"No improvement for {wait} epochs. Best accuracy: {best_acc:.4f} at epoch {best_epoch+1}"
            )

            # Check if we should stop early
            if wait >= patience:
                print(
                    f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs."
                )
                break

    # If we have a best model (which we should), load it back
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            f"Loaded best model from epoch {best_epoch+1} with accuracy {best_acc:.4f}"
        )

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
