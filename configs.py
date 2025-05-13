import yaml
import torch
from typing import Any


class Configs:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg: dict = yaml.safe_load(f)

        self.device = torch.device(cfg.get("device", "cuda"))

        # Dataset configuration
        self.datasets = cfg.get("datasets", {})
        self.selected_dataset = cfg.get("selected_dataset", "cifar10")

        # Check if data_root is directly specified in config
        self.data_root = cfg.get("data_root")

        # If not, get it from dataset configuration
        if self.data_root is None and self.selected_dataset in self.datasets:
            dataset_config = self.datasets[self.selected_dataset]
            self.data_root = dataset_config.get("path", "./data")
        elif self.data_root is None:
            self.data_root = "./data"

        # Get number of classes from dataset configuration or use default
        if self.selected_dataset in self.datasets:
            dataset_config = self.datasets[self.selected_dataset]
            self.num_classes = dataset_config.get("num_classes", 10)
        else:
            self.num_classes = cfg.get("num_classes", 10)

        # Model configuration
        self.model_types = cfg.get("model_types", [])
        self.selected_model = cfg.get("selected_model", None)

        # ResNet variant configuration
        self.resnet_variants = cfg.get("resnet_variants", ["resnet18"])
        self.selected_resnet_variant = cfg.get("selected_resnet_variant", "resnet18")

        self.training_config: dict[str, Any] = cfg.get(
            "train",
            {"n_epochs": 30, "batch_size": 64, "lr": 0.001, "output_dir": "ckpts"},
        )

        self.test_config: dict[str, Any] = cfg.get("test", {"batch_size": 1, "ckpt_path": None})

    def to_dict(self):
        return {
            "device": self.device,
            "data_root": self.data_root,
            "num_classes": self.num_classes,
            "selected_dataset": self.selected_dataset,
            "selected_model": self.selected_model,
            "selected_resnet_variant": self.selected_resnet_variant,
            "training_config": self.training_config,
            "test_config": self.test_config,
        }

    def update_dataset(self, dataset_name):
        """Update current selected dataset and its related configuration"""
        if dataset_name in self.datasets:
            self.selected_dataset = dataset_name
            dataset_config = self.datasets[dataset_name]

            # Only update data_root if it wasn't directly specified in the config
            # This preserves any user modifications to the path
            if self.data_root == "./data" or self.data_root is None:
                self.data_root = dataset_config.get("path", "./data")

            self.num_classes = dataset_config.get("num_classes", 10)
            return True
        return False
