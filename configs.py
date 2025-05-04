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
        
        # Get current selected dataset configuration
        if self.selected_dataset in self.datasets:
            dataset_config = self.datasets[self.selected_dataset]
            self.data_root = dataset_config.get("path", "./data")
            self.num_classes = dataset_config.get("num_classes", 10)
        else:
            self.data_root = "./data"
            self.num_classes = 10
        
        # Model configuration
        self.model_types = cfg.get("model_types", [])
        self.selected_model = cfg.get("selected_model", None)

        self.training_config = cfg.get(
            "train",
            {"n_epochs": 30, "batch_size": 64, "lr": 0.001, "output_dir": "ckpts"},
        )

        self.test_config = cfg.get("test", {"batch_size": 1, "ckpt_path": None})

    def to_dict(self):
        return {
            "device": self.device,
            "data_root": self.data_root,
            "num_classes": self.num_classes,
            "selected_dataset": self.selected_dataset,
            "selected_model_type": self.selected_model,
            "training_configs": self.training_config,
            "test_config": self.test_config,
        }

    def update_dataset(self, dataset_name):
        """Update current selected dataset and its related configuration"""
        if dataset_name in self.datasets:
            self.selected_dataset = dataset_name
            dataset_config = self.datasets[dataset_name]
            self.data_root = dataset_config.get("path", "./data")
            self.num_classes = dataset_config.get("num_classes", 10)
            return True
        return False
