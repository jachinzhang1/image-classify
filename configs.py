import yaml
import torch
from typing import Any


class Configs:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg: dict = yaml.safe_load(f)

        self.device = torch.device(cfg.get("device", "cuda"))
        self.data_root = cfg.get("data_root", "data")
        self.model_types = cfg.get("model_types", [])
        self.selected_model = cfg.get("selected_model", None)
        self.num_classes = cfg.get("num_classes", 10)

        self.training_config = cfg.get(
            "train",
            {"n_epochs": 30, "batch_size": 64, "lr": 0.001, "output_dir": "ckpts"},
        )

        self.test_config = cfg.get("test", {"batch_size": 1, "ckpt_path": None})

    def to_dict(self):
        return {
            "device": self.device,
            "data_root": self.data_root,
            "selected_model_type": self.selected_model,
            "training_configs": self.training_config,
            "test_config": self.test_config,
        }
