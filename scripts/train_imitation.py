#!/usr/bin/env python3
"""Train the baseline MLP from NPZ data or a synthetic smoke dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from _bootstrap import PROJECT_ROOT
from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.data.dataset import ControllerDataset
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.training.checkpoint import save_checkpoint
from vehicle_controller.training.losses import ControllerLoss
from vehicle_controller.training.trainer import Trainer
from vehicle_controller.utils.config import load_yaml
from vehicle_controller.utils.random import seed_everything


def synthetic_dataset(sample_count: int, seed: int) -> ControllerDataset:
    generator = np.random.default_rng(seed)
    features = generator.normal(0.0, 0.5, size=(sample_count, FEATURE_COUNT)).astype(
        np.float32
    )
    targets = np.stack(
        (
            np.tanh(0.6 * features[:, 10] + 0.3 * features[:, 11]),
            np.tanh(0.4 * features[:, 14] + 0.3 * features[:, 12]),
        ),
        axis=-1,
    ).astype(np.float32)
    return ControllerDataset(features, targets)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def nested_config_path(config: dict[str, object], section: str) -> Path:
    section_config = config.get(section)
    if not isinstance(section_config, dict) or "config" not in section_config:
        raise ValueError(f"Missing '{section}.config' in the main configuration")
    return project_path(str(section_config["config"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output", default="artifacts/checkpoints/baseline.pt")
    parser.add_argument("--device")
    parser.add_argument("--model-config")
    parser.add_argument("--training-config")
    args = parser.parse_args()

    main_config = load_yaml(project_path(args.config))
    model_config_path = (
        project_path(args.model_config)
        if args.model_config
        else nested_config_path(main_config, "model")
    )
    training_config_path = (
        project_path(args.training_config)
        if args.training_config
        else nested_config_path(main_config, "training")
    )
    model_config = load_yaml(model_config_path)
    training_config = load_yaml(training_config_path)
    seed = int(main_config.get("seed", 42))
    device = args.device or str(main_config.get("device", "cpu"))
    epochs = args.epochs if args.epochs is not None else int(training_config["epochs"])
    if epochs < 0:
        raise ValueError("epochs must be non-negative")

    seed_everything(seed)
    dataset = (
        ControllerDataset.from_npz(project_path(args.dataset))
        if args.dataset
        else synthetic_dataset(2048, seed)
    )
    loader = DataLoader(
        dataset,
        batch_size=min(int(training_config["batch_size"]), len(dataset)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
    )
    model = build_model(model_config_path)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    trainer = Trainer(
        model,
        optimizer,
        ControllerLoss(
            steering_weight=float(training_config["steering_loss_weight"]),
            acceleration_weight=float(training_config["acceleration_loss_weight"]),
        ),
        device=device,
        gradient_clip_norm=float(training_config["gradient_clip_norm"]),
    )
    for epoch in range(epochs):
        result = trainer.train_epoch(loader)
        print(f"epoch={epoch + 1} loss={result.loss:.6f}")
    output_path = project_path(args.output)
    save_checkpoint(output_path, model, optimizer, model_config, epochs)
    print(f"checkpoint={output_path}")


if __name__ == "__main__":
    main()
