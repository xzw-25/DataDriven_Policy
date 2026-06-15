#!/usr/bin/env python3
"""Evaluate a checkpoint on an NPZ dataset."""

from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from _bootstrap import PROJECT_ROOT
from vehicle_controller.data.dataset import ControllerDataset
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.training.checkpoint import load_model_state
from vehicle_controller.training.evaluator import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("dataset_npz")
    parser.add_argument("--model-config", default="configs/model/mlp_controller.yaml")
    args = parser.parse_args()
    model = build_model(PROJECT_ROOT / args.model_config)
    load_model_state(PROJECT_ROOT / args.checkpoint, model)
    dataset = ControllerDataset.from_npz(PROJECT_ROOT / args.dataset_npz)
    metrics = evaluate(model, DataLoader(dataset, batch_size=512))
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
