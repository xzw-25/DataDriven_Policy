#!/usr/bin/env python3
"""Export a trained MLP to TorchScript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import PROJECT_ROOT
from vehicle_controller.constants import FEATURE_COUNT, FEATURE_NAMES
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.training.checkpoint import load_model_state
from vehicle_controller.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", default="artifacts/exported_models/controller")
    parser.add_argument("--model-config", default="configs/model/mlp_controller.yaml")
    args = parser.parse_args()
    output = PROJECT_ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    model_config_path = PROJECT_ROOT / args.model_config
    model = build_model(model_config_path).eval()
    load_model_state(PROJECT_ROOT / args.checkpoint, model)
    traced = torch.jit.trace(model, torch.zeros(1, FEATURE_COUNT))
    traced.save(str(output / "model.pt"))
    metadata = {
        "model_version": "controller_v003",
        "model_type": load_yaml(model_config_path)["type"],
        "feature_version": "features_v003",
        "feature_count": FEATURE_COUNT,
        "feature_names": FEATURE_NAMES,
        "output_names": ["steering_normalized", "signed_accel_normalized"],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
