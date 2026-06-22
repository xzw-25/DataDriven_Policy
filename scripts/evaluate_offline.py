#!/usr/bin/env python3
"""Evaluate a checkpoint on an NPZ dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from _bootstrap import PROJECT_ROOT
from vehicle_controller.data.dataset import ControllerDataset
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.training.checkpoint import load_model_state
from vehicle_controller.training.evaluator import predict
from vehicle_controller.training.metrics import controller_metrics
from vehicle_controller.training.offline_plots import save_offline_control_comparison_plots
from vehicle_controller.units import steering_limit_deg_from_config
from vehicle_controller.utils.config import load_yaml


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _control_scales(model_config: dict[str, object]) -> np.ndarray:
    return np.asarray(
        [
            steering_limit_deg_from_config(model_config),
            float(model_config["accel_limit_mps2"]),
        ],
        dtype=np.float64,
    )


def _dataset_expert_controls(data: np.lib.npyio.NpzFile, scales: np.ndarray) -> np.ndarray:
    if "physical_targets" in data:
        controls = np.asarray(data["physical_targets"], dtype=np.float64)
    else:
        targets = np.asarray(data["targets"], dtype=np.float64)
        if np.max(np.abs(targets)) <= 1.05:
            controls = targets * scales
        else:
            controls = targets
    if "target_valid_mask" in data:
        controls = controls[np.asarray(data["target_valid_mask"], dtype=bool)]
    return controls

def _optional_filtered(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    values = data[key]
    if "target_valid_mask" in data:
        values = values[np.asarray(data["target_valid_mask"], dtype=bool)]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("dataset_npz")
    parser.add_argument("--model-config", default="configs/model/mlp_controller.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", default="artifacts/reports/offline_validation")
    parser.add_argument("--dataset-label", default="validation")
    parser.add_argument("--max-plot-scenarios", type=int, default=8)
    parser.add_argument("--max-plot-samples", type=int, default=2000)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()
    model_config_path = project_path(args.model_config)
    checkpoint_path = project_path(args.checkpoint)
    dataset_path = project_path(args.dataset_npz)

    model_config = load_yaml(model_config_path)
    scales = _control_scales(model_config)
    model = build_model(model_config_path)
    load_model_state(checkpoint_path, model, device=args.device)
    dataset = ControllerDataset.from_npz(dataset_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    normalized_predictions, normalized_targets = predict(model, loader, device=args.device)
    metrics = controller_metrics(normalized_predictions, normalized_targets)
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    if args.no_plots:
        return

    data = np.load(dataset_path)
    predicted_controls = normalized_predictions.numpy().astype(np.float64) * scales
    expert_controls = _dataset_expert_controls(data, scales)
    timestamps_s = _optional_filtered(data, "timestamps_s")
    scenario_ids = _optional_filtered(data, "scenario_ids")
    plot_paths = save_offline_control_comparison_plots(
        predicted_controls,
        expert_controls,
        project_path(args.output_dir),
        timestamps_s=timestamps_s,
        scenario_ids=scenario_ids,
        dataset_label=args.dataset_label,
        max_scenarios=args.max_plot_scenarios,
        maximum_samples_per_plot=args.max_plot_samples,
        show_plots=args.show_plots,
    )
    for path in plot_paths:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
