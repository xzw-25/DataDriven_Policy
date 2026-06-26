"""Validate supervised controller predictions against dataset targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from vehicle_controller.data.dataset import ControllerDataset
from vehicle_controller.data.feature_builder import STANDSTILL_REQUEST_NPZ_KEY
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.training.checkpoint import load_model_state
from vehicle_controller.training.evaluator import predict
from vehicle_controller.training.metrics import controller_metrics
from vehicle_controller.training.offline_plots import save_offline_control_comparison_plots
from vehicle_controller.units import steering_limit_deg_from_config
from vehicle_controller.utils.config import load_yaml


@dataclass(frozen=True)
class SupervisedValidationResult:
    metrics: dict[str, float]
    prediction_path: Path
    metrics_path: Path
    plot_paths: tuple[Path, ...]


def control_scales(model_config: Mapping[str, object]) -> np.ndarray:
    return np.asarray(
        [
            steering_limit_deg_from_config(model_config),
            float(model_config["accel_limit_mps2"]),
        ],
        dtype=np.float64,
    )


def load_model_config_from_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a model config")
    return dict(config)


def load_validation_model(
    checkpoint_path: str | Path,
    model_config_path: str | Path | None = None,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_config = (
        dict(load_yaml(Path(model_config_path)))
        if model_config_path is not None
        else load_model_config_from_checkpoint(checkpoint_path)
    )
    model = build_model(model_config)
    load_model_state(checkpoint_path, model, device=device)
    return model, model_config


def valid_sample_mask(data: np.lib.npyio.NpzFile) -> np.ndarray:
    feature_count = int(data["features"].shape[0])
    if "target_valid_mask" not in data:
        return np.ones(feature_count, dtype=bool)
    mask = np.asarray(data["target_valid_mask"], dtype=bool)
    if mask.shape != (feature_count,):
        raise ValueError("target_valid_mask must have shape [N]")
    return mask


def physical_targets_from_npz(data: np.lib.npyio.NpzFile, scales: np.ndarray) -> np.ndarray:
    mask = valid_sample_mask(data)
    if "physical_targets" in data:
        values = np.asarray(data["physical_targets"], dtype=np.float64)
    else:
        values = np.asarray(data["targets"], dtype=np.float64) * scales
    if values.shape != (len(mask), 2):
        raise ValueError("targets/physical_targets must have shape [N, 2]")
    return values[mask]


def optional_filtered_array(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray | None:
    if key not in data:
        return None
    values = np.asarray(data[key])
    mask = valid_sample_mask(data)
    if values.shape[0] != len(mask):
        raise ValueError(f"{key} must have first dimension N")
    return values[mask]


def optional_standstill_requests(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    values = optional_filtered_array(data, STANDSTILL_REQUEST_NPZ_KEY)
    if values is not None:
        return values
    return optional_filtered_array(data, "standstill_requests")


def physical_error_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return {
        "physical_steering_mae_deg": float(np.mean(np.abs(error[:, 0]))),
        "physical_acceleration_mae_mps2": float(np.mean(np.abs(error[:, 1]))),
        "physical_steering_rmse_deg": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        "physical_acceleration_rmse_mps2": float(np.sqrt(np.mean(error[:, 1] ** 2))),
        "physical_steering_max_abs_deg": float(np.max(np.abs(error[:, 0]))),
        "physical_acceleration_max_abs_mps2": float(np.max(np.abs(error[:, 1]))),
    }


def validate_supervised_dataset(
    checkpoint_path: str | Path,
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    model_config_path: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 512,
    dataset_label: str = "supervised_validation",
    max_plot_scenarios: int = 8,
    max_plot_samples: int = 2000,
    show_plots: bool = False,
) -> SupervisedValidationResult:
    checkpoint = Path(checkpoint_path)
    dataset_npz = Path(dataset_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model, model_config = load_validation_model(checkpoint, model_config_path, device=device)
    scales = control_scales(model_config)
    dataset = ControllerDataset.from_npz(dataset_npz)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    normalized_predictions, normalized_targets = predict(model, loader, device=device)
    normalized_metrics = {
        f"normalized_{name}": value
        for name, value in controller_metrics(normalized_predictions, normalized_targets).items()
    }

    with np.load(dataset_npz, allow_pickle=False) as data:
        physical_predictions = normalized_predictions.numpy().astype(np.float64) * scales
        physical_targets = physical_targets_from_npz(data, scales)
        timestamps_s = optional_filtered_array(data, "timestamps_s")
        scenario_ids = optional_filtered_array(data, "scenario_ids")
        clip_ids = optional_filtered_array(data, "clip_ids")
        frame_indices = optional_filtered_array(data, "frame_indices")
        standstill_requests = optional_standstill_requests(data)

    if physical_predictions.shape != physical_targets.shape:
        raise ValueError("Prediction/target shape mismatch after valid-mask filtering")

    metrics = {
        **normalized_metrics,
        **physical_error_metrics(physical_predictions, physical_targets),
        "sample_count": float(len(physical_predictions)),
    }

    prediction_path = output / f"{dataset_label}_predictions.npz"
    np.savez_compressed(
        prediction_path,
        normalized_predictions=normalized_predictions.numpy().astype(np.float32),
        normalized_targets=normalized_targets.numpy().astype(np.float32),
        predicted_controls=physical_predictions.astype(np.float32),
        target_controls=physical_targets.astype(np.float32),
        timestamps_s=timestamps_s if timestamps_s is not None else np.asarray([], dtype=np.float64),
        scenario_ids=scenario_ids if scenario_ids is not None else np.asarray([], dtype=str),
        clip_ids=clip_ids if clip_ids is not None else np.asarray([], dtype=str),
        frame_indices=frame_indices if frame_indices is not None else np.asarray([], dtype=np.int32),
        standstill_requests=(
            standstill_requests
            if standstill_requests is not None
            else np.asarray([], dtype=bool)
        ),
    )

    metrics_path = output / f"{dataset_label}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    plot_paths = save_offline_control_comparison_plots(
        physical_predictions,
        physical_targets,
        output,
        timestamps_s=timestamps_s,
        scenario_ids=scenario_ids,
        standstill_requests=standstill_requests,
        dataset_label=dataset_label,
        max_scenarios=max_plot_scenarios,
        maximum_samples_per_plot=max_plot_samples,
        show_plots=show_plots,
    )
    return SupervisedValidationResult(metrics, prediction_path, metrics_path, plot_paths)
