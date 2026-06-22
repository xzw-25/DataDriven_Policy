#!/usr/bin/env python3
"""Train/validate from a generated feature NPZ and plot validation behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts._bootstrap import PROJECT_ROOT

from vehicle_controller.data.dataset import ControllerDataset
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.training.checkpoint import save_checkpoint
from vehicle_controller.training.evaluator import predict
from vehicle_controller.training.loss_plots import (
    make_loss_history,
    save_loss_curve,
    save_loss_history_csv,
)
from vehicle_controller.training.losses import ControllerLoss
from vehicle_controller.training.metrics import controller_metrics
from vehicle_controller.training.offline_plots import save_offline_control_comparison_plots
from vehicle_controller.training.trainer import Trainer
from vehicle_controller.utils.config import load_yaml
from vehicle_controller.utils.random import seed_everything


DEFAULT_DATASET = "data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz"
DEFAULT_OUTPUT_DIR = "artifacts/reports/feature_train_validation"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def nested_config_path(config: dict[str, object], section: str) -> Path:
    section_config = config.get(section)
    if not isinstance(section_config, dict) or "config" not in section_config:
        raise ValueError(f"Missing '{section}.config' in the main configuration")
    return project_path(str(section_config["config"]))


def _load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    return dict(json.loads(str(data["metadata_json"])))


def _valid_indices(data: np.lib.npyio.NpzFile) -> np.ndarray:
    count = int(data["features"].shape[0])
    if "target_valid_mask" not in data:
        return np.arange(count, dtype=np.int64)
    mask = np.asarray(data["target_valid_mask"], dtype=bool)
    if mask.shape != (count,):
        raise ValueError("target_valid_mask must have shape [N]")
    return np.flatnonzero(mask).astype(np.int64)


def _split_indices(
    indices: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if indices.ndim != 1 or len(indices) < 2:
        raise ValueError("At least two valid samples are required for train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    validation_count = int(round(len(shuffled) * validation_fraction))
    validation_count = min(max(validation_count, 1), len(shuffled) - 1)
    validation_indices = np.sort(shuffled[:validation_count])
    train_indices = np.sort(shuffled[validation_count:])
    return train_indices, validation_indices


def _optional_array(
    data: np.lib.npyio.NpzFile,
    name: str,
    indices: np.ndarray,
) -> np.ndarray | None:
    if name not in data:
        return None
    values = np.asarray(data[name])
    if values.shape[0] != data["features"].shape[0]:
        raise ValueError(f"{name} must be frame-aligned with features")
    return values[indices]


def _target_scales(metadata: dict[str, Any]) -> np.ndarray | None:
    target_normalization = metadata.get("target_normalization")
    if not isinstance(target_normalization, dict):
        return None
    steering_scale = target_normalization.get("steering_scale_deg")
    accel_scale = target_normalization.get("accel_scale_mps2")
    if steering_scale is None or accel_scale is None:
        return None
    return np.asarray([float(steering_scale), float(accel_scale)], dtype=np.float32)


def _physical_predictions(
    normalized_predictions: np.ndarray,
    validation_targets: np.ndarray,
    physical_targets: np.ndarray | None,
    metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    if physical_targets is not None:
        scales = _target_scales(metadata)
        if scales is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where(
                    np.abs(validation_targets) > 1e-6,
                    physical_targets / validation_targets,
                    np.nan,
                )
            scales = np.nanmedian(ratios, axis=0).astype(np.float32)
        if scales.shape != (2,) or not np.all(np.isfinite(scales)):
            raise ValueError("Could not infer target physical scales")
        return (
            normalized_predictions * scales[None, :],
            physical_targets,
            "physical",
        )
    return normalized_predictions, validation_targets, "normalized"


def _control_metrics(predicted: np.ndarray, target: np.ndarray, prefix: str) -> dict[str, float]:
    error = predicted - target
    return {
        f"{prefix}_steering_mae": float(np.mean(np.abs(error[:, 0]))),
        f"{prefix}_acceleration_mae": float(np.mean(np.abs(error[:, 1]))),
        f"{prefix}_steering_rmse": float(np.sqrt(np.mean(error[:, 0] ** 2))),
        f"{prefix}_acceleration_rmse": float(np.sqrt(np.mean(error[:, 1] ** 2))),
    }


def train_validate_from_features(
    *,
    dataset: str | Path = DEFAULT_DATASET,
    config: str | Path = "configs/default.yaml",
    model_config: str | Path | None = None,
    training_config: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    validation_fraction: float = 0.2,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
    seed: int | None = None,
    max_validation_scenarios: int = 8,
    show_plots: bool = False,
) -> Path:
    dataset_path = project_path(dataset)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    main_config = dict(load_yaml(project_path(config)))
    model_config_path = project_path(model_config) if model_config else nested_config_path(main_config, "model")
    training_config_path = (
        project_path(training_config)
        if training_config
        else nested_config_path(main_config, "training")
    )
    resolved_device = device or str(main_config.get("device", "cpu"))
    resolved_seed = int(seed if seed is not None else main_config.get("seed", 42))
    training = dict(load_yaml(training_config_path))
    resolved_epochs = int(epochs if epochs is not None else training["epochs"])
    resolved_batch_size = int(batch_size if batch_size is not None else training["batch_size"])
    if resolved_epochs < 0:
        raise ValueError("epochs must be non-negative")
    if resolved_batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_path = project_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    seed_everything(resolved_seed)

    with np.load(dataset_path, allow_pickle=False) as data:
        valid_indices = _valid_indices(data)
        train_indices, validation_indices = _split_indices(
            valid_indices,
            validation_fraction,
            resolved_seed,
        )
        features = np.asarray(data["features"], dtype=np.float32)
        targets = np.asarray(data["targets"], dtype=np.float32)
        metadata = _load_metadata(data)
        physical_targets = _optional_array(data, "physical_targets", validation_indices)
        timestamps_s = _optional_array(data, "timestamps_s", validation_indices)
        scenario_ids = _optional_array(data, "scenario_ids", validation_indices)
        if scenario_ids is None:
            scenario_ids = _optional_array(data, "clip_ids", validation_indices)

    train_dataset = ControllerDataset(features[train_indices], targets[train_indices])
    validation_dataset = ControllerDataset(features[validation_indices], targets[validation_indices])
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(resolved_batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 0)),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=min(resolved_batch_size, len(validation_dataset)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 0)),
    )

    model = build_model(model_config_path)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    trainer = Trainer(
        model,
        optimizer,
        ControllerLoss(
            steering_weight=float(training["steering_loss_weight"]),
            acceleration_weight=float(training["acceleration_loss_weight"]),
        ),
        device=resolved_device,
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )

    batch_losses_by_epoch: list[tuple[float, ...]] = []
    epoch_losses: list[float] = []
    for epoch in range(resolved_epochs):
        result = trainer.train_epoch(train_loader)
        batch_losses_by_epoch.append(result.batch_losses)
        epoch_losses.append(result.loss)
        validation_predictions, validation_targets = predict(
            model,
            validation_loader,
            device=resolved_device,
        )
        validation_metrics = controller_metrics(validation_predictions, validation_targets)
        print(
            f"epoch={epoch + 1} "
            f"train_loss={result.loss:.6f} "
            f"val_steering_mae={validation_metrics['steering_mae']:.6f} "
            f"val_acceleration_mae={validation_metrics['acceleration_mae']:.6f}"
        )

    if batch_losses_by_epoch:
        history = make_loss_history(batch_losses_by_epoch, epoch_losses)
        history_path = save_loss_history_csv(history, output_path / "loss_history.csv")
        curve_path = save_loss_curve(history, output_path / "loss_curve.png", show_plots=show_plots)
        print(f"loss_history={history_path}")
        print(f"loss_curve={curve_path}")

    validation_predictions, validation_targets = predict(
        model,
        validation_loader,
        device=resolved_device,
    )
    normalized_predictions = validation_predictions.numpy()
    normalized_targets = validation_targets.numpy()
    normalized_metrics = controller_metrics(validation_predictions, validation_targets)
    plotted_predictions, plotted_targets, plot_units = _physical_predictions(
        normalized_predictions,
        normalized_targets,
        None if physical_targets is None else np.asarray(physical_targets, dtype=np.float32),
        metadata,
    )
    metrics = {
        "dataset": str(dataset_path),
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "validation_fraction": float(validation_fraction),
        "seed": int(resolved_seed),
        "epochs": int(resolved_epochs),
        "plot_units": plot_units,
        **{f"normalized_{key}": value for key, value in normalized_metrics.items()},
        **_control_metrics(plotted_predictions, plotted_targets, plot_units),
    }
    metrics_path = output_path / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    split_path = output_path / "split_indices.npz"
    np.savez_compressed(
        split_path,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )
    plot_paths = save_offline_control_comparison_plots(
        plotted_predictions,
        plotted_targets,
        output_path / "validation_plots",
        timestamps_s=None if timestamps_s is None else np.asarray(timestamps_s, dtype=np.float64),
        scenario_ids=None if scenario_ids is None else np.asarray(scenario_ids).astype(str),
        dataset_label=f"validation_{plot_units}",
        max_scenarios=max_validation_scenarios,
        show_plots=show_plots,
    )
    checkpoint_path = output_path / "model.pt"
    model_config_dict = dict(load_yaml(model_config_path))
    save_checkpoint(checkpoint_path, model, optimizer, model_config_dict, resolved_epochs)

    print(f"split_indices={split_path}")
    print(f"metrics={metrics_path}")
    print(f"checkpoint={checkpoint_path}")
    for plot_path in plot_paths:
        print(f"validation_plot={plot_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-config")
    parser.add_argument("--training-config")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-validation-scenarios", type=int, default=8)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    train_validate_from_features(
        dataset=args.dataset,
        config=args.config,
        model_config=args.model_config,
        training_config=args.training_config,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        max_validation_scenarios=args.max_validation_scenarios,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()
22