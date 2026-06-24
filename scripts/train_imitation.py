#!/usr/bin/env python3
"""Train the baseline MLP from an NPZ dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts._bootstrap import PROJECT_ROOT
from vehicle_controller.data.dataset import ControllerDataset
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.simulation.showcase import run_typical_reference_showcase
from vehicle_controller.training.checkpoint import save_checkpoint
from vehicle_controller.training.evaluator import predict
from vehicle_controller.training.losses import ControllerLoss
from vehicle_controller.training.loss_plots import (
    make_loss_history,
    save_loss_curve,
    save_loss_history_csv,
)
from vehicle_controller.training.metrics import controller_metrics
from vehicle_controller.training.offline_plots import (
    save_feature_signal_plots,
    save_offline_control_comparison_plots,
)
from vehicle_controller.training.trainer import Trainer
from vehicle_controller.units import steering_limit_deg_from_config
from vehicle_controller.utils.config import load_yaml
from vehicle_controller.utils.device import is_cuda_device, preferred_training_device
from vehicle_controller.utils.random import seed_everything


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def nested_config_path(config: dict[str, object], section: str) -> Path:
    section_config = config.get(section)
    if not isinstance(section_config, dict) or "config" not in section_config:
        raise ValueError(f"Missing '{section}.config' in the main configuration")
    return project_path(str(section_config["config"]))


def config_value_path(
    config: dict[str, object],
    section: str,
    key: str,
) -> Path:
    section_config = config.get(section)
    if not isinstance(section_config, dict) or key not in section_config:
        raise ValueError(f"Missing '{section}.{key}' in the main configuration")
    return project_path(str(section_config[key]))


def resolve_dataset_path(
    explicit_dataset: str | None,
    main_config: dict[str, object],
) -> Path:
    if explicit_dataset:
        dataset_path = project_path(explicit_dataset)
        if not dataset_path.is_file():
            raise ValueError(f"Dataset file not found: {dataset_path}")
        return dataset_path

    data_config_path = nested_config_path(main_config, "data")
    data_config = load_yaml(data_config_path)
    processed_dir = data_config.get("processed_dir")
    if not isinstance(processed_dir, str) or not processed_dir:
        raise ValueError("Missing 'processed_dir' in the data configuration")

    processed_path = project_path(processed_dir)
    if not processed_path.is_dir():
        raise ValueError(f"Processed data directory not found: {processed_path}")

    datasets = sorted(processed_path.glob("*.npz"))
    if not datasets:
        raise ValueError(
            f"No NPZ dataset found in processed_dir: {processed_path}"
        )
    if len(datasets) > 1:
        names = ", ".join(path.name for path in datasets)
        raise ValueError(
            "Multiple NPZ datasets found in processed_dir; "
            f"please specify --dataset explicitly. Candidates: {names}"
        )
    return datasets[0]


def split_dataset_path(dataset_path: Path, split_dir: Path, split_name: str) -> Path:
    return split_dir / f"{dataset_path.stem}_{split_name}.npz"


def resolve_split_dataset_paths(
    dataset_path: Path,
    main_config: dict[str, object],
    explicit_train_dataset: str | Path | None = None,
    explicit_validation_dataset: str | Path | None = None,
    explicit_test_dataset: str | Path | None = None,
) -> dict[str, Path | None]:
    data_config_path = nested_config_path(main_config, "data")
    data_config = load_yaml(data_config_path)
    split_dir = project_path(str(data_config.get("split_dir", dataset_path.parent)))

    train_path = (
        project_path(explicit_train_dataset)
        if explicit_train_dataset is not None
        else split_dataset_path(dataset_path, split_dir, "train")
    )
    validation_path = (
        project_path(explicit_validation_dataset)
        if explicit_validation_dataset is not None
        else split_dataset_path(dataset_path, split_dir, "val")
    )
    test_path = (
        project_path(explicit_test_dataset)
        if explicit_test_dataset is not None
        else split_dataset_path(dataset_path, split_dir, "test")
    )
    if explicit_train_dataset is None and not train_path.is_file():
        train_path = dataset_path
    if explicit_validation_dataset is None and not validation_path.is_file():
        validation_path = None
    if explicit_test_dataset is None and not test_path.is_file():
        test_path = None

    for name, path in {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
    }.items():
        if path is not None and not path.is_file():
            raise ValueError(f"{name} dataset file not found: {path}")
    return {"train": train_path, "validation": validation_path, "test": test_path}


def make_controller_loader(
    dataset_path: Path,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> tuple[ControllerDataset, DataLoader]:
    dataset = ControllerDataset.from_npz(dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, loader


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    sample_count = 0
    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            loss = loss_function(model(features), targets)
            batch_size = int(features.shape[0])
            total_loss += float(loss.item()) * batch_size
            sample_count += batch_size
    if sample_count == 0:
        raise ValueError("Cannot evaluate loss on an empty loader")
    return total_loss / sample_count


def valid_sample_mask(data: np.lib.npyio.NpzFile) -> np.ndarray:
    count = int(data["features"].shape[0])
    if "target_valid_mask" not in data:
        return np.ones(count, dtype=bool)
    mask = np.asarray(data["target_valid_mask"], dtype=bool)
    if mask.shape != (count,):
        raise ValueError("target_valid_mask must have shape [N]")
    return mask


def target_scales_from_metadata(data: np.lib.npyio.NpzFile, model_config: dict[str, object]) -> np.ndarray:
    if "metadata_json" in data:
        metadata = json.loads(str(data["metadata_json"]))
        target_normalization = metadata.get("target_normalization")
        if isinstance(target_normalization, dict):
            steering_scale = target_normalization.get("steering_scale_deg")
            accel_scale = target_normalization.get("accel_scale_mps2")
            if steering_scale is not None and accel_scale is not None:
                return np.asarray([float(steering_scale), float(accel_scale)], dtype=np.float64)
    return np.asarray(
        [
            steering_limit_deg_from_config(model_config),
            float(model_config["accel_limit_mps2"]),
        ],
        dtype=np.float64,
    )


def filtered_optional_array(data: np.lib.npyio.NpzFile, key: str, mask: np.ndarray) -> np.ndarray | None:
    if key not in data:
        return None
    values = np.asarray(data[key])
    if values.shape[0] != len(mask):
        raise ValueError(f"{key} must be frame-aligned with features")
    return values[mask]


def physical_targets_from_npz(
    data: np.lib.npyio.NpzFile,
    mask: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    if "physical_targets" in data:
        values = np.asarray(data["physical_targets"], dtype=np.float64)
    else:
        values = np.asarray(data["targets"], dtype=np.float64) * scales
    if values.shape != (len(mask), 2):
        raise ValueError("targets/physical_targets must have shape [N, 2]")
    return values[mask]


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


def save_split_output_comparison(
    *,
    split_name: str,
    dataset_path: Path,
    model: torch.nn.Module,
    model_config: dict[str, object],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    device: str,
    output_dir: Path,
    max_plot_scenarios: int | None,
    max_plot_samples: int,
    show_plots: bool,
) -> dict[str, float]:
    _, loader = make_controller_loader(
        dataset_path,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    normalized_predictions, normalized_targets = predict(model, loader, device=device)
    normalized_metrics = {
        f"normalized_{key}": value
        for key, value in controller_metrics(normalized_predictions, normalized_targets).items()
    }
    with np.load(dataset_path, allow_pickle=False) as data:
        mask = valid_sample_mask(data)
        scales = target_scales_from_metadata(data, model_config)
        features = np.asarray(data["features"], dtype=np.float64)[mask]
        feature_names = (
            tuple(str(value) for value in data["feature_names"])
            if "feature_names" in data
            else None
        )
        physical_predictions = normalized_predictions.numpy().astype(np.float64) * scales
        physical_targets = physical_targets_from_npz(data, mask, scales)
        timestamps_s = filtered_optional_array(data, "timestamps_s", mask)
        scenario_ids = filtered_optional_array(data, "scenario_ids", mask)
        pose_headings_rad = filtered_optional_array(data, "pose_heading_rad", mask)
        if pose_headings_rad is None:
            pose_headings_rad = filtered_optional_array(data, "headings_rad", mask)
        pose_positions_enu = None
        if "positions_enu" in data:
            pose_positions_enu = np.asarray(data["positions_enu"], dtype=np.float64)[mask]
        else:
            pose_x_enu_m = filtered_optional_array(data, "pose_position_x_enu_m", mask)
            pose_y_enu_m = filtered_optional_array(data, "pose_position_y_enu_m", mask)
            if pose_x_enu_m is not None and pose_y_enu_m is not None:
                pose_positions_enu = np.stack(
                    (
                        np.asarray(pose_x_enu_m, dtype=np.float64),
                        np.asarray(pose_y_enu_m, dtype=np.float64),
                    ),
                    axis=1,
                )
        if scenario_ids is None:
            scenario_ids = filtered_optional_array(data, "clip_ids", mask)
    if physical_predictions.shape != physical_targets.shape:
        raise ValueError(f"{split_name} prediction/target shape mismatch")

    split_output = output_dir / split_name
    split_output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "sample_count": float(len(physical_predictions)),
        **normalized_metrics,
        **physical_error_metrics(physical_predictions, physical_targets),
    }
    metrics_path = split_output / f"{split_name}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        split_output / f"{split_name}_predictions.npz",
        normalized_predictions=normalized_predictions.numpy().astype(np.float32),
        normalized_targets=normalized_targets.numpy().astype(np.float32),
        predicted_controls=physical_predictions.astype(np.float32),
        target_controls=physical_targets.astype(np.float32),
        timestamps_s=timestamps_s if timestamps_s is not None else np.asarray([], dtype=np.float64),
        scenario_ids=scenario_ids if scenario_ids is not None else np.asarray([], dtype=str),
    )
    plot_paths = save_offline_control_comparison_plots(
        physical_predictions,
        physical_targets,
        split_output,
        timestamps_s=None if timestamps_s is None else np.asarray(timestamps_s, dtype=np.float64),
        scenario_ids=None if scenario_ids is None else np.asarray(scenario_ids).astype(str),
        dataset_label=split_name,
        max_scenarios=max_plot_scenarios,
        maximum_samples_per_plot=max_plot_samples,
        show_plots=show_plots,
    )
    print(f"{split_name}_metrics={metrics_path}")
    for plot_path in plot_paths:
        print(f"{split_name}_comparison_plot={plot_path}")
    feature_plot_paths = save_feature_signal_plots(
        features,
        split_output / "feature_signals",
        timestamps_s=None if timestamps_s is None else np.asarray(timestamps_s, dtype=np.float64),
        scenario_ids=None if scenario_ids is None else np.asarray(scenario_ids).astype(str),
        positions_enu=None if pose_positions_enu is None else np.asarray(pose_positions_enu, dtype=np.float64),
        headings_rad=None if pose_headings_rad is None else np.asarray(pose_headings_rad, dtype=np.float64),
        dataset_label=split_name,
        feature_names=feature_names,
        max_scenarios=max_plot_scenarios,
        maximum_samples_per_plot=max_plot_samples,
        show_plots=show_plots,
    )
    for plot_path in feature_plot_paths:
        print(f"{split_name}_feature_plot={plot_path}")
    return metrics


def train_imitation(
    *,
    config: str | Path = "configs/default.yaml",
    dataset: str | Path | None = None,
    train_dataset: str | Path | None = None,
    validation_dataset: str | Path | None = None,
    test_dataset: str | Path | None = None,
    epochs: int | None = None,
    output: str | Path = "artifacts/checkpoints/baseline.pt",
    device: str | None = None,
    model_config: str | Path | None = None,
    training_config: str | Path | None = None,
    loss_curve_output: str | Path = "artifacts/reports/imitation_training/loss_curve.png",
    loss_history_output: str | Path = "artifacts/reports/imitation_training/loss_history.csv",
    no_loss_plot: bool = False,
    comparison_output_dir: str | Path | None = None,
    max_comparison_scenarios: int | None = None,
    max_comparison_samples: int = 2000,
    showcase_output_dir: str | Path = "artifacts/reports/training_showcase",
    no_showcase: bool = False,
    show_plots: bool = False,
) -> Path:
    main_config = load_yaml(project_path(config))
    model_config_path = (
        project_path(model_config)
        if model_config
        else nested_config_path(main_config, "model")
    )
    training_config_path = (
        project_path(training_config)
        if training_config
        else nested_config_path(main_config, "training")
    )
    data_config_path = nested_config_path(main_config, "data")
    generation_config_path = config_value_path(main_config, "data", "generation")
    vehicle_parameters_path = config_value_path(main_config, "vehicle", "parameters")
    actuator_limits_path = config_value_path(main_config, "vehicle", "actuator_limits")
    normalization_path = config_value_path(main_config, "data", "normalization")
    safety_limits_path = config_value_path(main_config, "deployment", "safety_limits")
    model_config = load_yaml(model_config_path)
    training_config = load_yaml(training_config_path)
    generation_config = load_yaml(generation_config_path)
    generation_time_step_s = float(generation_config["time_step_s"])
    seed = int(main_config.get("seed", 42))
    resolved_device = preferred_training_device(device, main_config.get("device"))
    resolved_epochs = epochs if epochs is not None else int(training_config["epochs"])
    dataset_path = resolve_dataset_path(None if dataset is None else str(dataset), main_config)
    split_paths = resolve_split_dataset_paths(
        dataset_path,
        main_config,
        explicit_train_dataset=train_dataset,
        explicit_validation_dataset=validation_dataset,
        explicit_test_dataset=test_dataset,
    )
    if resolved_epochs < 0:
        raise ValueError("epochs must be non-negative")

    seed_everything(seed)
    print(f"data={dataset_path}")
    print(f"train_dataset={split_paths['train']}")
    if split_paths["validation"] is not None:
        print(f"validation_dataset={split_paths['validation']}")
    if split_paths["test"] is not None:
        print(f"test_dataset={split_paths['test']}")
    print(f"device={resolved_device}")
    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config.get("num_workers", 0))
    pin_memory = is_cuda_device(resolved_device)
    _, train_loader = make_controller_loader(
        split_paths["train"],  # type: ignore[arg-type]
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = None
    if split_paths["validation"] is not None:
        _, validation_loader = make_controller_loader(
            split_paths["validation"],
            batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
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
        device=resolved_device,
        gradient_clip_norm=float(training_config["gradient_clip_norm"]),
    )
    batch_losses_by_epoch: list[tuple[float, ...]] = []
    epoch_losses: list[float] = []
    validation_epoch_losses: list[float] = []
    for epoch in range(resolved_epochs):
        result = trainer.train_epoch(train_loader)
        batch_losses_by_epoch.append(result.batch_losses)
        epoch_losses.append(result.loss)
        if validation_loader is not None:
            validation_loss = evaluate_loss(
                model,
                validation_loader,
                trainer.loss_function,
                resolved_device,
            )
            validation_epoch_losses.append(validation_loss)
            print(
                f"epoch={epoch + 1} train_loss={result.loss:.6f} "
                f"validation_loss={validation_loss:.6f}"
            )
        else:
            print(f"epoch={epoch + 1} train_loss={result.loss:.6f}")
    if batch_losses_by_epoch and not no_loss_plot:
        history = make_loss_history(
            batch_losses_by_epoch,
            epoch_losses,
            validation_epoch_losses if validation_epoch_losses else None,
        )
        history_path = save_loss_history_csv(
            history,
            project_path(loss_history_output),
        )
        curve_path = save_loss_curve(
            history,
            project_path(loss_curve_output),
            show_plots=show_plots,
        )
        print(f"loss_history={history_path}")
        print(f"loss_curve={curve_path}")
    output_path = project_path(output)
    save_checkpoint(output_path, model, optimizer, model_config, resolved_epochs)
    print(f"checkpoint={output_path}")
    comparisons_dir = (
        project_path(comparison_output_dir)
        if comparison_output_dir is not None
        else project_path(loss_curve_output).parent / "output_comparison"
    )
    for split_name, split_path in split_paths.items():
        if split_path is None:
            continue
        metrics = save_split_output_comparison(
            split_name=split_name,
            dataset_path=split_path,
            model=model,
            model_config=model_config,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            device=resolved_device,
            output_dir=comparisons_dir,
            max_plot_scenarios=max_comparison_scenarios,
            max_plot_samples=max_comparison_samples,
            show_plots=show_plots,
        )
        metric_summary = " ".join(
            f"{name}={value:.6f}"
            for name, value in sorted(metrics.items())
            if name != "sample_count"
        )
        print(
            f"{split_name}_evaluation sample_count={int(metrics['sample_count'])} "
            f"{metric_summary}"
        )
    if not no_showcase:
        showcase = run_typical_reference_showcase(
            model,
            output_dir=project_path(showcase_output_dir),
            project_root=PROJECT_ROOT,
            device=resolved_device,
            model_config_path=model_config_path,
            vehicle_parameters_path=vehicle_parameters_path,
            actuator_limits_path=actuator_limits_path,
            normalization_path=normalization_path,
            safety_limits_path=safety_limits_path,
            dataset_config_path=data_config_path,
            generation_config_path=generation_config_path,
            generation_time_step_s=generation_time_step_s,
            show_plots=show_plots,
        )
        if showcase.overview_plot is not None:
            print(f"showcase_overview={showcase.overview_plot}")
        for result in showcase.scenario_results:
            summary_text = " ".join(
                f"{name}={value:.6f}" for name, value in sorted(result.summary.items())
            )
            print(f"showcase_scenario={result.name} {summary_text}")
            for plot_path in result.plot_paths:
                print(f"showcase_plot={plot_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset")
    parser.add_argument("--train-dataset")
    parser.add_argument("--validation-dataset")
    parser.add_argument("--test-dataset")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output", default="artifacts/checkpoints/baseline.pt")
    parser.add_argument("--device", help="Training device. Defaults to cuda when available, else CPU/config.")
    parser.add_argument("--model-config")
    parser.add_argument("--training-config")
    parser.add_argument(
        "--loss-curve-output",
        default="artifacts/reports/imitation_training/loss_curve.png",
    )
    parser.add_argument(
        "--loss-history-output",
        default="artifacts/reports/imitation_training/loss_history.csv",
    )
    parser.add_argument("--no-loss-plot", action="store_true")
    parser.add_argument("--comparison-output-dir")
    parser.add_argument(
        "--max-comparison-scenarios",
        type=int,
        default=0,
        help="Maximum clips/scenarios to plot per split. Use 0 to plot all.",
    )
    parser.add_argument("--max-comparison-samples", type=int, default=2000)
    parser.add_argument("--showcase-output-dir", default="artifacts/reports/training_showcase")
    parser.add_argument("--no-showcase", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    train_imitation(
        config=args.config,
        dataset=args.dataset,
        train_dataset=args.train_dataset,
        validation_dataset=args.validation_dataset,
        test_dataset=args.test_dataset,
        epochs=args.epochs,
        output=args.output,
        device=args.device,
        model_config=args.model_config,
        training_config=args.training_config,
        loss_curve_output=args.loss_curve_output,
        loss_history_output=args.loss_history_output,
        no_loss_plot=args.no_loss_plot,
        comparison_output_dir=args.comparison_output_dir,
        max_comparison_scenarios=(
            None if args.max_comparison_scenarios <= 0 else args.max_comparison_scenarios
        ),
        max_comparison_samples=args.max_comparison_samples,
        showcase_output_dir=args.showcase_output_dir,
        no_showcase=args.no_showcase,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()
