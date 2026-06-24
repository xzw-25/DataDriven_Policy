#!/usr/bin/env python3
"""Train the baseline MLP from an NPZ dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

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
from vehicle_controller.training.losses import ControllerLoss
from vehicle_controller.training.loss_plots import (
    make_loss_history,
    save_loss_curve,
    save_loss_history_csv,
)
from vehicle_controller.training.trainer import Trainer
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


def train_imitation(
    *,
    config: str | Path = "configs/default.yaml",
    dataset: str | Path | None = None,
    epochs: int | None = None,
    output: str | Path = "artifacts/checkpoints/baseline.pt",
    device: str | None = None,
    model_config: str | Path | None = None,
    training_config: str | Path | None = None,
    loss_curve_output: str | Path = "artifacts/reports/imitation_training/loss_curve.png",
    loss_history_output: str | Path = "artifacts/reports/imitation_training/loss_history.csv",
    no_loss_plot: bool = False,
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
    if resolved_epochs < 0:
        raise ValueError("epochs must be non-negative")

    seed_everything(seed)
    print(f"data: {dataset_path}")
    print(f"device={resolved_device}")
    dataset = ControllerDataset.from_npz(dataset_path)
    loader = DataLoader(
        dataset,
        batch_size=min(int(training_config["batch_size"]), len(dataset)),
        shuffle=True,
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=is_cuda_device(resolved_device),
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
    for epoch in range(resolved_epochs):
        result = trainer.train_epoch(loader)
        batch_losses_by_epoch.append(result.batch_losses)
        epoch_losses.append(result.loss)
        print(f"epoch={epoch + 1} loss={result.loss:.6f}")
    if batch_losses_by_epoch and not no_loss_plot:
        history = make_loss_history(batch_losses_by_epoch, epoch_losses)
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
    parser.add_argument("--showcase-output-dir", default="artifacts/reports/training_showcase")
    parser.add_argument("--no-showcase", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    train_imitation(
        config=args.config,
        dataset=args.dataset,
        epochs=args.epochs,
        output=args.output,
        device=args.device,
        model_config=args.model_config,
        training_config=args.training_config,
        loss_curve_output=args.loss_curve_output,
        loss_history_output=args.loss_history_output,
        no_loss_plot=args.no_loss_plot,
        showcase_output_dir=args.showcase_output_dir,
        no_showcase=args.no_showcase,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()
