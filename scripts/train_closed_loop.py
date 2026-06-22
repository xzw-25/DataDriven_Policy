#!/usr/bin/env python3
"""Fine-tune an imitation policy with differentiable closed-loop rollout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from _bootstrap import PROJECT_ROOT
from vehicle_controller.data.synthetic_scenarios import build_typical_scenarios
from vehicle_controller.features.normalizer import FeatureNormalizer
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.training.checkpoint import load_model_state, save_checkpoint
from vehicle_controller.training.closed_loop_trainer import (
    ClosedLoopScales,
    build_reference_batch,
    closed_loop_loss_from_config,
    differentiable_closed_loop_rollout,
)
from vehicle_controller.training.loss_plots import (
    make_loss_history,
    save_loss_curve,
    save_loss_history_csv,
)
from vehicle_controller.units import steering_limit_deg_from_config
from vehicle_controller.utils.config import load_yaml
from vehicle_controller.utils.random import seed_everything
from vehicle_controller.vehicle.parameter_loader import VehicleParameters


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def config_value_path(
    config: dict[str, object],
    section: str,
    key: str,
) -> Path:
    section_config = config.get(section)
    if not isinstance(section_config, dict) or key not in section_config:
        raise ValueError(f"Missing '{section}.{key}' in the main configuration")
    return project_path(str(section_config[key]))


def nested_config_path(config: dict[str, object], section: str) -> Path:
    section_config = config.get(section)
    if not isinstance(section_config, dict) or "config" not in section_config:
        raise ValueError(f"Missing '{section}.config' in the main configuration")
    return project_path(str(section_config["config"]))


def _checkpoint_model_config(checkpoint_path: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Initial checkpoint does not contain a model config")
    return dict(config)


def _profile_batches(
    profiles: tuple[object, ...],
    batch_size: int,
    rng: np.random.Generator,
) -> list[list[object]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = np.arange(len(profiles))
    rng.shuffle(indices)
    return [
        [profiles[int(index)] for index in indices[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--closed-loop-config", default="configs/training/closed_loop.yaml")
    parser.add_argument("--initial-checkpoint", default="artifacts/checkpoints/baseline.pt")
    parser.add_argument("--output", default="artifacts/checkpoints/closed_loop.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--horizon-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device")
    parser.add_argument(
        "--loss-curve-output",
        default="artifacts/reports/closed_loop_training/loss_curve.png",
    )
    parser.add_argument(
        "--loss-history-output",
        default="artifacts/reports/closed_loop_training/loss_history.csv",
    )
    parser.add_argument("--no-loss-plot", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    main_config = load_yaml(project_path(args.config))
    closed_loop_config = load_yaml(project_path(args.closed_loop_config))
    initial_checkpoint = project_path(args.initial_checkpoint)
    if not initial_checkpoint.is_file():
        raise ValueError(f"Initial checkpoint not found: {initial_checkpoint}")

    data_config_path = nested_config_path(main_config, "data")
    dataset_config = load_yaml(data_config_path)
    normalization_path = config_value_path(main_config, "data", "normalization")
    generation_config_path = config_value_path(main_config, "data", "generation")
    vehicle_parameters_path = config_value_path(main_config, "vehicle", "parameters")
    generation_config = load_yaml(generation_config_path)

    seed = int(closed_loop_config.get("seed", main_config.get("seed", 42)))
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    device = args.device or str(closed_loop_config.get("device", main_config.get("device", "cpu")))
    epochs = args.epochs if args.epochs is not None else int(closed_loop_config.get("epochs", 10))
    horizon_steps = (
        args.horizon_steps
        if args.horizon_steps is not None
        else int(closed_loop_config["horizon_steps"])
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(closed_loop_config["batch_size"])
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(closed_loop_config["learning_rate"])
    )
    if epochs < 0:
        raise ValueError("epochs must be non-negative")

    model_config = _checkpoint_model_config(initial_checkpoint)
    model = build_model(model_config).to(device)
    load_model_state(initial_checkpoint, model, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(closed_loop_config.get("weight_decay", 0.0)),
    )
    normalizer = FeatureNormalizer.from_yaml(str(normalization_path))
    vehicle = VehicleParameters.from_yaml(str(vehicle_parameters_path))
    scales = ClosedLoopScales(
        steering_limit_deg=steering_limit_deg_from_config(model_config),
        accel_limit_mps2=float(model_config["accel_limit_mps2"]),
    )
    loss_function = closed_loop_loss_from_config(closed_loop_config)
    profiles = build_typical_scenarios(float(generation_config["time_step_s"]))
    preview_times = dataset_config.get("preview_times_s", (0.1, 0.2, 0.3, 0.4, 0.5))
    fixed_lookahead_distances = dataset_config.get("lookahead_distances_m")
    curvature_weights = dataset_config["curvature_weights"]

    print(f"initial_checkpoint={initial_checkpoint}")
    print(f"profiles={len(profiles)} horizon_steps={horizon_steps} device={device}")

    batch_losses_by_epoch: list[tuple[float, ...]] = []
    epoch_losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        batch_losses: list[float] = []
        tracking_losses: list[float] = []
        stability_losses: list[float] = []
        comfort_losses: list[float] = []
        for batch_profiles in _profile_batches(profiles, batch_size, rng):
            reference_batch = build_reference_batch(
                batch_profiles,  # type: ignore[arg-type]
                preview_times_s=preview_times,
                curvature_weights=curvature_weights,
                horizon_steps=horizon_steps,
                device=device,
                lookahead_distances_m=fixed_lookahead_distances,
            )
            optimizer.zero_grad(set_to_none=True)
            result = differentiable_closed_loop_rollout(
                model,
                reference_batch,
                normalizer,
                vehicle,
                scales,
                loss_function,
                curvature_weights,
            )
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(closed_loop_config.get("gradient_clip_norm", 5.0)),
            )
            optimizer.step()
            batch_losses.append(float(result.loss.detach().cpu().item()))
            tracking_losses.append(float(result.tracking_loss.detach().cpu().item()))
            stability_losses.append(float(result.stability_loss.detach().cpu().item()))
            comfort_losses.append(float(result.comfort_loss.detach().cpu().item()))

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        batch_losses_by_epoch.append(tuple(batch_losses))
        epoch_losses.append(epoch_loss)
        print(
            f"epoch={epoch + 1} loss={epoch_loss:.6f} "
            f"tracking={np.mean(tracking_losses):.6f} "
            f"stability={np.mean(stability_losses):.6f} "
            f"comfort={np.mean(comfort_losses):.6f}"
        )

    if batch_losses_by_epoch and not args.no_loss_plot:
        history = make_loss_history(batch_losses_by_epoch, epoch_losses)
        history_path = save_loss_history_csv(history, project_path(args.loss_history_output))
        curve_path = save_loss_curve(
            history,
            project_path(args.loss_curve_output),
            show_plots=args.show_plots,
        )
        print(f"loss_history={history_path}")
        print(f"loss_curve={curve_path}")

    output_path = project_path(args.output)
    save_checkpoint(output_path, model, optimizer, model_config, epochs)
    print(f"checkpoint={output_path}")


if __name__ == "__main__":
    main()
