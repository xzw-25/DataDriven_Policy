"""Training loss history persistence and plotting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from vehicle_controller.plotting import load_pyplot

LOSS_PLOT_FLOOR = 1e-12


@dataclass(frozen=True)
class LossHistory:
    iterations: tuple[int, ...]
    batch_losses: tuple[float, ...]
    epoch_iterations: tuple[int, ...]
    epoch_losses: tuple[float, ...]
    validation_epoch_losses: tuple[float, ...] = ()


def save_loss_history_csv(history: LossHistory, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epoch_lookup = {
        iteration: loss
        for iteration, loss in zip(history.epoch_iterations, history.epoch_losses)
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        has_validation = bool(history.validation_epoch_losses)
        header = ["iteration", "batch_loss", "epoch_mean_loss"]
        if has_validation:
            header.append("validation_epoch_loss")
        writer.writerow(header)
        validation_lookup = {
            iteration: loss
            for iteration, loss in zip(
                history.epoch_iterations,
                history.validation_epoch_losses,
            )
        }
        for iteration, batch_loss in zip(history.iterations, history.batch_losses):
            row: list[object] = [iteration, batch_loss, epoch_lookup.get(iteration, "")]
            if has_validation:
                row.append(validation_lookup.get(iteration, ""))
            writer.writerow(row)
    return path


def save_loss_curve(
    history: LossHistory,
    output_path: str | Path,
    show_plots: bool = False,
) -> Path:
    if not history.iterations:
        raise ValueError("Cannot plot an empty loss history")
    if len(history.iterations) != len(history.batch_losses):
        raise ValueError("iterations and batch_losses must have matching lengths")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    iterations = np.asarray(history.iterations, dtype=np.int64)
    batch_losses = np.asarray(history.batch_losses, dtype=np.float64)
    epoch_iterations = np.asarray(history.epoch_iterations, dtype=np.int64)
    epoch_losses = np.asarray(history.epoch_losses, dtype=np.float64)
    validation_losses = np.asarray(history.validation_epoch_losses, dtype=np.float64)
    if (
        not np.all(np.isfinite(batch_losses))
        or not np.all(np.isfinite(epoch_losses))
        or not np.all(np.isfinite(validation_losses))
    ):
        raise ValueError("Loss history contains non-finite values")
    if len(validation_losses) not in (0, len(epoch_iterations)):
        raise ValueError("validation_epoch_losses must match epoch_iterations when provided")
    plot_batch_losses = np.maximum(batch_losses, LOSS_PLOT_FLOOR)
    plot_epoch_losses = np.maximum(epoch_losses, LOSS_PLOT_FLOOR)
    plot_validation_losses = np.maximum(validation_losses, LOSS_PLOT_FLOOR)

    plt = load_pyplot(show_plots)
    figure, axis = plt.subplots(1, 1, figsize=(11, 5.8))
    axis.plot(
        iterations,
        plot_batch_losses,
        linewidth=0.8,
        alpha=0.35,
        label="Batch training loss",
    )
    if len(epoch_iterations) > 0:
        axis.plot(
            epoch_iterations,
            plot_epoch_losses,
            marker="o",
            linewidth=2.0,
            label="Epoch mean training loss",
        )
    if len(plot_validation_losses) > 0:
        axis.plot(
            epoch_iterations,
            plot_validation_losses,
            marker="s",
            linewidth=2.0,
            label="Epoch mean validation loss",
        )
    axis.set_title("Imitation Learning Training Loss")
    axis.set_xlabel("Optimizer iteration")
    axis.set_ylabel("Loss")
    axis.set_yscale("log")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(figure)
    return path


def make_loss_history(
    batch_losses_by_epoch: Sequence[Sequence[float]],
    epoch_losses: Sequence[float],
    validation_epoch_losses: Sequence[float] | None = None,
) -> LossHistory:
    iterations: list[int] = []
    batch_losses: list[float] = []
    epoch_iterations: list[int] = []
    iteration = 0
    for epoch_batch_losses in batch_losses_by_epoch:
        for loss in epoch_batch_losses:
            iteration += 1
            iterations.append(iteration)
            batch_losses.append(float(loss))
        if epoch_batch_losses:
            epoch_iterations.append(iteration)
    if len(epoch_iterations) != len(epoch_losses):
        raise ValueError("epoch_losses must match non-empty epoch loss groups")
    validation_losses = (
        ()
        if validation_epoch_losses is None
        else tuple(float(loss) for loss in validation_epoch_losses)
    )
    if validation_losses and len(validation_losses) != len(epoch_iterations):
        raise ValueError("validation_epoch_losses must match non-empty epoch loss groups")
    return LossHistory(
        iterations=tuple(iterations),
        batch_losses=tuple(batch_losses),
        epoch_iterations=tuple(epoch_iterations),
        epoch_losses=tuple(float(loss) for loss in epoch_losses),
        validation_epoch_losses=validation_losses,
    )
