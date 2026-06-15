"""Checkpoint persistence with interface metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from vehicle_controller.constants import FEATURE_NAMES


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    epoch: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": None if optimizer is None else optimizer.state_dict(),
            "config": config,
            "epoch": epoch,
            "feature_names": FEATURE_NAMES,
            "feature_count": len(FEATURE_NAMES),
        },
        path,
    )


def load_model_state(path: str | Path, model: nn.Module, device: str = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if tuple(checkpoint["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Checkpoint feature contract is incompatible")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint

