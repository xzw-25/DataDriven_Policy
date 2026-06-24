"""Device selection helpers for training and evaluation scripts."""

from __future__ import annotations

import torch


def is_cuda_device(device: str | torch.device) -> bool:
    return torch.device(device).type == "cuda"


def preferred_training_device(
    explicit_device: str | None = None,
    configured_device: object | None = None,
) -> str:
    """Resolve the training device, preferring CUDA unless explicitly overridden."""
    if explicit_device:
        return explicit_device
    if torch.cuda.is_available():
        return "cuda"

    if configured_device is None:
        return "cpu"
    resolved = str(configured_device)
    if is_cuda_device(resolved):
        return "cpu"
    return resolved
