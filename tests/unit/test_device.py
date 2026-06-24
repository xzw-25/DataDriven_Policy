from __future__ import annotations

import torch

from vehicle_controller.utils.device import is_cuda_device, preferred_training_device


def test_preferred_training_device_uses_explicit_device(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert preferred_training_device("cpu", "cuda") == "cpu"


def test_preferred_training_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert preferred_training_device(configured_device="cpu") == "cuda"


def test_preferred_training_device_falls_back_from_configured_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert preferred_training_device(configured_device="cuda") == "cpu"


def test_is_cuda_device_accepts_indexed_cuda_device() -> None:
    assert is_cuda_device("cuda:0")
