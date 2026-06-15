"""TorchScript runtime."""

from __future__ import annotations

import numpy as np
import torch


class TorchRuntime:
    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.model = torch.jit.load(model_path, map_location=self.device).eval()

    def infer(self, features: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            return self.model(tensor).cpu().numpy()

