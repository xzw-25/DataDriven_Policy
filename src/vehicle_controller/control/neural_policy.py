"""Neural policy inference wrapper."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from vehicle_controller.features.normalizer import FeatureNormalizer
from vehicle_controller.types import ControllerFeatures, NeuralPolicyOutput


class NeuralPolicy:
    def __init__(
        self,
        model: nn.Module,
        normalizer: FeatureNormalizer,
        steering_limit_deg: float,
        accel_limit_mps2: float,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.normalizer = normalizer
        self.steering_limit_deg = steering_limit_deg
        self.accel_limit_mps2 = accel_limit_mps2
        self.device = torch.device(device)

    def predict(self, features: ControllerFeatures) -> NeuralPolicyOutput:
        normalized = self.normalizer.normalize(features)
        tensor = torch.from_numpy(np.asarray(normalized, dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            output = self.model(tensor.to(self.device)).cpu().numpy()[0]
        return NeuralPolicyOutput(
            steering_des_deg=float(output[0] * self.steering_limit_deg),
            signed_accel_des_mps2=float(output[1] * self.accel_limit_mps2),
        )
