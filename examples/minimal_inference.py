"""Minimal neural controller inference example."""

import numpy as np
import torch

from _bootstrap import PROJECT_ROOT  # noqa: F401
from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.models.mlp_controller import MLPController

model = MLPController().eval()
features = torch.from_numpy(np.zeros((1, FEATURE_COUNT), dtype=np.float32))
with torch.inference_mode():
    normalized_output = model(features)
print(normalized_output)
