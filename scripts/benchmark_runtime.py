#!/usr/bin/env python3
"""Benchmark raw neural model inference."""

from time import perf_counter

import numpy as np
import torch

from _bootstrap import PROJECT_ROOT  # noqa: F401
from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.models.mlp_controller import MLPController


def main() -> None:
    model = MLPController().eval()
    features = torch.zeros(1, FEATURE_COUNT)
    durations = []
    with torch.inference_mode():
        for _ in range(1000):
            start = perf_counter()
            model(features)
            durations.append((perf_counter() - start) * 1000.0)
    values = np.asarray(durations)
    print(f"mean_ms={values.mean():.4f}")
    print(f"p99_ms={np.quantile(values, 0.99):.4f}")
    print(f"max_ms={values.max():.4f}")


if __name__ == "__main__":
    main()
