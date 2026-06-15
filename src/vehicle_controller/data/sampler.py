"""Balanced sample weight helpers."""

import numpy as np


def inverse_frequency_weights(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    indices = np.clip(np.digitize(values, bins) - 1, 0, len(bins) - 2)
    counts = np.bincount(indices, minlength=len(bins) - 1)
    return 1.0 / np.maximum(counts[indices], 1)

