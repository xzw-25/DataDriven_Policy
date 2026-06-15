"""Backend-independent model runtime protocol."""

from typing import Protocol

import numpy as np


class ModelRuntime(Protocol):
    def infer(self, features: np.ndarray) -> np.ndarray:
        """Infer a [batch, 2] normalized output."""

