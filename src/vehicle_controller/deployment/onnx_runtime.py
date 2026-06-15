"""Optional ONNX Runtime backend."""

from __future__ import annotations

import numpy as np


class OnnxRuntime:
    def __init__(self, model_path: str) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("Install the deploy optional dependencies for ONNX support") from error
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, features: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: features.astype(np.float32)})[0]

