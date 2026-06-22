"""Deployment package validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vehicle_controller.constants import FEATURE_COUNT, FEATURE_NAMES
from vehicle_controller.deployment.onnx_runtime import OnnxRuntime
from vehicle_controller.deployment.torch_runtime import TorchRuntime


@dataclass(frozen=True)
class DeploymentValidationResult:
    package_dir: Path
    input_shape: tuple[int, ...]
    torch_output_shape: tuple[int, ...] | None
    onnx_output_shape: tuple[int, ...] | None
    maximum_abs_output: float
    maximum_torch_onnx_error: float | None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing deployment file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Deployment file must contain a JSON object: {path}")
    return data


def _validate_metadata(metadata: dict[str, Any]) -> None:
    if int(metadata.get("feature_count", -1)) != FEATURE_COUNT:
        raise ValueError("metadata feature_count does not match FEATURE_COUNT")
    if tuple(metadata.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("metadata feature_names do not match FEATURE_NAMES")
    if metadata.get("output_names") != [
        "steering_normalized",
        "signed_accel_normalized",
    ]:
        raise ValueError("metadata output_names do not match controller contract")
    output_scales = metadata.get("physical_output_scales")
    if not isinstance(output_scales, dict):
        raise ValueError("metadata physical_output_scales must be an object")
    if float(output_scales.get("steering_limit_deg", 0.0)) <= 0.0:
        raise ValueError("metadata physical_output_scales.steering_limit_deg is required")
    if float(output_scales.get("accel_limit_mps2", 0.0)) <= 0.0:
        raise ValueError("metadata physical_output_scales.accel_limit_mps2 is required")


def _validate_normalization(normalization: dict[str, Any]) -> None:
    if tuple(normalization.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("normalization feature_names do not match FEATURE_NAMES")
    mean = np.asarray(normalization.get("mean"), dtype=np.float32)
    std = np.asarray(normalization.get("std"), dtype=np.float32)
    if mean.shape != (FEATURE_COUNT,) or std.shape != (FEATURE_COUNT,):
        raise ValueError("normalization mean/std must match FEATURE_COUNT")
    if np.any(std <= 0.0):
        raise ValueError("normalization std values must be positive")
    if float(normalization.get("clip", 0.0)) <= 0.0:
        raise ValueError("normalization clip must be positive")


def _concrete_input_shape(metadata: dict[str, Any], batch_size: int) -> tuple[int, ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    raw_shape = metadata.get("input_shape")
    if not isinstance(raw_shape, list) or not raw_shape:
        raise ValueError("metadata input_shape must be a non-empty list")

    shape: list[int] = []
    for dimension in raw_shape:
        if dimension == "batch":
            shape.append(batch_size)
        elif isinstance(dimension, int) and dimension > 0:
            shape.append(dimension)
        else:
            raise ValueError(f"Unsupported input_shape dimension: {dimension!r}")
    if shape[-1] != FEATURE_COUNT:
        raise ValueError("input_shape last dimension must match FEATURE_COUNT")
    return tuple(shape)


def _validate_output(name: str, output: np.ndarray, batch_size: int) -> None:
    if output.shape != (batch_size, 2):
        raise ValueError(f"{name} output must have shape [{batch_size}, 2], got {output.shape}")
    if not np.all(np.isfinite(output)):
        raise ValueError(f"{name} output contains non-finite values")
    if np.max(np.abs(output)) > 1.05:
        raise ValueError(f"{name} normalized output is outside the expected [-1, 1] range")


def validate_deployment_package(
    package_dir: str | Path,
    batch_size: int = 4,
    compare_onnx: bool = True,
    comparison_atol: float = 1e-5,
    device: str = "cpu",
) -> DeploymentValidationResult:
    """Validate an exported controller package can be loaded and run for deployment."""
    root = Path(package_dir)
    metadata = _read_json(root / "metadata.json")
    normalization = _read_json(root / str(metadata.get("normalization_file", "normalization.json")))
    _validate_metadata(metadata)
    _validate_normalization(normalization)

    input_shape = _concrete_input_shape(metadata, batch_size)
    features = np.zeros(input_shape, dtype=np.float32)
    maximum_abs_output = 0.0

    torch_output = None
    torch_file = metadata.get("torchscript_file")
    if torch_file is not None:
        torch_path = root / str(torch_file)
        if not torch_path.is_file():
            raise ValueError(f"Missing TorchScript model: {torch_path}")
        torch_output = TorchRuntime(str(torch_path), device=device).infer(features)
        _validate_output("TorchScript", torch_output, batch_size)
        maximum_abs_output = max(maximum_abs_output, float(np.max(np.abs(torch_output))))

    onnx_output = None
    maximum_torch_onnx_error = None
    onnx_file = metadata.get("onnx_file")
    if compare_onnx and onnx_file is not None:
        onnx_path = root / str(onnx_file)
        if not onnx_path.is_file():
            raise ValueError(f"Missing ONNX model: {onnx_path}")
        onnx_output = OnnxRuntime(str(onnx_path)).infer(features)
        _validate_output("ONNX", onnx_output, batch_size)
        maximum_abs_output = max(maximum_abs_output, float(np.max(np.abs(onnx_output))))
        if torch_output is not None:
            maximum_torch_onnx_error = float(np.max(np.abs(torch_output - onnx_output)))
            if maximum_torch_onnx_error > comparison_atol:
                raise ValueError(
                    "TorchScript and ONNX outputs diverge: "
                    f"max_error={maximum_torch_onnx_error}"
                )

    if torch_output is None and onnx_output is None:
        raise ValueError("Deployment package does not contain a runnable model")

    return DeploymentValidationResult(
        package_dir=root,
        input_shape=input_shape,
        torch_output_shape=None if torch_output is None else tuple(torch_output.shape),
        onnx_output_shape=None if onnx_output is None else tuple(onnx_output.shape),
        maximum_abs_output=maximum_abs_output,
        maximum_torch_onnx_error=maximum_torch_onnx_error,
    )
