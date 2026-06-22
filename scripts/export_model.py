#!/usr/bin/env python3
"""Export a trained neural vehicle controller for deployment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import torch

from _bootstrap import PROJECT_ROOT
from vehicle_controller.constants import FEATURE_COUNT, FEATURE_NAMES
from vehicle_controller.models.model_factory import build_model
from vehicle_controller.training.checkpoint import load_model_state
from vehicle_controller.units import (
    steering_limit_deg_from_config,
    steering_limit_rad_from_config,
)
from vehicle_controller.utils.config import load_yaml


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_checkpoint_config(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("Checkpoint does not contain a model config")
    return dict(config)


def _resolve_model_config(
    checkpoint_path: Path,
    model_config_path: str | None,
) -> dict[str, Any]:
    if model_config_path is not None:
        return dict(load_yaml(project_path(model_config_path)))
    return _load_checkpoint_config(checkpoint_path)


def _normalization_metadata(path: Path) -> dict[str, Any]:
    config = dict(load_yaml(path))
    names = tuple(str(name) for name in config["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError("Normalization feature order does not match FEATURE_NAMES")
    return {
        "version": config.get("version", "unknown"),
        "feature_names": list(names),
        "mean": [float(value) for value in config["mean"]],
        "std": [float(value) for value in config["std"]],
        "clip": float(config.get("clip", 5.0)),
    }


def _metadata(
    checkpoint_path: Path,
    model_config: dict[str, Any],
    normalization: dict[str, Any],
    formats: list[str],
    input_shape: list[int | str],
    onnx_opset_version: int,
) -> dict[str, Any]:
    return {
        "model_version": "controller_v003",
        "model_type": model_config["type"],
        "export_formats": formats,
        "checkpoint": str(checkpoint_path),
        "feature_version": normalization["version"],
        "feature_count": FEATURE_COUNT,
        "feature_names": list(FEATURE_NAMES),
        "input_tensor_name": "normalized_features",
        "input_shape": input_shape,
        "output_names": ["steering_normalized", "signed_accel_normalized"],
        "output_tensor_name": "normalized_controls",
        "output_units": ["normalized_steering_deg", "normalized_signed_acceleration"],
        "physical_output_scales": {
            "steering_limit_deg": steering_limit_deg_from_config(model_config),
            "accel_limit_mps2": float(model_config["accel_limit_mps2"]),
        },
        "vehicle_interface_scales": {
            "steering_limit_rad": steering_limit_rad_from_config(model_config),
        },
        "normalization_file": "normalization.json",
        "torchscript_file": "model.pt" if "torchscript" in formats else None,
        "onnx_file": "model.onnx" if "onnx" in formats else None,
        "onnx_opset_version": onnx_opset_version if "onnx" in formats else None,
        "model_config": model_config,
    }


def _trace_torchscript(model: torch.nn.Module, example: torch.Tensor, output_path: Path) -> None:
    traced = torch.jit.trace(model, example)
    traced.save(str(output_path))


def _dynamic_axes(example: torch.Tensor) -> dict[str, dict[int, str]]:
    input_axes = {0: "batch"}
    if example.ndim == 3:
        input_axes[1] = "sequence"
    return {
        "normalized_features": input_axes,
        "normalized_controls": {0: "batch"},
    }


def _export_onnx(
    model: torch.nn.Module,
    example: torch.Tensor,
    output_path: Path,
    opset_version: int,
) -> None:
    try:
        import onnx  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "ONNX export requires the deploy extra dependencies: "
            "python3 -m pip install -e '.[deploy]'"
        ) from error
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using the legacy TorchScript-based ONNX export.*",
            category=DeprecationWarning,
        )
        torch.onnx.export(
            model,
            example,
            str(output_path),
            input_names=["normalized_features"],
            output_names=["normalized_controls"],
            dynamic_axes=_dynamic_axes(example),
            opset_version=opset_version,
            dynamo=False,
        )


def _verify_torchscript(
    model: torch.nn.Module,
    output_path: Path,
    example: torch.Tensor,
    atol: float,
) -> None:
    scripted = torch.jit.load(str(output_path), map_location="cpu").eval()
    with torch.inference_mode():
        expected = model(example)
        actual = scripted(example)
    if not torch.allclose(expected, actual, atol=atol, rtol=0.0):
        maximum_error = torch.max(torch.abs(expected - actual)).item()
        raise RuntimeError(f"TorchScript verification failed: max_error={maximum_error}")


def _verify_onnx(
    model: torch.nn.Module,
    output_path: Path,
    example: torch.Tensor,
    atol: float,
) -> None:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise SystemExit(
            "ONNX verification requires onnxruntime: python3 -m pip install -e '.[deploy]'"
        ) from error
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    with torch.inference_mode():
        expected = model(example).cpu().numpy()
    actual = session.run(None, {input_name: example.cpu().numpy().astype(np.float32)})[0]
    if not np.allclose(expected, actual, atol=atol, rtol=0.0):
        maximum_error = float(np.max(np.abs(expected - actual)))
        raise RuntimeError(f"ONNX verification failed: max_error={maximum_error}")


def _example_input(
    model_config: dict[str, Any],
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, list[int | str]]:
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    model_type = str(model_config["type"])
    if model_type == "gru":
        sequence_length = int(model_config.get("sequence_length", 1))
        if sequence_length <= 0:
            raise ValueError("GRU sequence_length must be positive")
        shape = [batch_size, sequence_length, FEATURE_COUNT]
        metadata_shape: list[int | str] = ["batch", sequence_length, FEATURE_COUNT]
    else:
        shape = [batch_size, FEATURE_COUNT]
        metadata_shape = ["batch", FEATURE_COUNT]
    return torch.zeros(*shape, dtype=torch.float32, device=device), metadata_shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output-dir", default="artifacts/exported_models/controller")
    parser.add_argument(
        "--model-config",
        help="Optional model config override. By default the checkpoint config is used.",
    )
    parser.add_argument("--normalization", default="configs/data/normalization.yaml")
    parser.add_argument(
        "--format",
        choices=("torchscript", "onnx", "both"),
        default="torchscript",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--onnx-opset", type=int, default=18)
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", action="store_false", dest="verify")
    parser.add_argument("--verify-atol", type=float, default=1e-5)
    args = parser.parse_args()

    checkpoint_path = project_path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint not found: {checkpoint_path}")
    output = project_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    model_config = _resolve_model_config(checkpoint_path, args.model_config)
    model = build_model(model_config).to(args.device).eval()
    load_model_state(checkpoint_path, model, device=args.device)

    example, input_shape = _example_input(model_config, args.batch_size, args.device)
    formats = ["torchscript", "onnx"] if args.format == "both" else [args.format]

    exported_files: list[Path] = []
    if "torchscript" in formats:
        torchscript_path = output / "model.pt"
        _trace_torchscript(model, example, torchscript_path)
        if args.verify:
            _verify_torchscript(model.cpu().eval(), torchscript_path, example.cpu(), args.verify_atol)
            model.to(args.device).eval()
        exported_files.append(torchscript_path)

    if "onnx" in formats:
        onnx_path = output / "model.onnx"
        _export_onnx(model.cpu().eval(), example.cpu(), onnx_path, int(args.onnx_opset))
        if args.verify:
            _verify_onnx(model.cpu().eval(), onnx_path, example.cpu(), args.verify_atol)
        model.to(args.device).eval()
        exported_files.append(onnx_path)

    normalization = _normalization_metadata(project_path(args.normalization))
    normalization_path = output / "normalization.json"
    normalization_path.write_text(
        json.dumps(normalization, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    metadata = _metadata(
        checkpoint_path,
        model_config,
        normalization,
        formats,
        input_shape,
        int(args.onnx_opset),
    )
    metadata_path = output / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"output_dir={output}")
    for path in exported_files:
        print(f"exported={path}")
    print(f"metadata={metadata_path}")
    print(f"normalization={normalization_path}")


if __name__ == "__main__":
    main()
