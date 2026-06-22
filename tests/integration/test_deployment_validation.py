import subprocess
import sys

import pytest

from vehicle_controller.deployment.validation import validate_deployment_package
from vehicle_controller.models.direct_mlp_controller import DirectMLPController
from vehicle_controller.training.checkpoint import save_checkpoint


def _write_checkpoint(path, hidden_sizes=None) -> None:
    hidden_sizes = hidden_sizes or [16]
    model = DirectMLPController(hidden_sizes=hidden_sizes)
    save_checkpoint(
        path,
        model,
        None,
        {
            "type": "direct_mlp",
            "hidden_sizes": hidden_sizes,
            "steering_limit_deg": 458.3662361046586,
            "accel_limit_mps2": 6.0,
        },
        epoch=1,
    )


def _export(checkpoint_path, output_dir, export_format: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/export_model.py",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--format",
            export_format,
        ],
        check=True,
    )


def test_validate_deployment_package_accepts_torchscript_export(tmp_path) -> None:
    checkpoint_path = tmp_path / "controller.pt"
    output_dir = tmp_path / "exported"
    _write_checkpoint(checkpoint_path)
    _export(checkpoint_path, output_dir, "torchscript")

    result = validate_deployment_package(output_dir, batch_size=3)

    assert result.input_shape == (3, 22)
    assert result.torch_output_shape == (3, 2)
    assert result.onnx_output_shape is None
    assert result.maximum_abs_output <= 1.05


def test_validate_deployment_package_compares_onnx_when_available(tmp_path) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    checkpoint_path = tmp_path / "controller.pt"
    output_dir = tmp_path / "exported"
    _write_checkpoint(checkpoint_path)
    _export(checkpoint_path, output_dir, "both")

    result = validate_deployment_package(output_dir, batch_size=2)

    assert result.torch_output_shape == (2, 2)
    assert result.onnx_output_shape == (2, 2)
    assert result.maximum_torch_onnx_error is not None
    assert result.maximum_torch_onnx_error <= 1e-5


def test_validate_deployment_package_rejects_bad_feature_contract(tmp_path) -> None:
    checkpoint_path = tmp_path / "controller.pt"
    output_dir = tmp_path / "exported"
    _write_checkpoint(checkpoint_path)
    _export(checkpoint_path, output_dir, "torchscript")
    metadata_path = output_dir / "metadata.json"
    text = metadata_path.read_text(encoding="utf-8")
    metadata_path.write_text(text.replace('"feature_count": 22', '"feature_count": 21'), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_count"):
        validate_deployment_package(output_dir)
