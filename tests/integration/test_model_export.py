import json
import subprocess
import sys

import torch

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.models.direct_mlp_controller import DirectMLPController
from vehicle_controller.models.mlp_controller import MLPController
from vehicle_controller.training.checkpoint import save_checkpoint


def test_torchscript_trace_matches_eager() -> None:
    model = MLPController().eval()
    inputs = torch.randn(2, FEATURE_COUNT)
    traced = torch.jit.trace(model, torch.zeros(1, FEATURE_COUNT))
    assert torch.allclose(model(inputs), traced(inputs), atol=1e-6)


def test_export_model_uses_checkpoint_config_and_writes_metadata(tmp_path) -> None:
    checkpoint_path = tmp_path / "direct.pt"
    model = DirectMLPController(hidden_sizes=[16])
    model_config = {
        "type": "direct_mlp",
        "hidden_sizes": [16],
        "steering_limit_deg": 458.3662361046586,
        "accel_limit_mps2": 6.0,
    }
    save_checkpoint(checkpoint_path, model, None, model_config, epoch=1)
    output_dir = tmp_path / "exported"

    subprocess.run(
        [
            sys.executable,
            "scripts/export_model.py",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
    )

    exported = torch.jit.load(str(output_dir / "model.pt")).eval()
    with torch.inference_mode():
        output = exported(torch.zeros(4, FEATURE_COUNT))
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    normalization = json.loads((output_dir / "normalization.json").read_text(encoding="utf-8"))

    assert output.shape == (4, 2)
    assert metadata["model_type"] == "direct_mlp"
    assert metadata["input_shape"] == ["batch", FEATURE_COUNT]
    assert metadata["physical_output_scales"]["steering_limit_deg"] == 458.3662361046586
    assert normalization["feature_names"][0] == "x1"
