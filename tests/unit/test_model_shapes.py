from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.models.direct_mlp_controller import DirectMLPController
from vehicle_controller.models.gru_controller import GRUController
from vehicle_controller.models.mlp_controller import MLPController
from vehicle_controller.models.model_factory import build_model


def test_mlp_output_shape_and_bounds() -> None:
    model = build_model(Path("configs/model/mlp_controller.yaml"))
    assert isinstance(model, MLPController)
    output = model(torch.zeros(4, FEATURE_COUNT))
    assert output.shape == (4, 2)
    assert torch.all(torch.abs(output) <= 1.0)
    shared_widths = [
        layer.out_features
        for layer in model.shared
        if isinstance(layer, nn.Linear)
    ]
    assert shared_widths == [128, 64]


def test_direct_mlp_default_hidden_layers() -> None:
    model = DirectMLPController()
    output = model(torch.zeros(4, FEATURE_COUNT))
    assert output.shape == (4, 2)
    assert torch.all(torch.abs(output) <= 1.0)
    widths = [
        layer.out_features
        for layer in model.network
        if isinstance(layer, nn.Linear)
    ]
    assert widths == [128, 128, 64, 2]
    assert not hasattr(model, "trajectory_encoder")
    assert not hasattr(model, "error_encoder")
    assert not hasattr(model, "state_encoder")


def test_gru_output_shape_and_bounds() -> None:
    output = GRUController()(torch.zeros(4, 5, FEATURE_COUNT))
    assert output.shape == (4, 2)
    assert torch.all(torch.abs(output) <= 1.0)


def test_model_factory_reads_mlp_architecture_from_yaml(tmp_path) -> None:
    config_path = tmp_path / "mlp.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "type": "mlp",
                "trajectory_hidden": [24, 12],
                "error_hidden": [10],
                "state_hidden": [8],
                "shared_hidden": [20, 6],
                "head_hidden": 4,
            }
        ),
        encoding="utf-8",
    )

    model = build_model(config_path)

    assert isinstance(model, MLPController)
    assert _linear_widths(model.trajectory_encoder) == [24, 12]
    assert _linear_widths(model.error_encoder) == [10]
    assert _linear_widths(model.state_encoder) == [8]
    assert _linear_widths(model.shared) == [20, 6]
    assert _linear_widths(model.steering_head.network) == [4, 1]


def test_model_factory_reads_direct_mlp_architecture_from_yaml(tmp_path) -> None:
    config_path = tmp_path / "direct_mlp.yaml"
    config_path.write_text(
        yaml.safe_dump({"type": "direct_mlp", "hidden_sizes": [40, 20]}),
        encoding="utf-8",
    )

    model = build_model(config_path)

    assert isinstance(model, DirectMLPController)
    assert _linear_widths(model.network) == [40, 20, 2]


def test_model_factory_reads_gru_architecture_from_yaml(tmp_path) -> None:
    config_path = tmp_path / "gru.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "type": "gru",
                "hidden_size": 18,
                "num_layers": 2,
                "head_hidden": 7,
            }
        ),
        encoding="utf-8",
    )

    model = build_model(config_path)

    assert isinstance(model, GRUController)
    assert model.gru.hidden_size == 18
    assert model.gru.num_layers == 2
    assert _linear_widths(model.steering_head.network) == [7, 1]


@pytest.mark.parametrize(
    ("config_path", "expected_type"),
    [
        ("configs/model/mlp_controller.yaml", MLPController),
        ("configs/model/direct_mlp_controller.yaml", DirectMLPController),
        ("configs/model/gru_controller.yaml", GRUController),
    ],
)
def test_model_factory_builds_project_model_configs(
    config_path: str,
    expected_type: type[nn.Module],
) -> None:
    assert isinstance(build_model(config_path), expected_type)


def test_model_factory_rejects_missing_architecture_parameter() -> None:
    with pytest.raises(ValueError, match="hidden_sizes"):
        build_model({"type": "direct_mlp"})


def test_model_factory_rejects_invalid_architecture_parameter() -> None:
    with pytest.raises(ValueError, match="shared_hidden"):
        build_model(
            {
                "type": "mlp",
                "trajectory_hidden": [24],
                "error_hidden": [10],
                "state_hidden": [8],
                "shared_hidden": [],
                "head_hidden": 4,
            }
        )


def _linear_widths(module: nn.Module) -> list[int]:
    return [
        layer.out_features
        for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]
