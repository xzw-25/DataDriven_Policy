"""Configuration-driven model construction."""

from collections.abc import Mapping
from pathlib import Path

from torch import nn

from vehicle_controller.models.direct_mlp_controller import DirectMLPController
from vehicle_controller.models.gru_controller import GRUController
from vehicle_controller.models.mlp_controller import MLPController
from vehicle_controller.utils.config import load_yaml


ModelConfig = str | Path | Mapping[str, object]


def _load_model_config(config: ModelConfig) -> Mapping[str, object]:
    if isinstance(config, (str, Path)):
        return load_yaml(config)
    return config


def _required(config: Mapping[str, object], key: str, model_type: str) -> object:
    if key not in config:
        raise ValueError(
            f"Missing required model parameter '{key}' for model type '{model_type}'"
        )
    return config[key]


def _positive_int(config: Mapping[str, object], key: str, model_type: str) -> int:
    value = _required(config, key, model_type)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"Model parameter '{key}' for model type '{model_type}' "
            "must be a positive integer"
        )
    return value


def _positive_int_list(
    config: Mapping[str, object],
    key: str,
    model_type: str,
) -> list[int]:
    value = _required(config, key, model_type)
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in value
        )
    ):
        raise ValueError(
            f"Model parameter '{key}' for model type '{model_type}' "
            "must be a non-empty list of positive integers"
        )
    return value


def build_model(config: ModelConfig) -> nn.Module:
    """Build a controller from a YAML path or an already loaded configuration."""
    model_config = _load_model_config(config)
    model_type_value = _required(model_config, "type", "unknown")
    if not isinstance(model_type_value, str) or not model_type_value:
        raise ValueError("Model parameter 'type' must be a non-empty string")
    model_type = model_type_value

    if model_type == "mlp":
        return MLPController(
            trajectory_hidden=_positive_int_list(
                model_config, "trajectory_hidden", model_type
            ),
            error_hidden=_positive_int_list(model_config, "error_hidden", model_type),
            state_hidden=_positive_int_list(model_config, "state_hidden", model_type),
            shared_hidden=_positive_int_list(model_config, "shared_hidden", model_type),
            head_hidden=_positive_int(model_config, "head_hidden", model_type),
        )
    if model_type == "direct_mlp":
        return DirectMLPController(
            hidden_sizes=_positive_int_list(model_config, "hidden_sizes", model_type),
        )
    if model_type == "gru":
        return GRUController(
            hidden_size=_positive_int(model_config, "hidden_size", model_type),
            num_layers=_positive_int(model_config, "num_layers", model_type),
            head_hidden=_positive_int(model_config, "head_hidden", model_type),
        )
    raise ValueError(f"Unsupported model type: {model_type}")
