import torch

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.models.mlp_controller import MLPController


def test_torchscript_trace_matches_eager() -> None:
    model = MLPController().eval()
    inputs = torch.randn(2, FEATURE_COUNT)
    traced = torch.jit.trace(model, torch.zeros(1, FEATURE_COUNT))
    assert torch.allclose(model(inputs), traced(inputs), atol=1e-6)
