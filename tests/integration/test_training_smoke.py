import numpy as np
import torch
from torch.utils.data import DataLoader

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.data.dataset import ControllerDataset
from vehicle_controller.models.mlp_controller import MLPController
from vehicle_controller.training.losses import ControllerLoss
from vehicle_controller.training.trainer import Trainer


def test_single_training_epoch_runs() -> None:
    rng = np.random.default_rng(1)
    dataset = ControllerDataset(
        rng.normal(size=(32, FEATURE_COUNT)).astype(np.float32),
        rng.uniform(-1.0, 1.0, size=(32, 2)).astype(np.float32),
    )
    model = MLPController()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = Trainer(model, optimizer, ControllerLoss()).train_epoch(
        DataLoader(dataset, batch_size=8)
    )
    assert result.sample_count == 32
    assert result.loss >= 0.0
