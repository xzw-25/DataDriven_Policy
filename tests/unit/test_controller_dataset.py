from __future__ import annotations

import numpy as np

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.data.dataset import ControllerDataset


def test_controller_dataset_from_npz_filters_invalid_targets(tmp_path):
    path = tmp_path / "dataset.npz"
    np.savez_compressed(
        path,
        features=np.arange(3 * FEATURE_COUNT, dtype=np.float32).reshape(3, FEATURE_COUNT),
        targets=np.asarray([[0.1, 0.2], [np.nan, np.nan], [0.3, 0.4]], dtype=np.float32),
        target_valid_mask=np.asarray([True, False, True]),
    )

    dataset = ControllerDataset.from_npz(path)

    assert len(dataset) == 2
    _, first_target = dataset[0]
    _, second_target = dataset[1]
    np.testing.assert_allclose(first_target.numpy(), np.asarray([0.1, 0.2], dtype=np.float32))
    np.testing.assert_allclose(second_target.numpy(), np.asarray([0.3, 0.4], dtype=np.float32))
