from __future__ import annotations

import numpy as np

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.training.supervised_validation import (
    optional_filtered_array,
    physical_error_metrics,
    physical_targets_from_npz,
    valid_sample_mask,
)


def test_physical_targets_from_npz_filters_valid_mask(tmp_path):
    path = tmp_path / "dataset.npz"
    np.savez_compressed(
        path,
        features=np.zeros((3, FEATURE_COUNT), dtype=np.float32),
        targets=np.asarray([[0.1, 0.2], [9.0, 9.0], [0.3, 0.4]], dtype=np.float32),
        target_valid_mask=np.asarray([True, False, True]),
        timestamps_s=np.asarray([1.0, 2.0, 3.0]),
    )

    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(valid_sample_mask(data), np.asarray([True, False, True]))
        physical = physical_targets_from_npz(data, np.asarray([10.0, 2.0]))
        timestamps = optional_filtered_array(data, "timestamps_s")

    np.testing.assert_allclose(physical, np.asarray([[1.0, 0.4], [3.0, 0.8]]))
    np.testing.assert_allclose(timestamps, np.asarray([1.0, 3.0]))


def test_physical_error_metrics_reports_control_errors():
    predicted = np.asarray([[1.0, 0.0], [3.0, 4.0]])
    target = np.asarray([[0.0, 0.0], [1.0, 2.0]])

    metrics = physical_error_metrics(predicted, target)

    assert metrics["physical_steering_mae_deg"] == 1.5
    assert metrics["physical_acceleration_mae_mps2"] == 1.0
    assert metrics["physical_steering_max_abs_deg"] == 2.0
    assert metrics["physical_acceleration_max_abs_mps2"] == 2.0
