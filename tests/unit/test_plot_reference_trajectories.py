from __future__ import annotations

import numpy as np

from scripts.plot_reference_trajectories import (
    reference_points_for_frame,
    selected_frame_indices,
)


def test_selected_frame_indices_keeps_endpoints():
    np.testing.assert_array_equal(
        selected_frame_indices(frame_count=10, maximum_frames=4),
        np.array([0, 3, 6, 9]),
    )


def test_reference_points_for_frame_uses_valid_length():
    x_values = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    y_values = np.array([[4.0, 5.0, 6.0], [40.0, 50.0, 60.0]])
    valid_lengths = np.array([2, 3])

    points = reference_points_for_frame(x_values, y_values, valid_lengths, frame_index=0)

    np.testing.assert_array_equal(points, np.array([[1.0, 4.0], [2.0, 5.0]]))
