import numpy as np

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.training.offline_plots import (
    EXPERT_ACCELERATION_LABEL,
    EXPERT_STEERING_LABEL,
    _feature_axis_label,
    _feature_plot_values,
    _preview_trajectory_points_enu,
    NEURAL_ACCELERATION_LABEL,
    NEURAL_STEERING_LABEL,
    save_feature_signal_plots,
    save_offline_control_comparison_plots,
)


def test_offline_control_plot_writes_one_file_per_selected_scenario(tmp_path) -> None:
    expert = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [0.2, 0.4],
            [1.0, -0.1],
            [1.1, -0.2],
            [1.2, -0.3],
        ],
        dtype=np.float32,
    )
    predicted = expert + 0.05
    timestamps = np.asarray([0.0, 0.1, 0.2, 0.0, 0.1, 0.2], dtype=np.float32)
    scenario_ids = np.asarray(["scenario/a"] * 3 + ["scenario/b"] * 3)

    paths = save_offline_control_comparison_plots(
        predicted,
        expert,
        tmp_path,
        timestamps_s=timestamps,
        scenario_ids=scenario_ids,
        max_scenarios=1,
    )

    assert len(paths) == 1
    assert paths[0].name == "validation_scenario_a_control_comparison.png"
    assert paths[0].is_file()


def test_offline_control_plot_writes_all_scenarios_when_unlimited(tmp_path) -> None:
    expert = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.2],
            [1.0, -0.1],
            [1.1, -0.2],
            [2.0, 0.4],
            [2.1, 0.5],
        ],
        dtype=np.float32,
    )
    predicted = expert + 0.05
    scenario_ids = np.asarray(["scenario/a"] * 2 + ["scenario/b"] * 2 + ["scenario/c"] * 2)

    paths = save_offline_control_comparison_plots(
        predicted,
        expert,
        tmp_path,
        scenario_ids=scenario_ids,
        max_scenarios=None,
    )

    assert [path.name for path in paths] == [
        "validation_scenario_a_control_comparison.png",
        "validation_scenario_b_control_comparison.png",
        "validation_scenario_c_control_comparison.png",
    ]
    assert all(path.is_file() for path in paths)


def test_offline_control_plot_labels_name_expert_and_neural_outputs() -> None:
    assert EXPERT_STEERING_LABEL == "Generated-data expert steering"
    assert NEURAL_STEERING_LABEL == "Neural controller steering"
    assert EXPERT_ACCELERATION_LABEL == "Generated-data expert signed acceleration"
    assert NEURAL_ACCELERATION_LABEL == "Neural controller signed acceleration"


def test_feature_signal_plot_writes_all_scenarios_when_unlimited(tmp_path) -> None:
    features = np.arange(6 * FEATURE_COUNT, dtype=np.float32).reshape(6, FEATURE_COUNT)
    scenario_ids = np.asarray(["scenario/a"] * 2 + ["scenario/b"] * 2 + ["scenario/c"] * 2)

    paths = save_feature_signal_plots(
        features,
        tmp_path,
        scenario_ids=scenario_ids,
        max_scenarios=None,
    )

    assert [path.name for path in paths] == [
        "validation_scenario_a_feature_preview_xy.png",
        "validation_scenario_a_feature_reference_errors.png",
        "validation_scenario_a_feature_vehicle_state.png",
        "validation_scenario_b_feature_preview_xy.png",
        "validation_scenario_b_feature_reference_errors.png",
        "validation_scenario_b_feature_vehicle_state.png",
        "validation_scenario_c_feature_preview_xy.png",
        "validation_scenario_c_feature_reference_errors.png",
        "validation_scenario_c_feature_vehicle_state.png",
    ]
    assert all(path.is_file() for path in paths)


def test_feature_axis_label_adds_expected_units() -> None:
    assert _feature_axis_label("x1") == "x1 [m]"
    assert _feature_axis_label("e_v") == "e_v [m/s]"
    assert _feature_axis_label("a_ref") == "a_ref [m/s^2]"
    assert _feature_axis_label("r") == "r [deg/s]"


def test_feature_plot_values_converts_yaw_rate_to_deg_per_second() -> None:
    features = np.zeros((2, FEATURE_COUNT), dtype=np.float64)
    features[:, -1] = np.asarray([0.0, np.pi / 2.0], dtype=np.float64)

    converted = _feature_plot_values(features, FEATURE_COUNT - 1, "r")

    np.testing.assert_allclose(converted, np.asarray([0.0, 90.0], dtype=np.float64))


def test_preview_trajectory_points_restore_to_enu_frame() -> None:
    features = np.zeros((1, FEATURE_COUNT), dtype=np.float64)
    features[0, 0:10] = np.asarray([1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0])
    positions_enu = np.asarray([[10.0, 20.0]], dtype=np.float64)
    headings_rad = np.asarray([np.pi / 2.0], dtype=np.float64)

    preview_x_enu, preview_y_enu = _preview_trajectory_points_enu(
        features,
        positions_enu,
        headings_rad,
    )

    np.testing.assert_allclose(preview_x_enu[0], np.asarray([10.0, 10.0, 10.0, 10.0, 10.0]))
    np.testing.assert_allclose(preview_y_enu[0], np.asarray([21.0, 22.0, 23.0, 24.0, 25.0]))
