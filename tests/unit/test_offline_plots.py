import numpy as np

from vehicle_controller.training.offline_plots import (
    EXPERT_ACCELERATION_LABEL,
    EXPERT_STEERING_LABEL,
    NEURAL_ACCELERATION_LABEL,
    NEURAL_STEERING_LABEL,
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
