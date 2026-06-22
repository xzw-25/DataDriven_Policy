import numpy as np

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.data.dataset import ControllerDataset
from vehicle_controller.data.expert_controller import ExpertController
from vehicle_controller.data.simulation_generator import SimulationDataGenerator
from vehicle_controller.data.synthetic_scenarios import build_typical_scenarios
from vehicle_controller.features.normalizer import FeatureNormalizer
from vehicle_controller.types import Pose2D, VehicleState
from vehicle_controller.vehicle.parameter_loader import ActuatorLimits, VehicleParameters


def test_typical_scenarios_cover_turn_stop_and_lane_change() -> None:
    profiles = {profile.name: profile for profile in build_typical_scenarios(0.1)}

    assert {
        "left_turn",
        "right_turn",
        "stop_go",
        "lane_change_left",
        "lane_change_right",
        "double_lane_change",
    } <= profiles.keys()
    assert max(point.kappa for point in profiles["left_turn"].points) > 0.0
    assert min(point.kappa for point in profiles["right_turn"].points) < 0.0
    assert np.any(profiles["stop_go"].speed_mps == 0.0)
    assert max(point.y for point in profiles["lane_change_left"].points) > 3.0
    assert min(point.y for point in profiles["lane_change_right"].points) < -3.0


def test_simulation_generator_produces_training_compatible_arrays(tmp_path) -> None:
    normalizer = FeatureNormalizer(
        mean=np.zeros(FEATURE_COUNT),
        std=np.ones(FEATURE_COUNT) * 10.0,
    )
    generator = SimulationDataGenerator(
        vehicle=VehicleParameters(),
        actuator_limits=ActuatorLimits(),
        preview_times_s=(0.1, 0.2, 0.3, 0.4, 0.5),
        curvature_weights=(1.0, 0.8, 0.6, 0.4, 0.2),
        steering_scale_deg=458.3662361046586,
        acceleration_scale_mps2=6.0,
        time_step_s=0.1,
        normalizer=normalizer,
        seed=1,
    )

    dataset = generator.generate(
        build_typical_scenarios(0.1)[:1],
        repetitions=1,
        randomize_initial_state=False,
    )

    assert dataset.features.shape[1] == FEATURE_COUNT
    assert dataset.raw_features.shape == dataset.features.shape
    assert dataset.targets.shape == (len(dataset.features), 2)
    assert dataset.physical_targets.shape == dataset.targets.shape
    assert np.all(np.isfinite(dataset.features))
    assert np.all(np.isfinite(dataset.targets))
    assert np.max(np.abs(dataset.targets)) <= 1.0
    assert set(dataset.scenario_ids) == {"left_turn_000_lat_0"}

    output = dataset.save_npz(tmp_path / "generated.npz", {"source": "test"})
    loaded = ControllerDataset.from_npz(output)
    assert len(loaded) == len(dataset.features)


def test_simulation_generator_samples_symmetric_lateral_offsets() -> None:
    normalizer = FeatureNormalizer(
        mean=np.zeros(FEATURE_COUNT),
        std=np.ones(FEATURE_COUNT) * 10.0,
    )
    generator = SimulationDataGenerator(
        vehicle=VehicleParameters(),
        actuator_limits=ActuatorLimits(),
        preview_times_s=(0.1, 0.2, 0.3, 0.4, 0.5),
        curvature_weights=(1.0, 0.8, 0.6, 0.4, 0.2),
        steering_scale_deg=458.3662361046586,
        acceleration_scale_mps2=6.0,
        time_step_s=0.1,
        normalizer=normalizer,
        seed=1,
    )

    dataset = generator.generate(
        build_typical_scenarios(0.1)[:1],
        repetitions=1,
        randomize_initial_state=False,
        lateral_offset_samples_m=(0.1, 0.5, 1.0, 2.0),
        maximum_lateral_offset_m=2.0,
    )

    scenario_ids = sorted(set(dataset.scenario_ids))
    assert scenario_ids == [
        "left_turn_000_lat_0",
        "left_turn_000_lat_n0p1",
        "left_turn_000_lat_n0p5",
        "left_turn_000_lat_n1",
        "left_turn_000_lat_n2",
        "left_turn_000_lat_p0p1",
        "left_turn_000_lat_p0p5",
        "left_turn_000_lat_p1",
        "left_turn_000_lat_p2",
    ]

    initial_lateral_errors = {}
    for scenario_id in scenario_ids:
        mask = dataset.scenario_ids == scenario_id
        initial_lateral_errors[scenario_id] = float(dataset.raw_features[mask, 11][0])

    assert np.isclose(initial_lateral_errors["left_turn_000_lat_0"], 0.0)
    assert np.isclose(initial_lateral_errors["left_turn_000_lat_p2"], -2.0, atol=1e-4)
    assert np.isclose(initial_lateral_errors["left_turn_000_lat_n2"], 2.0, atol=1e-4)


def test_expert_controller_steers_in_turn_direction() -> None:
    profiles = {profile.name: profile for profile in build_typical_scenarios(0.1)}
    outputs = {}
    for name in ("left_turn", "right_turn"):
        profile = profiles[name]
        index = int(np.argmax(np.abs([point.kappa for point in profile.points])))
        point = profile.points[index]
        previous = profile.points[index - 1]
        following = profile.points[index + 1]
        yaw = np.arctan2(following.y - previous.y, following.x - previous.x)
        state = VehicleState(
            Pose2D(point.x, point.y, float(yaw)),
            vx=5.0,
            vy=0.0,
            ax=0.0,
            ay=0.0,
            r=0.0,
            s=point.s,
        )
        outputs[name] = ExpertController(
            VehicleParameters(),
            ActuatorLimits(),
        ).compute(
            profile.points,
            state,
            reference_s_m=point.s,
            reference_speed_mps=5.0,
            reference_acceleration_mps2=0.0,
            dt=0.1,
        )

    assert outputs["left_turn"].steering_des_rad > 0.0
    assert outputs["right_turn"].steering_des_rad < 0.0


def test_expert_longitudinal_controller_accelerates_and_brakes() -> None:
    points = build_typical_scenarios(0.1)[2].points
    vehicle = VehicleParameters()
    limits = ActuatorLimits()
    accelerate = ExpertController(vehicle, limits).compute(
        points,
        VehicleState(Pose2D(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0),
        reference_s_m=2.0,
        reference_speed_mps=4.0,
        reference_acceleration_mps2=1.0,
        dt=0.1,
    )
    brake = ExpertController(vehicle, limits).compute(
        points,
        VehicleState(Pose2D(0.0, 0.0, 0.0), 5.0, 0.0, 0.0, 0.0, 0.0),
        reference_s_m=0.0,
        reference_speed_mps=0.0,
        reference_acceleration_mps2=-1.0,
        dt=0.1,
    )

    assert accelerate.signed_accel_des_mps2 > 0.0
    assert brake.signed_accel_des_mps2 < 0.0
