"""Closed-loop expert simulation for neural-controller training data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from vehicle_controller.data.expert_controller import ExpertController
from vehicle_controller.data.synthetic_scenarios import (
    ReferenceProfile,
    initial_state_from_reference_profile,
)
from vehicle_controller.features.error_calculator import (
    calculate_tracking_errors,
    nearest_trajectory_index,
)
from vehicle_controller.features.feature_builder import FeatureBuilder
from vehicle_controller.features.normalizer import FeatureNormalizer
from vehicle_controller.geometry.coordinate_transform import global_to_body
from vehicle_controller.geometry.curvature import resolve_reference_curvature
from vehicle_controller.geometry.trajectory_sampler import (
    DEFAULT_PREVIEW_TIMES_S,
    preview_distances_from_times,
    sample_trajectory,
)
from vehicle_controller.types import VehicleState
from vehicle_controller.vehicle.dynamics import KinematicBicycleModel
from vehicle_controller.vehicle.parameter_loader import ActuatorLimits, VehicleParameters


@dataclass(frozen=True)
class GeneratedDataset:
    features: np.ndarray
    raw_features: np.ndarray
    targets: np.ndarray
    physical_targets: np.ndarray
    scenario_ids: np.ndarray
    timestamps_s: np.ndarray

    def save_npz(self, path: str | Path, metadata: Mapping[str, object]) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            features=self.features,
            raw_features=self.raw_features,
            targets=self.targets,
            physical_targets=self.physical_targets,
            scenario_ids=self.scenario_ids,
            timestamps_s=self.timestamps_s,
            metadata_json=np.asarray(json.dumps(dict(metadata), sort_keys=True)),
        )
        return output


class SimulationDataGenerator:
    def __init__(
        self,
        vehicle: VehicleParameters,
        actuator_limits: ActuatorLimits,
        curvature_weights: Sequence[float],
        acceleration_scale_mps2: float,
        time_step_s: float,
        normalizer: FeatureNormalizer,
        steering_scale_deg: float | None = None,
        preview_times_s: Sequence[float] = DEFAULT_PREVIEW_TIMES_S,
        lookahead_distances_m: Sequence[float] | None = None,
        expert_controller_config: Mapping[str, object] | None = None,
        seed: int = 42,
        steering_scale_rad: float | None = None,
    ) -> None:
        if time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive")
        if steering_scale_deg is None:
            if steering_scale_rad is None:
                raise ValueError("steering_scale_deg must be provided")
            steering_scale_deg = float(np.rad2deg(steering_scale_rad))
        if steering_scale_deg <= 0.0 or acceleration_scale_mps2 <= 0.0:
            raise ValueError("target scales must be positive")
        self.vehicle = vehicle
        self.actuator_limits = actuator_limits
        self.preview_times_s = tuple(float(value) for value in preview_times_s)
        self.fixed_lookahead_distances_m = (
            None
            if lookahead_distances_m is None
            else tuple(float(value) for value in lookahead_distances_m)
        )
        self.curvature_weights = tuple(float(value) for value in curvature_weights)
        self.steering_scale_deg = float(steering_scale_deg)
        self.acceleration_scale_mps2 = acceleration_scale_mps2
        self.time_step_s = time_step_s
        self.normalizer = normalizer
        self.expert_controller_config = dict(expert_controller_config or {})
        self.rng = np.random.default_rng(seed)
        self.feature_builder = FeatureBuilder()

    @staticmethod
    def _lateral_offset_candidates(
        lateral_offset_samples_m: Sequence[float] | None,
        maximum_lateral_offset_m: float | None,
    ) -> tuple[float, ...]:
        if lateral_offset_samples_m is None:
            return (0.0,)

        offsets = {0.0}
        for value in lateral_offset_samples_m:
            offset = float(value)
            if offset < 0.0:
                raise ValueError("lateral_offset_samples_m must contain non-negative values")
            if maximum_lateral_offset_m is not None and offset > maximum_lateral_offset_m:
                raise ValueError(
                    "lateral_offset_samples_m contains a value larger than "
                    f"maximum_lateral_offset_m={maximum_lateral_offset_m}"
                )
            offsets.add(offset)
            offsets.add(-offset)
        return tuple(sorted(offsets))

    @staticmethod
    def _scenario_id(
        profile_name: str,
        repetition: int,
        lateral_offset_m: float,
    ) -> str:
        if np.isclose(lateral_offset_m, 0.0):
            lateral_label = "lat_0"
        else:
            sign = "p" if lateral_offset_m > 0.0 else "n"
            magnitude = f"{abs(lateral_offset_m):g}".replace(".", "p")
            lateral_label = f"lat_{sign}{magnitude}"
        return f"{profile_name}_{repetition:03d}_{lateral_label}"

    def _initial_state(
        self,
        profile: ReferenceProfile,
        randomize: bool,
        lateral_offset_m: float | None = None,
    ) -> VehicleState:
        lateral_offset = (
            float(lateral_offset_m)
            if lateral_offset_m is not None
            else self.rng.uniform(-0.6, 0.6) if randomize else 0.0
        )
        yaw_offset = self.rng.uniform(-0.06, 0.06) if randomize else 0.0
        speed_offset = self.rng.uniform(-0.8, 0.8) if randomize else 0.0
        return initial_state_from_reference_profile(
            profile,
            lateral_offset_m=float(lateral_offset),
            yaw_offset_rad=float(yaw_offset),
            speed_offset_mps=float(speed_offset),
        )

    def generate(
        self,
        profiles: Sequence[ReferenceProfile],
        repetitions: int = 4,
        randomize_initial_state: bool = True,
        lateral_offset_samples_m: Sequence[float] | None = None,
        maximum_lateral_offset_m: float | None = None,
    ) -> GeneratedDataset:
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")

        all_features: list[np.ndarray] = []
        all_raw_features: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_physical_targets: list[np.ndarray] = []
        scenario_ids: list[str] = []
        timestamps: list[float] = []
        dynamics = KinematicBicycleModel(self.vehicle)
        lateral_offsets = self._lateral_offset_candidates(
            lateral_offset_samples_m,
            maximum_lateral_offset_m,
        )

        for profile in profiles:
            for repetition in range(repetitions):
                for lateral_offset_m in lateral_offsets:
                    expert = ExpertController(
                        self.vehicle,
                        self.actuator_limits,
                        **self.expert_controller_config,
                    )
                    state = self._initial_state(
                        profile,
                        randomize_initial_state,
                        lateral_offset_m=lateral_offset_m,
                    )
                    scenario_id = self._scenario_id(
                        profile.name,
                        repetition,
                        lateral_offset_m,
                    )
                    for time_s in profile.time_s[:-1]:
                        reference_s, reference_speed, reference_acceleration = profile.sample(
                            float(time_s)
                        )
                        errors = calculate_tracking_errors(
                            profile.points,
                            reference_speed,
                            reference_s,
                            state,
                        )
                        nearest_index = nearest_trajectory_index(profile.points, state)
                        path_start = min(nearest_index, len(profile.points) - 2)
                        body_points = global_to_body(profile.points[path_start:], state.pose)
                        lookahead_distances_m = self.fixed_lookahead_distances_m
                        if lookahead_distances_m is None:
                            lookahead_distances_m = preview_distances_from_times(
                                self.preview_times_s,
                                speed_mps=reference_speed,
                                acceleration_mps2=reference_acceleration,
                            )
                        sampled_points = sample_trajectory(
                            body_points,
                            lookahead_distances_m,
                        )
                        kappa = resolve_reference_curvature(
                            sampled_points,
                            supplied_kappa=None,
                            weights=self.curvature_weights,
                        )
                        features = self.feature_builder.build(
                            sampled_points,
                            kappa,
                            errors,
                            reference_acceleration,
                            reference_speed,
                            reference_s,
                            state,
                        ).values
                        expert_output = expert.compute(
                            profile.points,
                            state,
                            reference_s,
                            reference_speed,
                            reference_acceleration,
                            self.time_step_s,
                        )
                        physical_target = np.asarray(
                            [
                                expert_output.steering_des_deg,
                                expert_output.signed_accel_des_mps2,
                            ],
                            dtype=np.float32,
                        )
                        normalized_target = np.asarray(
                            [
                                physical_target[0] / self.steering_scale_deg,
                                physical_target[1] / self.acceleration_scale_mps2,
                            ],
                            dtype=np.float32,
                        )
                        all_raw_features.append(features)
                        all_features.append(self.normalizer.normalize(features))
                        all_targets.append(np.clip(normalized_target, -1.0, 1.0))
                        all_physical_targets.append(physical_target)
                        scenario_ids.append(scenario_id)
                        timestamps.append(float(time_s))
                        state = dynamics.step(
                            state,
                            expert_output.steering_des_rad,
                            expert_output.signed_accel_des_mps2,
                            self.time_step_s,
                        )

        return GeneratedDataset(
            features=np.asarray(all_features, dtype=np.float32),
            raw_features=np.asarray(all_raw_features, dtype=np.float32),
            targets=np.asarray(all_targets, dtype=np.float32),
            physical_targets=np.asarray(all_physical_targets, dtype=np.float32),
            scenario_ids=np.asarray(scenario_ids),
            timestamps_s=np.asarray(timestamps, dtype=np.float32),
        )


def generation_metadata(
    config: Mapping[str, Any],
    scenario_names: Sequence[str],
    sample_count: int,
) -> dict[str, object]:
    return {
        "generator": "preview_feedback_lateral_and_cascaded_pid_longitudinal",
        "scenario_names": list(scenario_names),
        "sample_count": sample_count,
        "target_format": "normalized_steering_deg_and_signed_acceleration",
        "config": dict(config),
    }
