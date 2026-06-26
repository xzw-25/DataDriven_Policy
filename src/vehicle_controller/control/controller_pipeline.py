"""End-to-end online controller pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from vehicle_controller.control.command_limiter import CommandLimiter
from vehicle_controller.control.fallback_controller import FallbackController
from vehicle_controller.control.longitudinal_allocator import LongitudinalAllocator
from vehicle_controller.control.neural_policy import NeuralPolicy
from vehicle_controller.control.safety_supervisor import SafetySupervisor
from vehicle_controller.features.error_calculator import (
    calculate_tracking_errors,
    nearest_trajectory_index,
)
from vehicle_controller.features.feature_builder import FeatureBuilder
from vehicle_controller.features.validator import FeatureValidator
from vehicle_controller.geometry.coordinate_transform import global_to_body
from vehicle_controller.geometry.curvature import resolve_reference_curvature
from vehicle_controller.geometry.trajectory_sampler import (
    DEFAULT_PREVIEW_TIMES_S,
    preview_distances_from_times,
    sample_trajectory,
)
from vehicle_controller.types import (
    CommandSource,
    ControllerStepDiagnostics,
    NeuralPolicyOutput,
    ReferenceTrajectory,
    TrackingErrors,
    VehicleCommand,
    VehicleState,
)


class ControllerPipeline:
    def __init__(
        self,
        neural_policy: NeuralPolicy,
        allocator: LongitudinalAllocator,
        limiter: CommandLimiter,
        fallback_controller: FallbackController,
        safety_supervisor: SafetySupervisor,
        feature_validator: FeatureValidator | None = None,
        preview_times_s: Sequence[float] = DEFAULT_PREVIEW_TIMES_S,
        lookahead_distances_m: Sequence[float] | None = None,
        curvature_weights: Sequence[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
    ) -> None:
        self.neural_policy = neural_policy
        self.allocator = allocator
        self.limiter = limiter
        self.fallback_controller = fallback_controller
        self.safety_supervisor = safety_supervisor
        self.feature_validator = feature_validator or FeatureValidator()
        self.preview_times_s = tuple(float(value) for value in preview_times_s)
        self.fixed_lookahead_distances_m = (
            None
            if lookahead_distances_m is None
            else tuple(float(value) for value in lookahead_distances_m)
        )
        self.curvature_weights = curvature_weights
        self.feature_builder = FeatureBuilder()
        self.previous_command = VehicleCommand()
        self.last_diagnostics: ControllerStepDiagnostics | None = None

    def reset(self) -> None:
        self.previous_command = VehicleCommand()
        self.last_diagnostics = None

    def _to_command(
        self,
        output: NeuralPolicyOutput,
        state: VehicleState,
        source: CommandSource,
        reason: str,
    ) -> VehicleCommand:
        longitudinal = self.allocator.allocate(output.signed_accel_des_mps2, state.vx)
        return VehicleCommand(
            steering_wheel_angle_deg=output.steering_des_deg,
            drive_wheel_torque_nm=longitudinal.drive_wheel_torque_nm,
            drive_valid=longitudinal.drive_valid,
            brake_decel_mps2=longitudinal.brake_decel_mps2,
            brake_valid=longitudinal.brake_valid,
            source=source,
            reason=reason,
        )

    def step(
        self,
        reference: ReferenceTrajectory,
        state: VehicleState,
        dt: float,
        tracking_errors: TrackingErrors | None = None,
    ) -> VehicleCommand:
        errors = tracking_errors or calculate_tracking_errors(
            reference.points,
            reference.v_ref,
            reference.s_ref,
            state,
        )
        self.last_diagnostics = None
        nearest_index = nearest_trajectory_index(reference.points, state)
        path_start = min(nearest_index, len(reference.points) - 2)
        body_points = global_to_body(reference.points[path_start:], state.pose)
        lookahead_distances_m = self.fixed_lookahead_distances_m
        if lookahead_distances_m is None:
            lookahead_distances_m = preview_distances_from_times(
                self.preview_times_s,
                speed_mps=reference.v_ref,
                acceleration_mps2=reference.a_ref,
            )
        sampled_points = sample_trajectory(body_points, lookahead_distances_m)
        kappa = resolve_reference_curvature(
            sampled_points,
            reference.kappa,
            self.curvature_weights,
        )
        features = self.feature_builder.build(
            sampled_points,
            kappa,
            errors,
            reference.a_ref,
            reference.v_ref,
            reference.s_ref,
            state,
        )
        fallback_output = self.fallback_controller.compute(
            sampled_points,
            errors,
            reference.a_ref,
            state,
        )
        fallback_candidate = self._to_command(
            fallback_output,
            state,
            CommandSource.FALLBACK,
            "fallback",
        )
        fallback_command = self.limiter.limit(fallback_candidate, self.previous_command, dt)
        fallback_command = VehicleCommand(
            steering_wheel_angle_deg=fallback_command.steering_wheel_angle_deg,
            drive_wheel_torque_nm=fallback_command.drive_wheel_torque_nm,
            drive_valid=fallback_command.drive_valid,
            brake_decel_mps2=fallback_command.brake_decel_mps2,
            brake_valid=fallback_command.brake_valid,
            source=CommandSource.FALLBACK,
            reason="fallback",
        )

        validation = self.feature_validator.validate_raw(features.values)
        if not validation.valid:
            self.previous_command = fallback_command
            command = VehicleCommand(
                steering_wheel_angle_deg=fallback_command.steering_wheel_angle_deg,
                drive_wheel_torque_nm=fallback_command.drive_wheel_torque_nm,
                drive_valid=fallback_command.drive_valid,
                brake_decel_mps2=fallback_command.brake_decel_mps2,
                brake_valid=fallback_command.brake_valid,
                source=CommandSource.FALLBACK,
                reason=validation.reason,
            )
            self.last_diagnostics = ControllerStepDiagnostics(
                tracking_errors=errors,
                neural_output=None,
                neural_candidate=None,
                limited_candidate=None,
                final_command=command,
            )
            return command

        neural_output = self.neural_policy.predict(features)
        candidate = self._to_command(
            neural_output,
            state,
            CommandSource.NEURAL,
            "ok",
        )
        limited_candidate = self.limiter.limit(candidate, self.previous_command, dt)
        decision = self.safety_supervisor.evaluate(
            limited_candidate,
            fallback_command,
            state,
            errors,
        )
        command = decision.command
        if decision.action.value == "fallback":
            command = VehicleCommand(
                steering_wheel_angle_deg=command.steering_wheel_angle_deg,
                drive_wheel_torque_nm=command.drive_wheel_torque_nm,
                drive_valid=command.drive_valid,
                brake_decel_mps2=command.brake_decel_mps2,
                brake_valid=command.brake_valid,
                source=CommandSource.FALLBACK,
                reason=decision.reason,
            )
        self.previous_command = command
        self.last_diagnostics = ControllerStepDiagnostics(
            tracking_errors=errors,
            neural_output=neural_output,
            neural_candidate=candidate,
            limited_candidate=limited_candidate,
            final_command=command,
        )
        return command
