"""Dataset field contract."""

from vehicle_controller.constants import FEATURE_NAMES

TARGET_NAMES = ("steering_target_deg", "signed_accel_target_mps2")
REQUIRED_COLUMNS = (*FEATURE_NAMES, *TARGET_NAMES, "scenario_id")


def missing_columns(columns: set[str]) -> set[str]:
    return set(REQUIRED_COLUMNS) - columns
