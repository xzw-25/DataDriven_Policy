"""Stable feature and interface constants."""

FEATURE_NAMES: tuple[str, ...] = (
    "x1",
    "y1",
    "x2",
    "y2",
    "x3",
    "y3",
    "x4",
    "y4",
    "x5",
    "y5",
    "kappa",
    "e_lat",
    "e_v",
    "e_s",
    "a_ref",
    "v_ref",
    "s_ref",
    "vx",
    "ax",
    "ay",
    "r",
)
FEATURE_COUNT = len(FEATURE_NAMES)
TRAJECTORY_FEATURE_COUNT = 10
REFERENCE_ERROR_FEATURE_COUNT = 7
STATE_FEATURE_COUNT = 4
