from vehicle_controller.constants import FEATURE_COUNT, FEATURE_NAMES


def test_feature_contract_uses_lateral_then_speed_error() -> None:
    assert FEATURE_COUNT == 21
    assert FEATURE_NAMES[10:17] == (
        "kappa",
        "e_lat",
        "e_v",
        "e_s",
        "a_ref",
        "v_ref",
        "s_ref",
    )
    assert FEATURE_NAMES[17:] == ("vx", "ax", "ay", "r")
