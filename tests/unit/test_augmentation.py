import numpy as np

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.data.augmentation import mirror_left_right


def test_mirror_reverses_lateral_signs_and_steering() -> None:
    features = np.ones((1, FEATURE_COUNT), dtype=np.float32)
    targets = np.ones((1, 2), dtype=np.float32)
    mirrored_features, mirrored_targets = mirror_left_right(features, targets)
    assert mirrored_features[0, 0] == 1.0
    assert mirrored_features[0, 1] == -1.0
    assert mirrored_features[0, 10] == -1.0
    assert mirrored_features[0, 11] == -1.0
    assert mirrored_features[0, 12] == 1.0
    assert mirrored_features[0, 15] == 1.0
    assert mirrored_targets[0, 0] == -1.0
    assert mirrored_targets[0, 1] == 1.0
