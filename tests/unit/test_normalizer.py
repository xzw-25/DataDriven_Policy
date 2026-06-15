import numpy as np

from vehicle_controller.constants import FEATURE_COUNT
from vehicle_controller.features.normalizer import FeatureNormalizer


def test_normalizer_round_trip_without_clipping() -> None:
    normalizer = FeatureNormalizer(
        np.zeros(FEATURE_COUNT),
        np.full(FEATURE_COUNT, 2.0),
        clip=5.0,
    )
    values = np.linspace(-2.0, 2.0, FEATURE_COUNT, dtype=np.float32)
    assert np.allclose(normalizer.denormalize(normalizer.normalize(values)), values)
