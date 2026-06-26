from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import plot_training_npz_physical_summary as summary
from vehicle_controller.constants import FEATURE_COUNT


def test_training_npz_summary_uses_physical_controls_and_raw_features(monkeypatch, tmp_path):
    captured: dict[str, np.ndarray] = {}
    dataset_path = tmp_path / "dataset.npz"
    raw_features = np.arange(3 * FEATURE_COUNT, dtype=np.float32).reshape(3, FEATURE_COUNT)
    np.savez_compressed(
        dataset_path,
        features=np.zeros((3, FEATURE_COUNT), dtype=np.float32),
        raw_features=raw_features,
        targets=np.asarray([[0.1, 0.2], [0.9, 0.9], [-0.3, 0.4]], dtype=np.float32),
        physical_targets=np.asarray([[1.0, 0.2], [99.0, 99.0], [-3.0, 0.8]], dtype=np.float32),
        target_valid_mask=np.asarray([True, False, True]),
        clip_ids=np.asarray(["clip/a", "clip/a", "clip/b"]),
        timestamps_s=np.asarray([0.0, 0.1, 0.0], dtype=np.float32),
    )

    def fake_histogram(steering_deg, output_dir, **kwargs):
        captured["steering"] = np.asarray(steering_deg)
        return Path(output_dir) / "hist.png"

    def fake_feature_plot(features, feature_names, clip_ids, output_dir, **kwargs):
        captured["features"] = np.asarray(features)
        captured["clip_ids"] = np.asarray(clip_ids)
        captured["timestamps_s"] = np.asarray(kwargs["timestamps_s"])
        return Path(output_dir) / "features.png"

    def fake_stats(steering_deg, output_dir):
        captured["stats_steering"] = np.asarray(steering_deg)
        return Path(output_dir) / "stats.json"

    monkeypatch.setattr(summary, "save_steering_histogram", fake_histogram)
    monkeypatch.setattr(summary, "save_all_clip_physical_feature_plot", fake_feature_plot)
    monkeypatch.setattr(summary, "write_steering_stats", fake_stats)

    paths = summary.plot_training_npz_physical_summary(dataset_path, tmp_path / "out")

    assert [path.name for path in paths] == ["hist.png", "features.png", "stats.json"]
    np.testing.assert_allclose(captured["steering"], np.asarray([1.0, -3.0]))
    np.testing.assert_allclose(captured["stats_steering"], np.asarray([1.0, -3.0]))
    np.testing.assert_allclose(captured["features"], raw_features[[0, 2]])
    np.testing.assert_array_equal(captured["clip_ids"], np.asarray(["clip/a", "clip/b"]))
    np.testing.assert_allclose(captured["timestamps_s"], np.asarray([0.0, 0.0]))


def test_physical_controls_fall_back_to_denormalized_targets(tmp_path):
    dataset_path = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset_path,
        features=np.zeros((2, FEATURE_COUNT), dtype=np.float32),
        targets=np.asarray([[0.2, -0.5], [0.4, 0.25]], dtype=np.float32),
        metadata_json=np.asarray(
            '{"target_normalization": {"steering_scale_deg": 10.0, "accel_scale_mps2": 2.0}}'
        ),
    )

    with np.load(dataset_path, allow_pickle=False) as data:
        mask = np.asarray([True, True])
        controls = summary.physical_controls_from_npz(
            data,
            mask,
            "configs/model/mlp_controller.yaml",
        )

    np.testing.assert_allclose(controls, np.asarray([[2.0, -1.0], [4.0, 0.5]]))
