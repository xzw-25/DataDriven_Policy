from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from scripts import train_imitation_from_raw_data as pipeline
from scripts import train_imitation
from vehicle_controller.constants import FEATURE_COUNT


def test_train_imitation_from_raw_data_builds_dataset_then_trains(monkeypatch, tmp_path):
    calls: list[tuple[str, object]] = []
    dataset_path = tmp_path / "dataset.npz"
    checkpoint_path = tmp_path / "checkpoint.pt"

    def fake_build_features_from_raw_data(*args, **kwargs):
        calls.append(("build", args, kwargs))
        return dataset_path

    def fake_train_imitation(*, dataset, output, **kwargs):
        calls.append(("train", dataset, output, kwargs))
        return Path(output)

    monkeypatch.setattr(pipeline, "build_features_from_raw_data", fake_build_features_from_raw_data)
    monkeypatch.setattr(pipeline, "train_imitation", fake_train_imitation)

    result = pipeline.train_imitation_from_raw_data(
        raw_data=tmp_path / "raw.pkl",
        dataset_output=dataset_path,
        checkpoint_output=checkpoint_path,
        epochs=2,
        no_showcase=True,
        no_loss_plot=True,
    )

    assert result == checkpoint_path
    assert calls[0][0] == "build"
    assert calls[1][0] == "train"
    assert calls[1][1] == dataset_path
    assert calls[1][2] == checkpoint_path
    assert calls[1][3]["epochs"] == 2
    assert calls[1][3]["no_showcase"] is True


class FixedOutputModel(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        outputs = torch.asarray(
            [[0.1, 0.25], [-0.2, -0.5]],
            dtype=features.dtype,
            device=features.device,
        )
        return outputs[: features.shape[0]]


def test_training_output_plots_use_physical_denormalized_controls(monkeypatch, tmp_path):
    captured: dict[str, np.ndarray] = {}
    dataset_path = tmp_path / "dataset.npz"
    metadata = {
        "target_normalization": {
            "steering_scale_deg": 10.0,
            "accel_scale_mps2": 2.0,
        }
    }
    normalized_features = np.zeros((2, FEATURE_COUNT), dtype=np.float32)
    raw_features = np.arange(2 * FEATURE_COUNT, dtype=np.float32).reshape(2, FEATURE_COUNT)
    np.savez_compressed(
        dataset_path,
        features=normalized_features,
        raw_features=raw_features,
        targets=np.asarray([[0.2, -0.5], [0.4, 0.25]], dtype=np.float32),
        physical_targets=np.asarray([[2.0, -1.0], [4.0, 0.5]], dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata)),
    )

    def fake_control_plots(predicted_controls, expert_controls, *args, **kwargs):
        captured["predicted"] = np.asarray(predicted_controls)
        captured["expert"] = np.asarray(expert_controls)
        return ()

    def fake_feature_plots(features, *args, **kwargs):
        captured["features"] = np.asarray(features)
        return ()

    monkeypatch.setattr(
        train_imitation,
        "save_offline_control_comparison_plots",
        fake_control_plots,
    )
    monkeypatch.setattr(train_imitation, "save_feature_signal_plots", fake_feature_plots)

    train_imitation.save_split_output_comparison(
        split_name="validation",
        dataset_path=dataset_path,
        model=FixedOutputModel(),
        model_config={"steering_limit_deg": 10.0, "accel_limit_mps2": 2.0},
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        device="cpu",
        output_dir=tmp_path,
        max_plot_scenarios=None,
        max_plot_samples=10,
        show_plots=False,
    )

    np.testing.assert_allclose(
        captured["predicted"],
        np.asarray([[1.0, 0.5], [-2.0, -1.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(
        captured["expert"],
        np.asarray([[2.0, -1.0], [4.0, 0.5]], dtype=np.float64),
    )
    np.testing.assert_allclose(captured["features"], raw_features)
