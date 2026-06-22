from __future__ import annotations

from pathlib import Path

from scripts import train_imitation_from_raw_data as pipeline


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
