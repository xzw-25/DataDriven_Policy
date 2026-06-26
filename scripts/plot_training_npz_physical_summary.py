#!/usr/bin/env python3
"""Plot physical-control and physical-feature summaries for a training NPZ."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts._bootstrap import PROJECT_ROOT

from vehicle_controller.constants import FEATURE_COUNT, FEATURE_NAMES
from vehicle_controller.plotting import load_pyplot
from vehicle_controller.training.offline_plots import _feature_axis_label
from vehicle_controller.units import steering_limit_deg_from_config
from vehicle_controller.utils.config import load_yaml


DEFAULT_DATASET = "data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz"
DEFAULT_OUTPUT_DIR = "artifacts/reports/training_npz_summary"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
    if "metadata_json" not in data:
        return {}
    return dict(json.loads(str(data["metadata_json"])))


def _target_scales(
    data: np.lib.npyio.NpzFile,
    model_config_path: str | Path,
) -> np.ndarray:
    metadata = _load_metadata(data)
    target_normalization = metadata.get("target_normalization")
    if isinstance(target_normalization, dict):
        steering_scale = target_normalization.get("steering_scale_deg")
        accel_scale = target_normalization.get("accel_scale_mps2")
        if steering_scale is not None and accel_scale is not None:
            return np.asarray([float(steering_scale), float(accel_scale)], dtype=np.float64)

    model_config = load_yaml(project_path(model_config_path))
    return np.asarray(
        [
            steering_limit_deg_from_config(model_config),
            float(model_config["accel_limit_mps2"]),
        ],
        dtype=np.float64,
    )


def _valid_mask(data: np.lib.npyio.NpzFile, *, include_invalid_targets: bool) -> np.ndarray:
    count = int(data["features"].shape[0])
    if include_invalid_targets or "target_valid_mask" not in data:
        return np.ones(count, dtype=bool)
    mask = np.asarray(data["target_valid_mask"], dtype=bool)
    if mask.shape != (count,):
        raise ValueError("target_valid_mask must have shape [N]")
    return mask


def physical_controls_from_npz(
    data: np.lib.npyio.NpzFile,
    mask: np.ndarray,
    model_config_path: str | Path,
) -> np.ndarray:
    if "physical_targets" in data:
        controls = np.asarray(data["physical_targets"], dtype=np.float64)
    else:
        controls = np.asarray(data["targets"], dtype=np.float64) * _target_scales(
            data,
            model_config_path,
        )
    if controls.shape != (len(mask), 2):
        raise ValueError("targets/physical_targets must have shape [N, 2]")
    return controls[mask]


def physical_features_from_npz(
    data: np.lib.npyio.NpzFile,
    mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    feature_key = "raw_features" if "raw_features" in data else "features"
    features = np.asarray(data[feature_key], dtype=np.float64)
    if features.shape != (len(mask), FEATURE_COUNT):
        raise ValueError(f"{feature_key} must have shape [N, {FEATURE_COUNT}]")
    return features[mask], feature_key


def feature_names_from_npz(data: np.lib.npyio.NpzFile) -> tuple[str, ...]:
    if "feature_names" not in data:
        return FEATURE_NAMES
    feature_names = tuple(str(value) for value in data["feature_names"])
    if len(feature_names) != FEATURE_COUNT:
        raise ValueError(f"feature_names must contain {FEATURE_COUNT} values")
    return feature_names


def optional_frame_aligned_array(
    data: np.lib.npyio.NpzFile,
    key: str,
    mask: np.ndarray,
) -> np.ndarray | None:
    if key not in data:
        return None
    values = np.asarray(data[key])
    if values.shape[0] != len(mask):
        raise ValueError(f"{key} must be frame-aligned with features")
    return values[mask]


def clip_ids_from_npz(data: np.lib.npyio.NpzFile, mask: np.ndarray) -> np.ndarray:
    for key in ("clip_ids", "scenario_ids"):
        values = optional_frame_aligned_array(data, key, mask)
        if values is not None:
            return np.asarray(values).astype(str)
    return np.asarray(["all_samples"] * int(mask.sum()), dtype=str)


def timestamps_from_npz(data: np.lib.npyio.NpzFile, mask: np.ndarray) -> np.ndarray | None:
    values = optional_frame_aligned_array(data, "timestamps_s", mask)
    if values is None:
        return None
    return np.asarray(values, dtype=np.float64)


def _ordered_unique(values: np.ndarray) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values))


def _downsample_indices(indices: np.ndarray, maximum_samples: int) -> np.ndarray:
    if maximum_samples <= 0 or len(indices) <= maximum_samples:
        return indices
    selected = np.unique(np.linspace(0, len(indices) - 1, maximum_samples).astype(np.int64))
    return indices[selected]


def save_steering_histogram(
    steering_deg: np.ndarray,
    output_dir: str | Path,
    *,
    bins: int,
    show_plots: bool = False,
) -> Path:
    values = np.asarray(steering_deg, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("steering_deg must be a non-empty 1D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("steering_deg contains non-finite values")

    plt = load_pyplot(show_plots)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(values, bins=bins, color="#2563eb", edgecolor="white", alpha=0.88)
    axis.axvline(
        float(np.mean(values)),
        color="#dc2626",
        linestyle="--",
        linewidth=1.6,
        label="Mean",
    )
    axis.axvline(
        float(np.median(values)),
        color="#16a34a",
        linestyle=":",
        linewidth=1.8,
        label="Median",
    )
    axis.set_title("Steering Command Distribution")
    axis.set_xlabel("Steering wheel angle command [deg]")
    axis.set_ylabel("Sample count")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()

    path = output_path / "steering_command_histogram.png"
    figure.savefig(path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(figure)
    return path


def save_all_clip_physical_feature_plot(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    clip_ids: np.ndarray,
    output_dir: str | Path,
    *,
    timestamps_s: np.ndarray | None = None,
    max_samples_per_clip: int = 1200,
    show_plots: bool = False,
) -> Path:
    feature_array = np.asarray(features, dtype=np.float64)
    if feature_array.shape[1:] != (FEATURE_COUNT,):
        raise ValueError(f"features must have shape [N, {FEATURE_COUNT}]")
    if clip_ids.shape != (len(feature_array),):
        raise ValueError("clip_ids must have shape [N]")
    if timestamps_s is not None and timestamps_s.shape != (len(feature_array),):
        raise ValueError("timestamps_s must have shape [N]")

    plt = load_pyplot(show_plots)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clip_names = _ordered_unique(clip_ids)
    colormap = plt.get_cmap("tab20")
    figure, axes = plt.subplots(7, 3, figsize=(21, 28), sharex=False)
    flat_axes = axes.ravel()
    for feature_index, axis in enumerate(flat_axes):
        feature_name = feature_names[feature_index]
        for clip_order, clip_name in enumerate(clip_names):
            indices = np.flatnonzero(clip_ids == clip_name)
            if len(indices) == 0:
                continue
            if timestamps_s is not None:
                indices = indices[np.argsort(timestamps_s[indices], kind="stable")]
                x_values = timestamps_s[indices] - timestamps_s[indices][0]
                x_label = "Relative time [s]"
            else:
                x_values = np.arange(len(indices), dtype=np.float64)
                x_label = "Frame index within clip"
            selected = _downsample_indices(np.arange(len(indices)), max_samples_per_clip)
            plotted_indices = indices[selected]
            x_plot = x_values[selected]
            axis.plot(
                x_plot,
                feature_array[plotted_indices, feature_index],
                color=colormap(clip_order % 20),
                linewidth=0.85,
                alpha=0.42,
            )
        axis.set_title(feature_name, fontsize=10)
        axis.set_ylabel(_feature_axis_label(feature_name))
        axis.grid(True, alpha=0.22)
        if feature_index >= FEATURE_COUNT - 3:
            axis.set_xlabel(x_label)

    figure.suptitle(
        f"Physical Feature Signals Across All Clips ({len(clip_names)} clips)",
        fontsize=16,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

    path = output_path / "physical_features_all_clips.png"
    figure.savefig(path, dpi=180)
    if show_plots:
        plt.show()
    plt.close(figure)
    return path


def write_steering_stats(steering_deg: np.ndarray, output_dir: str | Path) -> Path:
    values = np.asarray(steering_deg, dtype=np.float64)
    stats = {
        "sample_count": int(len(values)),
        "min_deg": float(np.min(values)),
        "max_deg": float(np.max(values)),
        "mean_deg": float(np.mean(values)),
        "median_deg": float(np.median(values)),
        "std_deg": float(np.std(values)),
        "p01_deg": float(np.percentile(values, 1.0)),
        "p05_deg": float(np.percentile(values, 5.0)),
        "p95_deg": float(np.percentile(values, 95.0)),
        "p99_deg": float(np.percentile(values, 99.0)),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / "steering_command_stats.json"
    path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return path


def plot_training_npz_physical_summary(
    dataset_npz: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    model_config: str | Path = "configs/model/mlp_controller.yaml",
    bins: int = 80,
    max_samples_per_clip: int = 1200,
    include_invalid_targets: bool = False,
    show_plots: bool = False,
) -> tuple[Path, Path, Path]:
    dataset_path = project_path(dataset_npz)
    output_path = project_path(output_dir)
    with np.load(dataset_path, allow_pickle=False) as data:
        mask = _valid_mask(data, include_invalid_targets=include_invalid_targets)
        controls = physical_controls_from_npz(data, mask, model_config)
        features, feature_source = physical_features_from_npz(data, mask)
        feature_names = feature_names_from_npz(data)
        clip_ids = clip_ids_from_npz(data, mask)
        timestamps_s = timestamps_from_npz(data, mask)

    histogram_path = save_steering_histogram(
        controls[:, 0],
        output_path,
        bins=bins,
        show_plots=show_plots,
    )
    feature_plot_path = save_all_clip_physical_feature_plot(
        features,
        feature_names,
        clip_ids,
        output_path,
        timestamps_s=timestamps_s,
        max_samples_per_clip=max_samples_per_clip,
        show_plots=show_plots,
    )
    stats_path = write_steering_stats(controls[:, 0], output_path)
    print(f"dataset={dataset_path}")
    print(f"sample_count={len(controls)}")
    print(f"clip_count={len(_ordered_unique(clip_ids))}")
    print(f"feature_source={feature_source}")
    print(f"steering_histogram={histogram_path}")
    print(f"physical_feature_plot={feature_plot_path}")
    print(f"steering_stats={stats_path}")
    return histogram_path, feature_plot_path, stats_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_npz",
        nargs="?",
        default=DEFAULT_DATASET,
        help="Full training NPZ containing features/raw_features and targets/physical_targets.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-config",
        default="configs/model/mlp_controller.yaml",
        help="Used only when physical_targets are absent and targets need denormalization.",
    )
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument(
        "--max-samples-per-clip",
        type=int,
        default=1200,
        help="Downsample each clip in the all-clip feature figure. Use 0 to disable.",
    )
    parser.add_argument(
        "--include-invalid-targets",
        action="store_true",
        help="Do not filter target_valid_mask; by default only training-valid samples are plotted.",
    )
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()
    plot_training_npz_physical_summary(
        args.dataset_npz,
        args.output_dir,
        model_config=args.model_config,
        bins=args.bins,
        max_samples_per_clip=args.max_samples_per_clip,
        include_invalid_targets=args.include_invalid_targets,
        show_plots=args.show_plots,
    )


if __name__ == "__main__":
    main()
