#!/usr/bin/env python3
"""Compare neural controller outputs with supervised NPZ targets."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from _bootstrap import PROJECT_ROOT
except ModuleNotFoundError:  # pragma: no cover - used when imported as scripts.*
    from scripts._bootstrap import PROJECT_ROOT

from vehicle_controller.training.supervised_validation import validate_supervised_dataset


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="artifacts/checkpoints/raw_data_imitation_smoke.pt",
        help="Trained controller checkpoint.",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/clean_ad_policy_sim_v1_aba9e399_imitation_dataset.npz",
        help="Supervised NPZ with features/targets/physical_targets.",
    )
    parser.add_argument("--model-config", help="Optional model config. Defaults to checkpoint config.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        default="artifacts/reports/supervised_validation",
        help="Directory for metrics, prediction arrays, and plots.",
    )
    parser.add_argument("--dataset-label", default="raw_data_imitation")
    parser.add_argument("--max-plot-scenarios", type=int, default=8)
    parser.add_argument("--max-plot-samples", type=int, default=2000)
    parser.add_argument("--show-plots", action="store_true")
    args = parser.parse_args()

    result = validate_supervised_dataset(
        project_path(args.checkpoint),
        project_path(args.dataset),
        project_path(args.output_dir),
        model_config_path=None if args.model_config is None else project_path(args.model_config),
        device=args.device,
        batch_size=args.batch_size,
        dataset_label=args.dataset_label,
        max_plot_scenarios=args.max_plot_scenarios,
        max_plot_samples=args.max_plot_samples,
        show_plots=args.show_plots,
    )
    for name, value in sorted(result.metrics.items()):
        print(f"{name}: {value:.6f}")
    print(f"metrics={result.metrics_path}")
    print(f"predictions={result.prediction_path}")
    for path in result.plot_paths:
        print(f"plot={path}")


if __name__ == "__main__":
    main()
