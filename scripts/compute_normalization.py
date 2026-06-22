#!/usr/bin/env python3
"""Compute training-set feature normalization statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from vehicle_controller.constants import FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_npz")
    parser.add_argument("output_json")
    args = parser.parse_args()
    dataset = np.load(args.dataset_npz)
    feature_key = "raw_features" if "raw_features" in dataset else "features"
    features = dataset[feature_key].astype(np.float64)
    output = {
        "version": "features_v003",
        "feature_names": FEATURE_NAMES,
        "mean": features.mean(axis=0).tolist(),
        "std": np.maximum(features.std(axis=0), 1e-6).tolist(),
        "clip": 5.0,
    }
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
