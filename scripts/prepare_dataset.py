#!/usr/bin/env python3
"""Convert a tabular controller dataset into the compact NPZ format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import PROJECT_ROOT  # noqa: F401
from vehicle_controller.constants import FEATURE_NAMES
from vehicle_controller.data.schema import TARGET_NAMES, missing_columns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv")
    parser.add_argument("output_npz")
    args = parser.parse_args()
    try:
        import pandas as pd
    except ImportError as error:
        raise SystemExit("Install pandas to convert CSV datasets") from error

    frame = pd.read_csv(args.input_csv)
    missing = missing_columns(set(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    features = frame.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float32)
    targets = frame.loc[:, TARGET_NAMES].to_numpy(dtype=np.float32)
    output = Path(args.output_npz)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=features, targets=targets)
    print(f"saved {len(frame)} samples to {output}")


if __name__ == "__main__":
    main()
