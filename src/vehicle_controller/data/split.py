"""Scenario-level dataset splitting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def split_scenarios(
    scenario_ids: Sequence[str],
    train_ratio: float,
    validation_ratio: float,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    unique = np.asarray(sorted(set(scenario_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    train_end = int(len(unique) * train_ratio)
    validation_end = train_end + int(len(unique) * validation_ratio)
    return (
        set(unique[:train_end]),
        set(unique[train_end:validation_end]),
        set(unique[validation_end:]),
    )

