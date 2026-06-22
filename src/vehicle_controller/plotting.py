"""Shared plotting helpers."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any


def load_pyplot(show_plots: bool) -> Any:
    try:
        cache_dir = Path(tempfile.gettempdir()) / "vehicle_controller_matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        import matplotlib

        if not show_plots:
            matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise SystemExit(
            "Plotting requires matplotlib. Install dependencies with "
            "'python3 -m pip install -e .'."
        ) from error
    return plt
