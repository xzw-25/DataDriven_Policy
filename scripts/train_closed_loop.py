#!/usr/bin/env python3
"""Entry point reserved for differentiable closed-loop training."""

from _bootstrap import PROJECT_ROOT  # noqa: F401
from vehicle_controller.training.closed_loop_trainer import tracking_rollout_loss


def main() -> None:
    print(
        "Closed-loop loss is available as "
        f"{tracking_rollout_loss.__module__}.{tracking_rollout_loss.__name__}; "
        "connect the target differentiable vehicle model before production training."
    )


if __name__ == "__main__":
    main()
