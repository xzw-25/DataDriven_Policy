import csv

import pytest

from vehicle_controller.training.loss_plots import (
    make_loss_history,
    save_loss_curve,
    save_loss_history_csv,
)


def test_make_loss_history_numbers_optimizer_iterations() -> None:
    history = make_loss_history(
        batch_losses_by_epoch=((0.4, 0.3), (0.2,)),
        epoch_losses=(0.35, 0.2),
    )

    assert history.iterations == (1, 2, 3)
    assert history.batch_losses == (0.4, 0.3, 0.2)
    assert history.epoch_iterations == (2, 3)
    assert history.epoch_losses == (0.35, 0.2)


def test_loss_history_csv_and_curve_are_written(tmp_path) -> None:
    history = make_loss_history(
        batch_losses_by_epoch=((0.4, 0.3), (0.2,)),
        epoch_losses=(0.35, 0.2),
    )

    csv_path = save_loss_history_csv(history, tmp_path / "loss_history.csv")
    curve_path = save_loss_curve(history, tmp_path / "loss_curve.png")

    assert csv_path.is_file()
    assert curve_path.is_file()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["iteration", "batch_loss", "epoch_mean_loss"]
    assert rows[2] == ["2", "0.3", "0.35"]


def test_loss_history_csv_includes_validation_loss_when_available(tmp_path) -> None:
    history = make_loss_history(
        batch_losses_by_epoch=((0.4, 0.3), (0.2,)),
        epoch_losses=(0.35, 0.2),
        validation_epoch_losses=(0.5, 0.25),
    )

    csv_path = save_loss_history_csv(history, tmp_path / "loss_history.csv")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == [
        "iteration",
        "batch_loss",
        "epoch_mean_loss",
        "validation_epoch_loss",
    ]
    assert rows[2] == ["2", "0.3", "0.35", "0.5"]
    assert rows[3] == ["3", "0.2", "0.2", "0.25"]


def test_loss_curve_uses_log_y_axis(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    class FakeAxis:
        def plot(self, *args, **kwargs):
            return None

        def set_title(self, value):
            return None

        def set_xlabel(self, value):
            return None

        def set_ylabel(self, value):
            return None

        def set_yscale(self, value):
            calls["yscale"] = value

        def grid(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

    class FakeFigure:
        def tight_layout(self):
            return None

        def savefig(self, path, dpi):
            path.write_bytes(b"fake png")

    class FakePyplot:
        def subplots(self, *args, **kwargs):
            return FakeFigure(), FakeAxis()

        def show(self):
            return None

        def close(self, figure):
            return None

    monkeypatch.setattr(
        "vehicle_controller.training.loss_plots.load_pyplot",
        lambda show_plots: FakePyplot(),
    )
    history = make_loss_history(
        batch_losses_by_epoch=((0.0, 0.3),),
        epoch_losses=(0.15,),
    )

    curve_path = save_loss_curve(history, tmp_path / "loss_curve.png")

    assert curve_path.is_file()
    assert calls["yscale"] == "log"


def test_loss_curve_rejects_non_finite_loss(tmp_path) -> None:
    history = make_loss_history(
        batch_losses_by_epoch=((float("nan"),),),
        epoch_losses=(0.1,),
    )

    with pytest.raises(ValueError, match="non-finite"):
        save_loss_curve(history, tmp_path / "loss_curve.png")
