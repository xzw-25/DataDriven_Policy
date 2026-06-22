from pathlib import Path
from xml.etree import ElementTree

import pytest


def assert_valid_drawio_pages(path: Path, expected_page_count: int) -> None:
    document = ElementTree.parse(path).getroot()
    diagrams = document.findall("diagram")
    assert len(diagrams) == expected_page_count
    for diagram in diagrams:
        cells = diagram.findall(".//mxCell")
        ids = {cell.get("id") for cell in cells}

        assert cells[0].get("id") == "0"
        assert cells[0].get("parent") is None
        assert cells[1].get("id") == "1"
        assert cells[1].get("parent") == "0"

        for cell in cells[2:]:
            parent = cell.get("parent")
            assert parent in ids
            if cell.get("edge") == "1":
                assert cell.get("source") in ids
                assert cell.get("target") in ids


def test_project_architecture_is_a_valid_single_page_drawio() -> None:
    assert_valid_drawio_pages(
        Path("docs/diagrams/vehicle_controller_project_architecture.drawio"),
        expected_page_count=1,
    )


def test_neural_network_architecture_is_a_valid_single_page_drawio() -> None:
    path = Path("docs/diagrams/vehicle_controller_neural_network_architecture.drawio")
    assert_valid_drawio_pages(path, expected_page_count=1)

    document = ElementTree.parse(path).getroot()
    graph = document.find("diagram/mxGraphModel")
    assert graph is not None
    assert graph.get("math") == "0"

    for cell in graph.findall(".//mxCell"):
        assert cell.get("id", "").isdigit()
        assert "html=1" not in cell.get("style", "")

    labels = "\n".join(cell.get("value", "") for cell in graph.findall(".//mxCell"))
    assert "三分支特征编码" in labels
    assert "128 - 64" in labels
    assert "128 - 128 - 64" in labels


@pytest.mark.parametrize(
    ("path", "expected_labels"),
    [
        (
            Path("docs/diagrams/mlp_controller_structure.drawio"),
            ("MLPController", "trajectory_encoder", "error_encoder", "state_encoder"),
        ),
        (
            Path("docs/diagrams/direct_mlp_controller_structure.drawio"),
            ("DirectMLPController", "Linear 21 -> 128", "Linear 64 -> 2"),
        ),
        (
            Path("docs/diagrams/gru_controller_structure.drawio"),
            ("GRUController", "GRU Cell t=1", "output[:, -1, :]"),
        ),
    ],
)
def test_controller_structure_drawio(
    path: Path,
    expected_labels: tuple[str, ...],
) -> None:
    assert_valid_drawio_pages(path, expected_page_count=1)

    graph = ElementTree.parse(path).getroot().find("diagram/mxGraphModel")
    assert graph is not None
    assert graph.get("math") == "0"

    labels = "\n".join(cell.get("value", "") for cell in graph.findall(".//mxCell"))
    for label in expected_labels:
        assert label in labels
