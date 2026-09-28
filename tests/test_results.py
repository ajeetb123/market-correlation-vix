"""Tests for results assembly, rendering, figures, README injection, and the report."""

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.fixtures.synthetic import make_price_panel, tiny_settings
from vixagent.analysis.frames import FrameStore
from vixagent.config import Preregistered, Settings, load_preregistered, load_settings
from vixagent.report import generate_report
from vixagent.report.figures import (
    plot_corr_heatmaps,
    plot_event_study,
    plot_grid_lift,
    plot_timeseries,
)
from vixagent.report.readme import replace_between_markers, update_readme
from vixagent.report.results import build_results, headline_sentence, render_results_md

REQUIRED_KEYS = {
    "dataset",
    "preregistered",
    "regression",
    "oos",
    "overfitting_check",
    "grid",
    "data_snapshot_end",
    "generated_at",
}
README = (
    "# Title\n\nIntro.\n\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n\n"
    "<!-- EVALS:START -->\n<!-- EVALS:END -->\n\nOutro.\n"
)


@pytest.fixture(scope="module")
def ctx(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    settings = tiny_settings(tmp_path_factory.mktemp("settings"))
    prereg = load_preregistered(load_settings())
    panel = make_price_panel(seed=0, spike_positions=[300, 700, 1100])
    store = FrameStore(panel, settings)
    return {
        "settings": settings,
        "prereg": prereg,
        "panel": panel,
        "store": store,
        "results": build_results(store, settings, prereg),
    }


def _strip(results: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(results)
    out.pop("generated_at")
    return out


def test_results_are_strict_json(ctx: dict[str, Any]) -> None:
    json.dumps(ctx["results"], allow_nan=False)


def test_results_have_required_keys(ctx: dict[str, Any]) -> None:
    assert set(ctx["results"]) == REQUIRED_KEYS
    assert set(ctx["results"]["preregistered"]) == {"spec", "train", "test", "reverse_test"}
    assert len(ctx["results"]["grid"]) == 24


def test_results_are_deterministic(ctx: dict[str, Any]) -> None:
    again = build_results(ctx["store"], ctx["settings"], ctx["prereg"])
    assert _strip(again) == _strip(ctx["results"])


def _hand_results(n_events: int, p_value: float | None) -> dict[str, Any]:
    return {
        "dataset": {"periods": {"test": ["2019-01-01", "2026-06-30"]}},
        "preregistered": {
            "spec": {
                "group": "risk",
                "window": 21,
                "z_threshold": 2.0,
                "horizon": 10,
                "clean_only": True,
                "alpha": 0.05,
            },
            "test": {
                "n_events": n_events,
                "hit_rate": 0.25,
                "base_rate": 0.1,
                "lift": 2.5,
                "p_value": p_value,
            },
        },
    }


@pytest.mark.parametrize(
    ("n_events", "p_value", "verdict"),
    [
        (3, 0.01, "Too few events for a reliable test."),
        (12, 0.01, "Significant at alpha = 0.05."),
        (12, 0.4, "Not significant at alpha = 0.05: no evidence of a lead effect."),
        (12, None, "Not significant at alpha = 0.05: no evidence of a lead effect."),
    ],
)
def test_headline_verdicts(n_events: int, p_value: float | None, verdict: str) -> None:
    sentence = headline_sentence(_hand_results(n_events, p_value))
    assert sentence.endswith(verdict)
    assert "2019-01-01 to 2026-06-30" in sentence
    assert "hit rate of 25.0%" in sentence


def test_render_results_md(ctx: dict[str, Any]) -> None:
    results = copy.deepcopy(ctx["results"])
    results["oos"]["r2_oos"] = None
    md = render_results_md(results)
    for heading in (
        "## Dataset",
        "## Headline",
        "## Reverse Direction Check",
        "## Regression",
        "## Out-of-Sample",
        "## Overfitting Check",
        "## Full Grid (exploratory)",
        "## Limitations",
    ):
        assert heading in md
    assert "R^2_oos = n/a" in md


def test_figures_are_written(ctx: dict[str, Any], tmp_path: Path) -> None:
    settings: Settings = ctx["settings"]
    prereg: Preregistered = ctx["prereg"]
    store: FrameStore = ctx["store"]
    paths = [
        plot_timeseries(store.frame("risk", 21), settings, tmp_path / "timeseries.png"),
        plot_corr_heatmaps(store.aligned("all")[0], settings, tmp_path / "corr_heatmaps.png"),
        plot_event_study(ctx["results"], store, settings, prereg, tmp_path / "event_study.png"),
        plot_grid_lift(ctx["results"], tmp_path / "grid_lift.png"),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_replace_between_markers() -> None:
    out = replace_between_markers(README, "RESULTS", "new")
    assert "<!-- RESULTS:START -->\nnew\n<!-- RESULTS:END -->" in out
    assert "old" not in out
    assert out.startswith("# Title\n\nIntro.") and out.endswith("Outro.\n")
    assert replace_between_markers(out, "RESULTS", "new") == out


def test_replace_between_markers_errors() -> None:
    with pytest.raises(ValueError):
        replace_between_markers("no markers", "RESULTS", "x")
    with pytest.raises(ValueError):
        replace_between_markers("<!-- X:END --> <!-- X:START -->", "X", "x")


def test_update_readme_idempotent(ctx: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text(README)
    update_readme(path, ctx["results"], None)
    first = path.read_text()
    assert "Evals not yet run." in first
    assert headline_sentence(ctx["results"]) in first
    update_readme(path, ctx["results"], None)
    assert path.read_text() == first


def test_generate_report(ctx: dict[str, Any], tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(README)
    args = (ctx["settings"], ctx["prereg"], ctx["panel"], tmp_path)
    generate_report(*args)
    reports = tmp_path / "reports"
    outputs = [
        reports / "results.json",
        reports / "results.md",
        *(
            reports / "figures" / f
            for f in ("timeseries.png", "corr_heatmaps.png", "event_study.png", "grid_lift.png")
        ),
    ]
    assert all(p.exists() for p in outputs)
    first = json.loads((reports / "results.json").read_text())
    assert "old" not in (tmp_path / "README.md").read_text()
    generate_report(*args)
    second = json.loads((reports / "results.json").read_text())
    assert _strip(first) == _strip(second)


def test_report_cli_has_command() -> None:
    from typer.testing import CliRunner

    from vixagent import cli

    result = CliRunner().invoke(cli.app, ["--help"])
    assert "report" in result.stdout
    assert isinstance(pd.__version__, str)


def test_update_readme_without_evals_block(ctx: dict[str, Any], tmp_path: Path) -> None:
    path = tmp_path / "README.md"
    path.write_text("# T\n<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n")
    update_readme(path, ctx["results"], None)
    text = path.read_text()
    assert headline_sentence(ctx["results"]) in text
    assert "Evals not yet run" not in text
