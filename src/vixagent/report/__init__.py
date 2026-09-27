"""Report generation: results files, figures, and README injection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from vixagent.analysis.frames import FrameStore
from vixagent.config import Preregistered, Settings
from vixagent.report.figures import (
    plot_corr_heatmaps,
    plot_event_study,
    plot_grid_lift,
    plot_timeseries,
)
from vixagent.report.readme import update_readme
from vixagent.report.results import build_results, render_results_md


def generate_report(
    settings: Settings, prereg: Preregistered, prices: pd.DataFrame, root: Path
) -> dict[str, Any]:
    """Build results, write reports/results.json, reports/results.md and the
    four figures under `root`, and update root/README.md markers.

    The EVALS block is filled from evals/results/latest.md when it exists.
    """
    store = FrameStore(prices, settings)
    results = build_results(store, settings, prereg)
    reports = root / "reports"
    figures = reports / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    (reports / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    (reports / "results.md").write_text(render_results_md(results))
    plot_timeseries(store.frame("risk", 21), settings, figures / "timeseries.png")
    plot_corr_heatmaps(store.aligned("all")[0], settings, figures / "corr_heatmaps.png")
    plot_event_study(results, store, settings, prereg, figures / "event_study.png")
    plot_grid_lift(results, figures / "grid_lift.png")
    evals = root / "evals" / "results" / "latest.md"
    update_readme(root / "README.md", results, evals.read_text() if evals.exists() else None)
    return results
