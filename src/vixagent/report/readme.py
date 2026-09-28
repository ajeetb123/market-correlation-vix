"""Insert generated results into README marker blocks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vixagent.report.results import headline_sentence


def replace_between_markers(text: str, name: str, content: str) -> str:
    """Replace everything between <!-- {name}:START --> and <!-- {name}:END -->
    with '\\n' + content + '\\n'. Raises ValueError if either marker is missing
    or END precedes START."""
    start, end = f"<!-- {name}:START -->", f"<!-- {name}:END -->"
    i, j = text.find(start), text.find(end)
    if i < 0 or j < 0:
        raise ValueError(f"README is missing the {name} markers")
    if j < i:
        raise ValueError(f"README {name}:END precedes {name}:START")
    return text[: i + len(start)] + "\n" + content + "\n" + text[j:]


def _fmt(x: float | None, digits: int) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def results_block(results: dict[str, Any]) -> str:
    """Markdown for the RESULTS block, built only from results.json values."""
    rev = results["preregistered"]["reverse_test"]
    return "\n".join(
        [
            f"**Headline.** {headline_sentence(results)}",
            "",
            f"**Reverse direction** (VIX events followed by correlation spikes, test period): "
            f"{rev['n_events']} events, lift {_fmt(rev['lift'], 2)}, "
            f"p = {_fmt(rev['p_value'], 3)}.",
            "",
            f"**Out-of-sample R^2** of the correlation z-score regression: "
            f"{_fmt(results['oos']['r2_oos'], 4)}.",
            "",
            "Full tables: [reports/results.md](reports/results.md).",
        ]
    )


def update_readme(readme_path: Path, results: dict[str, Any], evals_summary: str | None) -> None:
    """RESULTS block: headline sentence, a one-line reverse-direction summary,
    the out-of-sample R squared, and a link to reports/results.md.
    EVALS block: evals_summary if provided, else 'Evals not yet run'. The EVALS
    block is optional: if the README has no EVALS markers it is left alone."""
    text = readme_path.read_text()
    text = replace_between_markers(text, "RESULTS", results_block(results))
    if "<!-- EVALS:START -->" in text:
        text = replace_between_markers(text, "EVALS", evals_summary or "Evals not yet run.")
    readme_path.write_text(text)
