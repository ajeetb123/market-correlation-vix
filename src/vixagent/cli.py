"""Command-line interface for vixagent."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import anthropic
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from vixagent import __version__
from vixagent.agent.client import MissingAPIKeyError, agent_model, judge_model, make_client
from vixagent.agent.loop import AgentResult, run_agent
from vixagent.agent.service import ResearchService
from vixagent.agent.transcript import TranscriptWriter
from vixagent.config import find_project_root, load_preregistered, load_settings
from vixagent.data.align import align_group
from vixagent.data.cache import load_or_fetch
from vixagent.data.fetch import DataError
from vixagent.data.validate import validate_prices
from vixagent.evals.cases import load_cases
from vixagent.evals.judge import calibrate
from vixagent.evals.runner import run_suite
from vixagent.report import generate_report
from vixagent.report.results import headline_sentence

app = typer.Typer(help="VIX research agent CLI.", no_args_is_help=True)
console = Console()


@app.callback()
def main() -> None:
    """VIX research agent CLI."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def pull(refresh: bool = typer.Option(False, help="Ignore cache and redownload.")) -> None:
    """Download (or load cached) prices, validate, and print a summary."""
    load_dotenv()
    settings = load_settings()
    try:
        prices = load_or_fetch(settings, refresh)
        warnings = validate_prices(prices, settings.data.vix_ticker)
    except DataError as exc:
        console.print(f"[red]Data error: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    table = Table(title="Aligned price data by group")
    for col in ("group", "tickers", "first date", "last date", "trading days", "rows dropped"):
        table.add_column(col)
    for group in settings.universe:
        aligned, dropped = align_group(prices, settings, group)
        table.add_row(
            group,
            ", ".join(settings.universe[group]),
            str(aligned.index[0].date()) if len(aligned) else "-",
            str(aligned.index[-1].date()) if len(aligned) else "-",
            str(len(aligned)),
            str(dropped),
        )
    console.print(table)
    for warning in warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")


@app.command()
def report() -> None:
    """Compute all results, write figures and results files, update README."""
    load_dotenv()
    started = time.perf_counter()
    settings = load_settings()
    prereg = load_preregistered(settings)
    try:
        prices = load_or_fetch(settings)
    except DataError as exc:
        console.print(f"[red]Data error: {exc}[/red]")
        raise typer.Exit(code=1) from exc
    results = generate_report(settings, prereg, prices, find_project_root())
    console.print(headline_sentence(results))
    console.print(f"Report written in {time.perf_counter() - started:.1f}s.")


def _agent_setup() -> tuple[Any, ResearchService, Path]:
    """Create the client and service, exiting with a red message on failure."""
    try:
        client = make_client()
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    try:
        service = ResearchService.from_cache()
    except DataError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    return client, service, find_project_root()


def _render(result: AgentResult, transcript: TranscriptWriter, show_tools: bool) -> None:
    console.print(Markdown(result.final_text or "_(no answer)_"))
    if show_tools:
        for call in result.tool_calls:
            out = json.dumps(call.output)
            if len(out) > 600:
                out = out[:600] + "..."
            flag = " [error]" if call.is_error else ""
            console.print(
                f"[dim]tool {call.name}{flag} {json.dumps(call.input)}\n  -> {out}[/dim]",
                markup=True,
                highlight=False,
            )
    console.print(
        f"[dim]iterations {result.iterations} | tokens in {result.usage['input_tokens']} "
        f"out {result.usage['output_tokens']} | transcript {transcript.path}[/dim]"
    )


@app.command()
def ask(question: str, show_tools: bool = typer.Option(False, "--show-tools")) -> None:
    """Ask the research agent one question."""
    client, service, root = _agent_setup()
    transcript = TranscriptWriter(root / "runs", question)
    with console.status("Thinking..."):
        result = run_agent(
            question, client=client, service=service, model=agent_model(), transcript=transcript
        )
    _render(result, transcript, show_tools)


@app.command()
def chat(show_tools: bool = typer.Option(False, "--show-tools")) -> None:
    """Multi-turn chat with the research agent."""
    client, service, root = _agent_setup()
    history: list[dict[str, Any]] = []
    console.print(
        "[dim]Commands: /tools toggles tool display, /reset clears history, /exit quits.[/dim]"
    )
    while True:
        try:
            question = console.input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question == "/exit":
            break
        if question == "/reset":
            history = []
            console.print("History cleared.")
            continue
        if question == "/tools":
            show_tools = not show_tools
            console.print(f"Tool display {'on' if show_tools else 'off'}.")
            continue
        transcript = TranscriptWriter(root / "runs", question)
        try:
            with console.status("Thinking..."):
                result = run_agent(
                    question,
                    client=client,
                    service=service,
                    model=agent_model(),
                    history=history,
                    transcript=transcript,
                )
        except anthropic.APIError as exc:
            console.print(f"[red]API error: {exc}[/red]")
            continue
        history = result.messages
        _render(result, transcript, show_tools)


@app.command("eval")
def eval_cmd(
    category: str | None = typer.Option(None, help="Only run cases in this category."),
    case: str | None = typer.Option(None, "--case", help="Only run this case id."),
    repeats: int = typer.Option(1, min=1, max=5, help="Runs per case."),
    max_cases: int | None = typer.Option(None, "--max-cases", min=1),
    calibrate_judge: bool = typer.Option(False, "--calibrate-judge"),
) -> None:
    """Run the eval suite (or calibrate the LLM judge)."""
    try:
        client = make_client()
    except MissingAPIKeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    root = find_project_root()
    if calibrate_judge:
        agree, total, disagreements = calibrate(
            client, judge_model(), root / "evals" / "judge_calibration.yaml"
        )
        for d in disagreements:
            console.print(f"[red]{d}[/red]")
        console.print(f"Judge calibration: {agree}/{total}")
        raise typer.Exit(code=0 if agree == total else 1)

    try:
        cases = load_cases(root / "evals" / "cases")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if category:
        cases = [c for c in cases if c.category == category]
    if case:
        cases = [c for c in cases if c.id == case]
    if max_cases:
        cases = cases[:max_cases]
    if not cases:
        console.print("[red]No cases match the filters.[/red]")
        raise typer.Exit(code=1)
    try:
        service = ResearchService.from_cache()
    except DataError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    out_dir = run_suite(cases, repeats, root, service, client, client, agent_model(), judge_model())
    console.print(Markdown((out_dir / "summary.md").read_text()))
    console.print(f"Results written to {out_dir}")
