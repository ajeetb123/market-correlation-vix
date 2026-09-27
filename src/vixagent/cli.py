"""Command-line interface for vixagent."""

from __future__ import annotations

import time

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from vixagent import __version__
from vixagent.config import find_project_root, load_preregistered, load_settings
from vixagent.data.align import align_group
from vixagent.data.cache import load_or_fetch
from vixagent.data.fetch import DataError
from vixagent.data.validate import validate_prices
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
