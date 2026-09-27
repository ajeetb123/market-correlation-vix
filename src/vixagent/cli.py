"""Command-line interface for vixagent."""

import typer

from vixagent import __version__

app = typer.Typer(help="VIX research agent CLI.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """VIX research agent CLI."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)
