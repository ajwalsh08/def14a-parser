"""Command-line interface for def14a-parser."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from tqdm import tqdm

from .extractor import extract, DEFAULT_MODEL


@click.group()
def cli() -> None:
    """Extract CEO compensation and governance data from SEC DEF 14A proxy statements."""


@cli.command()
@click.argument("files", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="Ollama model name (must be pulled locally).")
@click.option("--output", "-o", default=None,
              help="Output file path. Defaults to stdout (single file) or "
                   "<filename>.json (batch).")
@click.option("--pretty/--no-pretty", default=True,
              help="Pretty-print JSON output.")
def run(files: tuple[str, ...], model: str, output: str | None, pretty: bool) -> None:
    """
    Extract CEO and governance data from one or more DEF 14A HTML files.

    \b
    Single file — result goes to stdout (or --output file):
        def14a run primary.htm

    \b
    Batch mode — each file gets a matching .json sidecar:
        def14a run filings/*/primary.htm --model qwen2.5:7b
    """
    indent = 2 if pretty else None
    paths  = [Path(f) for f in files]
    batch  = len(paths) > 1

    if batch and output:
        raise click.UsageError("--output is not supported in batch mode; "
                               "each file gets a <name>.json sidecar.")

    iterator = tqdm(paths, desc=f"DEF 14A → {model}") if batch else paths

    succeeded = skipped = 0
    for path in iterator:
        result = extract(path, model=model)

        if result is None:
            skipped += 1
            if not batch:
                click.echo(f"No compensation data found in {path}", err=True)
            continue

        result["source_file"] = str(path)
        json_str = json.dumps(result, indent=indent)

        if batch:
            out_path = path.with_suffix(".json")
            out_path.write_text(json_str)
            succeeded += 1
        else:
            if output:
                Path(output).write_text(json_str)
                click.echo(f"Wrote {output}")
            else:
                click.echo(json_str)
            succeeded += 1

    if batch:
        click.echo(f"\nDone: {succeeded} extracted, {skipped} skipped")


def main() -> None:
    cli()
