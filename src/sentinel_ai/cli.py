"""Command-line interface for sentinel-ai."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import click

from .analyser import SentinelAnalyser


@click.group()
@click.version_option(package_name="sentinel_ai")
def main() -> None:
    """Detect relational manipulation patterns in multi-session AI conversations."""


@main.command()
@click.option(
    "--transcripts",
    required=True,
    type=click.Path(exists=True),
    help="Path to transcripts directory or single JSON file.",
)
@click.option(
    "--domain",
    default=None,
    help="Built-in domain profile name (defence-welfare, general).",
)
@click.option(
    "--domain-profile",
    default=None,
    type=click.Path(exists=True),
    help="Path to custom domain profile YAML.",
)
@click.option(
    "--output",
    "-o",
    required=True,
    type=click.Path(),
    help="Output file path (.json or .html).",
)
@click.option(
    "--judge-model",
    default=None,
    help="LLM judge model (e.g. anthropic/claude-sonnet-4.5, ollama/llama3.1:8b).",
)
@click.option(
    "--api-key",
    default=None,
    help="API key for the judge model. Defaults to ANTHROPIC_API_KEY or OPENAI_API_KEY env var.",
)
def analyse(
    transcripts: str,
    domain: str | None,
    domain_profile: str | None,
    output: str,
    judge_model: str | None,
    api_key: str | None,
) -> None:
    """Analyse conversation transcripts for relational manipulation patterns."""
    if api_key is None and judge_model is not None:
        if "anthropic" in judge_model:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

    transcripts_path = Path(transcripts)
    analyser = SentinelAnalyser(
        transcripts_dir=transcripts_path if transcripts_path.is_dir() else None,
        transcripts_file=transcripts_path if transcripts_path.is_file() else None,
        domain=domain,
        domain_profile=domain_profile,
        judge_model=judge_model,
        api_key=api_key,
    )

    click.echo("Analysing transcripts...")
    try:
        report = analyser.analyse()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    output_path = Path(output)
    if output_path.suffix == ".html":
        report.to_html(output_path)
    else:
        report.to_json(output_path)

    severity = report.output.overall_severity.value
    n_sessions = len(report.output.session_trajectory)
    scores = {cs.category.value: f"{cs.score:.2f}" for cs in report.output.category_scores}
    click.echo(f"Done. {n_sessions} sessions analysed. Severity: {severity}")
    click.echo(f"Scores: {scores}")
    click.echo(f"Output written to {output_path}")
