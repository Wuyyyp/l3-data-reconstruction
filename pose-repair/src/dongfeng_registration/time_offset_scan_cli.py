from pathlib import Path

import typer

from .time_offset_scan import run_time_offset_scan_from_retained_matches


app = typer.Typer(no_args_is_help=True)


@app.command()
def scan(
    config: Path = typer.Option(...),
    retained_matches_input: Path = typer.Option(
        ...,
        "--retained-matches-input",
        help="Retained-match input. Pass a new match cache dir or a retained_matches.parquet file.",
    ),
    output: Path = typer.Option(...),
    camera: str = typer.Option("center_camera_fov30"),
    scan_from_ms: int = typer.Option(-300),
    scan_to_ms: int = typer.Option(300),
    step_ms: int = typer.Option(10, min=1),
) -> None:
    result = run_time_offset_scan_from_retained_matches(
        config_path=config,
        retained_matches_input=retained_matches_input,
        output=output,
        camera=camera,
        scan_from_ms=scan_from_ms,
        scan_to_ms=scan_to_ms,
        step_ms=step_ms,
    )
    typer.echo(f"Time offset scan report: {result / f'{camera}_time_offset_scan_report.md'}")


if __name__ == "__main__":
    app()
