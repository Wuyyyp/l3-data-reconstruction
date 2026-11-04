from pathlib import Path
from typing import Literal

import typer

from .time_varying_rotation import run_time_varying_rotation_diagnostic

app = typer.Typer(no_args_is_help=True)


@app.callback(invoke_without_command=True)
def diagnose(
    config: Path = typer.Option(Path("configs/t3.yaml"), "--config"),
    retained_matches_input: Path = typer.Option(
        Path("data_cache/matches"),
        "--retained-matches-input",
        help="Retained-match input. Pass a new match cache dir or a retained_matches.parquet file.",
    ),
    output: Path = typer.Option(
        Path("outputs/center_camera_fov30_time_varying_rotation_diagnostic_navp60"),
        "--output",
    ),
    camera: str = typer.Option("center_camera_fov30", "--camera"),
    max_observations: int = typer.Option(
        0,
        "--max-observations",
        min=0,
        help="0 means use all retained matches for optimization.",
    ),
    rotation_bound_deg: float = typer.Option(0.5, "--rotation-bound-deg", min=0.0),
    yaw_bound_deg: float = typer.Option(0.2, "--yaw-bound-deg", min=0.0),
    rotation_prior_sigma_deg: float = typer.Option(
        0.12, "--rotation-prior-sigma-deg", min=0.0
    ),
    yaw_prior_sigma_deg: float = typer.Option(0.06, "--yaw-prior-sigma-deg", min=0.0),
    disturbance_dof: Literal[3, 6] = typer.Option(3, "--disturbance-dof"),
    translation_bound_m: float = typer.Option(0.05, "--translation-bound-m", min=0.0),
    translation_prior_sigma_m: float = typer.Option(
        0.03, "--translation-prior-sigma-m", min=1e-12
    ),
    translation_smooth_sigma_m: float = typer.Option(
        0.03, "--translation-smooth-sigma-m", min=1e-12
    ),
    smooth_sigma_deg: float = typer.Option(0.08, "--smooth-sigma-deg", min=0.0),
    roll_smooth_sigma_deg: float | None = typer.Option(
        None, "--roll-smooth-sigma-deg", min=0.0
    ),
    pitch_smooth_sigma_deg: float | None = typer.Option(
        None, "--pitch-smooth-sigma-deg", min=0.0
    ),
    yaw_smooth_sigma_deg: float | None = typer.Option(
        None, "--yaw-smooth-sigma-deg", min=0.0
    ),
    max_nfev: int = typer.Option(100, "--max-nfev", min=1),
    knot_stride_frames: int = typer.Option(10, "--knot-stride-frames", min=1),
    export_pair_visualizations: bool = typer.Option(
        False, "--export-pair-visualizations/--no-export-pair-visualizations"
    ),
    visualization_max_pairs: int | None = typer.Option(
        None,
        "--visualization-max-pairs",
        help="Limit full pair reprojection visualizations; omit for all pairs.",
    ),
    visualization_max_per_row: int = typer.Option(
        150, "--visualization-max-per-row", min=1
    ),
    coordinate_sweeps: int = typer.Option(6, "--coordinate-sweeps", min=1),
    optimizer_method: Literal[
        "torch_sparse_graph", "sparse_graph", "coordinate_descent"
    ] = typer.Option(
        "torch_sparse_graph", "--optimizer-method"
    ),
    torch_learning_rate: float = typer.Option(0.03, "--torch-learning-rate", min=0.0),
    torch_device: str = typer.Option("cuda:0", "--torch-device"),
    temporal_regularizer: Literal["none", "l2", "huber"] = typer.Option(
        "l2", "--temporal-regularizer"
    ),
    temporal_huber_delta: float = typer.Option(
        1.0, "--temporal-huber-delta", min=0.0
    ),
    use_prior_regularizer: bool = typer.Option(
        True, "--use-prior-regularizer/--no-prior-regularizer"
    ),
    imu_adaptive_temporal: bool = typer.Option(
        False, "--imu-adaptive-temporal/--no-imu-adaptive-temporal"
    ),
    imu_shock_window_ms: float = typer.Option(120.0, "--imu-shock-window-ms", min=1.0),
    imu_shock_relax_factor: float = typer.Option(
        3.0, "--imu-shock-relax-factor", min=1.0
    ),
    use_reprojection_term: bool = typer.Option(
        False, "--use-reprojection-term/--no-use-reprojection-term"
    ),
    reprojection_loss_weight: float = typer.Option(
        0.25, "--reprojection-loss-weight", min=0.0
    ),
    lr_schedule: Literal["constant", "step", "cosine"] = typer.Option(
        "constant", "--lr-schedule"
    ),
    lr_step_size: int = typer.Option(300, "--lr-step-size", min=1),
    lr_decay: float = typer.Option(0.5, "--lr-decay", min=0.0),
    lr_min_factor: float = typer.Option(0.1, "--lr-min-factor", min=0.0),
) -> None:
    result = run_time_varying_rotation_diagnostic(
        config_path=config,
        retained_matches_input=retained_matches_input,
        output=output,
        camera=camera,
        max_observations=None if max_observations == 0 else max_observations,
        rotation_bound_deg=rotation_bound_deg,
        yaw_bound_deg=yaw_bound_deg,
        rotation_prior_sigma_deg=rotation_prior_sigma_deg,
        yaw_prior_sigma_deg=yaw_prior_sigma_deg,
        disturbance_dof=disturbance_dof,
        translation_bound_m=translation_bound_m,
        translation_prior_sigma_m=translation_prior_sigma_m,
        translation_smooth_sigma_m=translation_smooth_sigma_m,
        smooth_sigma_deg=smooth_sigma_deg,
        roll_smooth_sigma_deg=roll_smooth_sigma_deg,
        pitch_smooth_sigma_deg=pitch_smooth_sigma_deg,
        yaw_smooth_sigma_deg=yaw_smooth_sigma_deg,
        max_nfev=max_nfev,
        knot_stride_frames=knot_stride_frames,
        export_pair_visualizations=export_pair_visualizations,
        visualization_max_pairs=visualization_max_pairs,
        visualization_max_per_row=visualization_max_per_row,
        coordinate_sweeps=coordinate_sweeps,
        optimizer_method=optimizer_method,
        torch_learning_rate=torch_learning_rate,
        torch_device=torch_device,
        temporal_regularizer=temporal_regularizer,
        temporal_huber_delta=temporal_huber_delta,
        use_prior_regularizer=use_prior_regularizer,
        imu_adaptive_temporal=imu_adaptive_temporal,
        imu_shock_window_ms=imu_shock_window_ms,
        imu_shock_relax_factor=imu_shock_relax_factor,
        use_reprojection_term=use_reprojection_term,
        reprojection_loss_weight=reprojection_loss_weight,
        lr_schedule=lr_schedule,
        lr_step_size=lr_step_size,
        lr_decay=lr_decay,
        lr_min_factor=lr_min_factor,
    )
    typer.echo(f"Time-varying rotation diagnostic report: {result / f'{camera}_time_varying_rotation_report.md'}")


if __name__ == "__main__":
    app()

