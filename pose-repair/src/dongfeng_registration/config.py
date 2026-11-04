from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator

BACKEND_NAMES = ("sift_lk", "superpoint_lightglue", "loftr", "loma")
BackendName = Literal[*BACKEND_NAMES]


class MatchingConfig(BaseModel):
    backend: BackendName = "sift_lk"
    lk_mode: Literal["on", "off"] = "on"
    model_device: str = "cuda:0"
    # The internal names stay generic because SuperPoint/LoFTR also use them.
    # LoMa configurations use the clearer loma_* aliases.
    deep_image_size: int = Field(
        default=1600,
        validation_alias=AliasChoices("loma_image_size", "deep_image_size"),
    )
    deep_max_keypoints: int = Field(
        default=2048,
        validation_alias=AliasChoices("loma_max_keypoints", "deep_max_keypoints"),
    )
    deep_score_threshold: float = Field(
        default=0.0,
        validation_alias=AliasChoices("loma_score_threshold", "deep_score_threshold"),
    )
    deep_score_threshold_intra: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "loma_score_threshold_intra", "deep_score_threshold_intra"
        ),
    )
    deep_score_threshold_inter: float | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "loma_score_threshold_inter", "deep_score_threshold_inter"
        ),
    )
    loma_feature_cache_mode: Literal["two_tier", "cpu", "model"] = "model"
    loma_feature_cache_gpu_max_items: int = Field(default=5000, ge=1)
    loma_feature_cache_cpu_max_items: int | None = Field(default=None, ge=1)
    loma_matcher_compile_backend: Literal["off", "cudagraphs"] = "off"
    deep_batch_size: int = 4
    loma_process_count: int = Field(default=1, ge=1)
    filter_workers: int = Field(default=4, ge=1)
    overlap_filtering: bool = True
    max_features: int = 3000
    ratio_threshold: float = 0.75
    lk_forward_backward_threshold: float = 1.5
    ransac_threshold_pixels: float = 1.5
    visualization_pairs: int = 3
    cache_visualization_pair_ratio: float = Field(default=0.01, gt=0.0, le=1.0)
    min_texture_strength: float = 0.0
    min_corner_response: float = 0.0


class SemanticSegmentationConfig(BaseModel):
    processor_image_size: int = Field(default=1024, ge=1)
    inference_dtype: Literal["fp32", "fp16", "bf16"] = "fp16"
    use_fast_processor: bool = True
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    dilation_pixels: int = Field(default=5, ge=0)
    batch_size: int = Field(default=4, ge=1)
    write_workers: int = Field(default=1, ge=1)
    preview_ratio: float = Field(default=0.01, ge=0.0, le=1.0)


class ImageCacheConfig(BaseModel):
    enabled: bool = True
    mode: Literal["lazy", "preload", "stream"] = "preload"
    max_long_side: int = Field(default=1600, ge=1)
    max_items: int | None = Field(default=1024, ge=1)


class OptimizationConfig(BaseModel):
    pair_sample_max_matches: int | None = Field(default=None, ge=1)
    pair_sample_seed: int = 20260704
    pair_sample_strategy: Literal["random", "spatial_grid_random"] = "spatial_grid_random"
    pair_sample_grid_cols: int = Field(default=3, ge=1)
    pair_sample_grid_rows: int = Field(default=3, ge=1)
    max_nfev: int = Field(default=1500, ge=1)
    torch_learning_rate: float = Field(default=0.003, ge=0.0)
    torch_lr_schedule: Literal["none", "constant", "step"] = "step"
    torch_lr_step_size: int = Field(default=100, ge=1)
    torch_lr_decay: float = Field(default=0.5, ge=0.0)
    torch_lr_min_factor: float = Field(default=0.1, ge=0.0)
    torch_early_stopping_min_delta: float = Field(default=1e-3, ge=0.0)
    torch_early_stopping_min_steps: int = Field(default=500, ge=0)
    inter_weight: float = Field(default=1.0, ge=0.0)
    translation_bound_m: float = Field(default=1.0, ge=0.0)
    rotation_bound_deg: float = Field(default=10.0, ge=0.0)
    fixed_translation_l2_weight: float = 50.0
    fixed_rotation_l2_weight: float = 300.0
    optimize_time_varying_disturbance: bool = False
    time_varying_disturbance_dof: int = Field(default=6, ge=3, le=6)
    time_varying_rotation_bound_deg: float = Field(default=15.0, ge=0.0)
    time_varying_translation_bound_m: float = Field(default=0.15, ge=0.0)
    time_varying_translation_l2_weight: float = 200.0
    time_varying_rotation_l2_weight: float = 5.0
    time_varying_translation_smooth_l2_weight: float = 20.0
    time_varying_rotation_smooth_l2_weight: float = 0.5
    time_varying_translation_mean_l2_weight: float = 5000.0
    time_varying_rotation_mean_l2_weight: float = 2500.0
    time_varying_frame_common_translation_l2_weight: float = 5000.0
    time_varying_frame_common_rotation_l2_weight: float = 2500.0
    optimize_body_time_varying_disturbance: bool = False
    body_time_varying_disturbance_dof: int = Field(default=6, ge=3, le=6)
    body_time_varying_rotation_bound_deg: float = Field(default=15.0, ge=0.0)
    body_time_varying_translation_bound_m: float = Field(default=0.15, ge=0.0)
    body_time_varying_translation_l2_weight: float = 200.0
    body_time_varying_rotation_l2_weight: float = 5.0
    body_time_varying_translation_smooth_l2_weight: float = 20.0
    body_time_varying_rotation_smooth_l2_weight: float = 0.5
    mode_projection_enabled: bool = False
    mode_projection_interval_steps: int = Field(default=100, ge=1)
    mode_projection_absorb_frame_common: bool = True
    mode_projection_absorb_camera_long_term: bool = True
    mode_projection_reset_optimizer_state: bool = True
    absorb_camera_common_mode: bool = True
    camera_common_mode_min_cameras: int = 3
    camera_common_mode_min_median_cos: float = 0.5
    camera_common_mode_min_positive_projection_fraction: float = 0.8
    absorb_camera_long_term_mode: bool = True
    camera_long_term_mode_min_frames: int = 30
    camera_long_term_mode_min_median_cos: float = 0.5
    camera_long_term_mode_min_positive_projection_fraction: float = 0.8
    camera_long_term_mode_min_mean_to_rms: float = 0.35

    @field_validator("time_varying_disturbance_dof", "body_time_varying_disturbance_dof")
    @classmethod
    def validate_disturbance_dof(cls, value: int) -> int:
        if value not in (3, 6):
            raise ValueError("disturbance dof must be either 3 or 6")
        return value


class PreflightDiagnosticConfig(BaseModel):
    enabled: bool = True
    sample_strategy: Literal["temporal_stratified"] = "temporal_stratified"
    sample_pairs_per_relation: int = Field(default=20, ge=1)
    max_total_pairs: int = Field(default=800, ge=1)
    random_seed: int = 20260704
    reuse_preflight_matches: bool = True
    min_raw_matches_per_pair: int = Field(default=20, ge=0)
    min_ransac_inlier_ratio: float = Field(default=0.20, ge=0.0, le=1.0)
    min_valid_depth_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    reprojection_median_warn_px: float = Field(default=5.0, ge=0.0)
    reprojection_median_fail_px: float = Field(default=20.0, ge=0.0)
    reprojection_p95_warn_px: float = Field(default=15.0, ge=0.0)
    reprojection_p95_fail_px: float = Field(default=50.0, ge=0.0)
    max_bad_pair_fraction: float = Field(default=0.25, ge=0.0, le=1.0)
    min_bad_pairs_for_failure: int = Field(default=3, ge=1)
    min_failed_relation_fraction_for_failure: float = Field(
        default=0.20, ge=0.0, le=1.0
    )
    min_intra_valid_pairs: int = Field(default=3, ge=0)
    min_inter_valid_pairs: int = Field(default=3, ge=0)
    min_camera_graph_degree: int = Field(default=1, ge=0)
    visualize_top_bad_pairs: int = Field(default=20, ge=0)
    fail_action: Literal["stop_before_dynamic_compensation", "report_only"] = (
        "stop_before_dynamic_compensation"
    )


class PipelineConfig(BaseModel):
    dataset_root: Path
    output_root: Path
    run_name: str = "baseline"
    cameras: list[str]
    frame_offsets: list[int] = Field(default_factory=lambda: [1])
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    semantic_segmentation: SemanticSegmentationConfig = Field(
        default_factory=SemanticSegmentationConfig
    )
    image_cache: ImageCacheConfig = Field(default_factory=ImageCacheConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    preflight_diagnostic: PreflightDiagnosticConfig = Field(
        default_factory=PreflightDiagnosticConfig
    )
    assumptions: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_image_cache_resolution(self) -> "PipelineConfig":
        required = max(
            int(self.semantic_segmentation.processor_image_size),
            int(self.matching.deep_image_size),
        )
        if self.image_cache.enabled and self.image_cache.max_long_side < required:
            raise ValueError(
                "image cache resolution must be at least the semantic processor "
                f"and LoMa resolution ({required})"
            )
        return self

    @field_validator("frame_offsets")
    @classmethod
    def validate_offsets(cls, value: list[int]) -> list[int]:
        if not value or any(offset <= 0 for offset in value):
            raise ValueError("frame offsets must be positive")
        return value


def load_config(path: Path) -> PipelineConfig:
    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("configuration file must contain a mapping")
    missing = [key for key in ("dataset_root", "output_root", "cameras") if key not in raw]
    if missing:
        raise ValueError(f"configuration file is missing required keys: {', '.join(missing)}")
    raw = dict(raw)
    for key in ("dataset_root", "output_root"):
        candidate = Path(raw[key])
        if not candidate.is_absolute():
            raw[key] = (path.parent / candidate).resolve()
    return PipelineConfig.model_validate(raw)
