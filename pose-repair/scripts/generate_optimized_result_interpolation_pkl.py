from __future__ import annotations

import argparse
import json
from pathlib import Path

from dongfeng_registration.result_interpolation_export import (
    apply_optimized_transforms_to_payload,
    delta_transform,
    generate_optimized_result_interpolation_pkl,
    validate_optimized_result_interpolation_pkl,
)

__all__ = [
    "apply_optimized_transforms_to_payload",
    "delta_transform",
    "generate_optimized_result_interpolation_pkl",
    "validate_optimized_result_interpolation_pkl",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate result_interpolation_optimized.pkl from optimized multi-camera output."
    )
    parser.add_argument("--input-pkl", type=Path, required=True)
    parser.add_argument("--optimized-yaml", type=Path, required=True)
    parser.add_argument("--image-extrinsics-csv", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path)
    parser.add_argument("--validation-json", type=Path)
    parser.add_argument(
        "--source-ego-field",
        default="ego2global_transformation_matrix",
        help="Field used as the base ego pose before applying optimized body deltas.",
    )
    parser.add_argument(
        "--optimized-ego-output-field",
        default="ego2global_transformation_matrix",
        help="Field written with the optimized ego pose.",
    )
    args = parser.parse_args()
    manifest = generate_optimized_result_interpolation_pkl(
        input_pkl=args.input_pkl,
        optimized_yaml=args.optimized_yaml,
        image_extrinsics_csv=args.image_extrinsics_csv,
        output_pkl=args.output_pkl,
        manifest_json=args.manifest_json,
        validation_json=args.validation_json,
        source_ego_field=args.source_ego_field,
        optimized_ego_output_field=args.optimized_ego_output_field,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
