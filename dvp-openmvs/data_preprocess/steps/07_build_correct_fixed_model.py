#!/usr/bin/env python3
"""Build a COLMAP text model from manifest camera-to-world poses without BA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CAMERAS = ("00", "01", "02", "03", "04", "09", "10")


def image_headers(path: Path):
    headers = []
    for line in path.open(encoding="utf-8"):
        fields = line.split()
        if len(fields) >= 10 and fields[9].lower().endswith((".jpg", ".jpeg", ".png")):
            headers.append(line.strip())
    return headers


def rotmat_to_qvec(rotation):
    rxx, ryx, rzx, rxy, ryy, rzy, rxz, ryz, rzz = rotation.flat
    matrix = np.array([
        [rxx - ryy - rzz, ryx + rxy, rzx + rxz, rzy - ryz],
        [ryx + rxy, ryy - rxx - rzz, rzy + ryz, rxz - rzx],
        [rzx + rxz, rzy + ryz, rzz - rxx - ryy, ryx - rxy],
        [rzy - ryz, rxz - rzx, ryx - rxy, rxx + ryy + rzz],
    ]) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    qvec = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    # The symmetric-matrix construction above yields the conjugate under the
    # row-major convention used here; convert it to COLMAP's world-to-camera qvec.
    qvec[1:] *= -1
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def qvec_to_rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--extrinsics", type=Path, required=True)
    parser.add_argument("--template-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    poses = {record["label"]: np.asarray(record["camera_to_world"], dtype=float) for record in manifest["records"]}
    template_headers = image_headers(args.template_model / "images.txt")
    if len(template_headers) != 707 or set(poses) != {line.split()[9] for line in template_headers}:
        raise RuntimeError("template image set and manifest image set differ")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cameras.txt").write_text((args.template_model / "cameras.txt").read_text(), encoding="utf-8")
    (args.output / "points3D.txt").write_text("# Empty point list; point_triangulator fills this model\n", encoding="utf-8")

    rows = [
        "# Fixed metric poses corrected from input_manifest camera_to_world",
        "# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME",
    ]
    center_errors = []
    rotation_errors = []
    ego_by_frame = {f"{frame:06d}": [] for frame in range(101)}
    for line in template_headers:
        fields = line.split()
        image_id, camera_id, name = int(fields[0]), int(fields[8]), fields[9]
        camera_to_world = poses[name].copy()
        u, _, vt = np.linalg.svd(camera_to_world[:3, :3])
        camera_to_world[:3, :3] = u @ vt
        if np.linalg.det(camera_to_world[:3, :3]) < 0:
            u[:, -1] *= -1
            camera_to_world[:3, :3] = u @ vt
        world_to_camera = camera_to_world[:3, :3].T
        center = camera_to_world[:3, 3]
        translation = -world_to_camera @ center
        qvec = rotmat_to_qvec(world_to_camera)
        recovered_rotation = qvec_to_rotmat(qvec)
        recovered_center = -recovered_rotation.T @ translation
        center_errors.append(float(np.linalg.norm(recovered_center - center)))
        rotation_errors.append(float(np.linalg.norm(recovered_rotation - world_to_camera)))
        values = [image_id, *qvec.tolist(), *translation.tolist(), camera_id, name]
        rows.append(" ".join(str(value) for value in values))
        rows.append("")

        stem = Path(name).stem
        extrinsic = np.loadtxt(args.extrinsics / f"{stem}.txt")
        ego_to_world = camera_to_world @ extrinsic
        ego_by_frame[stem.split("_")[0]].append(ego_to_world[:3, 3])

    (args.output / "images.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    ego_origins = {}
    ego_residuals = []
    for frame, origins in ego_by_frame.items():
        origins = np.asarray(origins)
        mean = origins.mean(axis=0)
        ego_origins[frame] = mean.tolist()
        ego_residuals.extend(np.linalg.norm(origins - mean, axis=1).tolist())
    ego_residuals = np.asarray(ego_residuals)
    audit = {
        "status": "PASS",
        "images": len(template_headers),
        "pose_source": "input_manifest.json camera_to_world",
        "colmap_conversion": "R_world_to_camera = R_camera_to_world.T; t = -R_world_to_camera @ C",
        "bundle_adjustment": False,
        "center_roundtrip_max_m": max(center_errors),
        "rotation_roundtrip_frobenius_max": max(rotation_errors),
        "ego_origin_formula": "ego_to_world = camera_to_world @ ego_to_camera_extrinsic",
        "ego_cross_camera_residual_median_m": float(np.median(ego_residuals)),
        "ego_cross_camera_residual_p90_m": float(np.percentile(ego_residuals, 90)),
        "ego_cross_camera_residual_max_m": float(ego_residuals.max()),
        "ego_cross_camera_consistency": "PASS" if np.percentile(ego_residuals, 90) <= 1e-3 else "WARN",
        "ego_origins_world_by_frame": ego_origins,
    }
    if audit["center_roundtrip_max_m"] > 1e-8 or audit["rotation_roundtrip_frobenius_max"] > 1e-8:
        audit["status"] = "FAIL"
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    compact = dict(audit)
    compact.pop("ego_origins_world_by_frame")
    print(json.dumps(compact, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
