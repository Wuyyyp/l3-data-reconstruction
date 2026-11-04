#!/usr/bin/env python3
"""Build conservative, building-safe sky masks for the full 707-image clip."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation


CAMERAS = ("00", "01", "02", "03", "04", "09", "10")


def connected_sky(mask: np.ndarray) -> np.ndarray:
    """Keep confident sky components reaching the upper third, including sky below windshield occlusions."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    h, w = mask.shape
    x, y, cw, _ch, area = stats.T
    valid = ((y < h // 3) | ((x <= 2) & (y < h // 3)) |
             ((x + cw >= w - 2) & (y < h // 3))) & (area >= 300)
    valid[0] = False
    return valid[labels]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--device", default="cuda:3")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--sky-prob", type=float, default=0.60)
    p.add_argument("--sky-margin", type=float, default=0.25)
    p.add_argument("--erosion-px", type=int, default=5)
    p.add_argument("--sample-only", action="store_true")
    args = p.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)

    stems = [f"{frame:06d}_{cam}" for frame in range(101) for cam in CAMERAS]
    if args.sample_only:
        stems = [f"{frame:06d}_{cam}" for frame in (0, 25, 50, 75, 100) for cam in CAMERAS]
    for name in ("raw_sky", "safe_sky", "openmvs_masks", "overlays"):
        (args.output / name).mkdir(parents=True, exist_ok=True)

    model_source = str(args.cache) if (args.cache / "config.json").is_file() else args.model
    processor = AutoImageProcessor.from_pretrained(model_source, cache_dir=args.cache)
    model = SegformerForSemanticSegmentation.from_pretrained(model_source, cache_dir=args.cache).to(args.device).eval()
    labels = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    sky_ids = [k for k, v in labels.items() if v.strip() == "sky"]
    building_ids = [k for k, v in labels.items() if any(word in v for word in ("building", "skyscraper", "house"))]
    if len(sky_ids) != 1:
        raise RuntimeError(f"Expected one sky class, found {sky_ids}: {labels}")
    sky_id = sky_ids[0]
    kernel_size = 2 * args.erosion_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    totals = defaultdict(int)
    per_camera = defaultdict(lambda: defaultdict(int))
    per_image = {}
    with torch.inference_mode():
        for start in range(0, len(stems), args.batch_size):
            batch_stems = stems[start:start + args.batch_size]
            pil_images = [Image.open(args.images / f"{stem}.jpg").convert("RGB") for stem in batch_stems]
            inputs = processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            logits = model(**inputs).logits
            for i, (stem, pil) in enumerate(zip(batch_stems, pil_images)):
                w, h = pil.size
                low_prob = logits[i].softmax(dim=0)
                low_pred_sky = (low_prob.argmax(dim=0) == sky_id).float()
                sky_prob = F.interpolate(low_prob[sky_id][None, None], size=(h, w),
                                         mode="bilinear", align_corners=False)[0, 0]
                pred_sky = F.interpolate(low_pred_sky[None, None], size=(h, w),
                                         mode="nearest")[0, 0] > 0.5
                if building_ids:
                    low_building = low_prob[building_ids].amax(dim=0)
                    building_prob = F.interpolate(low_building[None, None], size=(h, w),
                                                  mode="bilinear", align_corners=False)[0, 0]
                else:
                    building_prob = torch.zeros_like(sky_prob)
                raw = (pred_sky & (sky_prob >= args.sky_prob) &
                       ((sky_prob - building_prob) >= args.sky_margin)).cpu().numpy()
                raw = connected_sky(raw)
                safe = cv2.erode(raw.astype(np.uint8), kernel, iterations=1) > 0

                raw_u8 = raw.astype(np.uint8) * 255
                safe_u8 = safe.astype(np.uint8) * 255
                cv2.imwrite(str(args.output / "raw_sky" / f"{stem}.png"), raw_u8)
                cv2.imwrite(str(args.output / "safe_sky" / f"{stem}.png"), safe_u8)
                cv2.imwrite(str(args.output / "openmvs_masks" / f"{stem}.mask.png"), safe_u8)

                if int(stem[:6]) in (0, 25, 50, 75, 100):
                    rgb = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
                    overlay = rgb.copy()
                    overlay[raw & ~safe] = (0, 0, 255)
                    overlay[safe] = (255, 255, 0)
                    overlay = cv2.addWeighted(rgb, 0.58, overlay, 0.42, 0)
                    cv2.putText(overlay, "cyan=safe sky; red=protected boundary", (20, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                    overlay = cv2.resize(overlay, (960, 640), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(args.output / "overlays" / f"{stem}.jpg"), overlay,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])

                cam = stem.rsplit("_", 1)[1]
                values = {
                    "pixels": h * w,
                    "raw_sky_pixels": int(raw.sum()),
                    "safe_sky_pixels": int(safe.sum()),
                    "protected_boundary_pixels": int((raw & ~safe).sum()),
                }
                per_image[stem] = values
                for key, value in values.items():
                    totals[key] += value
                    per_camera[cam][key] += value
            print(f"processed {min(start + args.batch_size, len(stems))}/{len(stems)}", flush=True)

    audit = {
        "status": "PASS",
        "model": args.model,
        "model_labels": labels,
        "sky_class_id": sky_id,
        "building_class_ids": building_ids,
        "rule": {
            "argmax_must_be_sky": True,
            "minimum_sky_probability": args.sky_prob,
            "minimum_sky_minus_building_probability": args.sky_margin,
            "component_rule": "high-confidence sky reaching upper third; windshield occlusions allowed",
            "erosion_into_sky_px_at_1920x1280": args.erosion_px,
        },
        "image_count": len(stems),
        "totals": dict(totals),
        "safe_sky_fraction": totals["safe_sky_pixels"] / max(1, totals["pixels"]),
        "per_camera": {cam: dict(data) for cam, data in sorted(per_camera.items())},
        "per_image": per_image,
        "note": "The eroded high-confidence sky interior is masked; ambiguous skyline pixels are deliberately retained to protect buildings.",
    }
    (args.output / "mask_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: audit[k] for k in ("status", "image_count", "safe_sky_fraction")}, indent=2))


if __name__ == "__main__":
    main()
