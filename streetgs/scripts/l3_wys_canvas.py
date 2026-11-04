"""Full-frame L3 conversion used by wys-test. Inputs are undistorted JPEGs."""
from pathlib import Path

import cv2
import numpy as np

VERSION = 'l3-wys-full-frame-v1'
SIZE = (1600, 900)
TEMPLATES = Path('/data/l3_data_test/street_gaussians/configs/preprocessing/ego_mask_l3_recon_20261030')


def geometry(camera, source_wh):
    w, h = map(int, source_wh)
    if min(w, h) <= 0:
        raise ValueError(f'Invalid dimensions for camera {camera}: {source_wh}')
    return dict(version=VERSION, source_wh=[w, h], canvas_wh=list(SIZE),
                content_xywh=[0, 0, *SIZE], projection='PINHOLE',
                A=np.diag([SIZE[0] / w, SIZE[1] / h, 1.]).tolist())


def render(image, g, mask=False):
    if image.shape[:2] != tuple(reversed(g['source_wh'])) or g['version'] != VERSION:
        raise ValueError('Image dimensions or canvas version do not match')
    return cv2.resize(image, SIZE, interpolation=cv2.INTER_NEAREST_EXACT if mask else cv2.INTER_AREA)


def write_masks(folder, camera, g):
    template = TEMPLATES / f'000000_{camera:02d}.png'
    mask = cv2.imread(str(template), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(template)
    if np.any(mask) and mask.shape[1] * g['source_wh'][1] != mask.shape[0] * g['source_wh'][0]:
        raise ValueError(f'Mask aspect ratio mismatch: {template}')
    folder.mkdir(exist_ok=True)
    if not cv2.imwrite(str(folder / template.name), cv2.resize(mask, SIZE, interpolation=cv2.INTER_NEAREST_EXACT)):
        raise OSError(folder / template.name)

