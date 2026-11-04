"""Validate the RGB and depth grids of the current L3 batch before OpenMVS."""
import struct
from pathlib import Path

from PIL import Image


def fusion_resolution_args(dmaps, images):
    images = Path(images)
    expected = {(images / f'{frame:06d}_{camera:02d}.jpg').resolve()
                for frame in range(101) for camera in (0, 1, 2, 3, 4, 9, 10)}
    paths = sorted(Path(dmaps).glob('*.dmap'))
    if len(expected) != 707 or len(paths) != 707:
        raise ValueError(f'Fusion requires 707 distinct RGB views and 707 depth maps, got {len(expected)} and {len(paths)}')
    seen = set()
    for path in paths:
        with path.open('rb') as stream:
            header = stream.read(28)
            if len(header) != 28 or header[:2] != b'DR':
                raise ValueError(f'Invalid OpenMVS depth header in {path}')
            width, height, depth_width, depth_height = struct.unpack_from('<4I', header, 4)
            length, = struct.unpack('<H', stream.read(2))
            image = (path.resolve().parent / stream.read(length).decode()).resolve()
        if image not in expected or image in seen:
            raise ValueError(f'{path} references an unexpected or duplicate image: {image}')
        seen.add(image)
        with Image.open(image) as rgb:
            rgb_size = rgb.size
        if rgb_size != (1600, 900) or (width, height, depth_width, depth_height) != (960, 540, 960, 540):
            raise ValueError(f'Fusion grid mismatch: {path.name} RGB {rgb_size}, DMAP {(width,height,depth_width,depth_height)}; expected 1600x900 RGB and 960x540 DMAP.')
    # At level 0, OpenMVS scales the original 1600x900 to exactly 960x540.
    return ['--resolution-level', 0, '--max-resolution', 960]
