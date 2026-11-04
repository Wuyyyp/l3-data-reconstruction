"""Capture the pilot's own forward pass on the seven cameras of frame 100."""
import json
import os
from pathlib import Path
import time

from PIL import Image

SAMPLE = 'sample_00324_clip_M18-2_07_20251227094237_DF_f187_287_right'


def save_view(root, iteration, camera, prediction, target, loss):
    # One complete shuffle of 707 views per set. Keep a fresh set every 2121 steps.
    epoch = (iteration - 1) // 707
    if epoch < 8 or (epoch - 8) % 3 or not camera.startswith('000100_'):
        return
    pid = os.getpid()
    tick = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    directory = root / 'release' / 'pilot_views' / f'{pid}-{tick}' / f'{epoch:03d}'
    directory.mkdir(parents=True, exist_ok=True)
    for role, tensor in [('render', prediction), ('gt', target)]:
        rgb = tensor.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        path = directory / f'{camera}.{role}.png'
        temp = path.with_suffix('.tmp')
        Image.fromarray(rgb).save(temp, format='PNG', compress_level=1)
        temp.replace(path)
    record = dict(camera=camera, sample=SAMPLE, branch='original/full', iteration=iteration,
                  pid=pid, start_tick=tick, node=os.uname().nodename, gpu=os.environ['CUDA_VISIBLE_DEVICES'],
                  time=time.time(), loss=float(loss.detach()))
    path = directory / f'{camera}.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record))
    temp.replace(path)
