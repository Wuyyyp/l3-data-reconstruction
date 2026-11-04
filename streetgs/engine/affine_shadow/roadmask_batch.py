"""Select the audited road supervision for a single 1031 training window."""
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

TOOLS = Path(__file__).resolve().parents[2]/'workflow/code/roadmask_1031'
ROOT = TOOLS.parent.parent
POSE_PY = os.environ.get('POSE_PYTHON',sys.executable)
CAMERAS = (0, 1, 2, 3, 4, 9, 10)


def export_masks(refined, destination, names):
    actual = {p.stem for p in (refined / 'final_mask').glob('*.png')}
    if actual != set(names):
        raise ValueError('Refined PNG keys differ from the 7-view manifest')
    for name in names:
        frame, cam = name.split('_')
        folder = destination / ('cam_' + cam)
        folder.mkdir(exist_ok=True, parents=True)
        with Image.open(refined / 'final_mask' / (name + '.png')) as im:
            pixels = np.asarray(im)
        if pixels.shape != (900, 1600) or not np.isin(pixels, [0, 255]).all():
            raise ValueError('Bad refined mask: ' + name)
        np.save(folder / (frame + '.npy'), pixels != 0, allow_pickle=False)


def prepare_case(config, gpu):
    case = config.parent
    c = json.loads(config.read_text())
    source = Path(c['source_path']).parent
    manifest = (source / 'input_manifest.json').read_bytes()
    views = json.loads(manifest)['views']
    names = [v['name'] for v in views]
    expected = {f'{f:06d}_{cam:02d}' for f in range(101) for cam in CAMERAS}
    if len(names) != 707 or set(names) != expected:
        raise ValueError('Expected one image per camera and frame')
    if case.name == 'shifted' and not (case.parent/'original/config.full_road.mask_audit.json').is_file():
        raise ValueError('Shifted training needs the paired original road-mask audit')
    final = case / 'full_road'
    signature = dict(version='full-road-1031-v1', manifest_sha256=hashlib.sha256(manifest).hexdigest(),
                     source=str(source), sample=case.parent.name, branch=case.name)
    with (case / '.full_road.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not final.exists():
            if (Path(c['model_path']) / 'pipeline_config.json').exists():
                raise RuntimeError('Existing model without its road-mask bundle; refusing to change supervision')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS='1',
                       MKL_NUM_THREADS='1', OMP_NUM_THREADS='4', PYTHONNOUSERSITE='1',
                       PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1')
            with tempfile.TemporaryDirectory(prefix='.road-build-', dir=case) as tmp:
                base = Path(tmp); semantic = base / 'semantic'; refined = base / 'refined'
                subprocess.run([POSE_PY, str(TOOLS/'run_road_semantic_masks.py'), '--source', str(source),
                                '--output', str(semantic), '--weights', os.environ['ROAD_SEGFORMER_WEIGHTS']], env=env, check=True)
                subprocess.run([sys.executable, str(TOOLS/'refine_road_semantic_masks.py'), '--source', str(source),
                                '--semantic', str(semantic), '--output', str(refined)], env=env, check=True)
                bundle = base / 'bundle'; bundle.mkdir()
                export_masks(refined, bundle / 'road_masks', names)
                for folder in ('final_mask', 'masked_rgb', 'previews'):
                    (refined / folder).rename(bundle / folder)
                for stage in (semantic, refined):
                    report = stage / 'report.json'
                    if json.loads(report.read_text())['views'] != 707:
                        raise ValueError('Incomplete mask generation: ' + stage.name)
                    shutil.copy2(report, bundle / (stage.name + '_report.json'))
                (bundle / 'source.json').write_text(json.dumps(signature, indent=2))
                bundle.rename(final)
        if json.loads((final / 'source.json').read_text()) != signature:
            raise ValueError('Road-mask source no longer matches this window')
        saved = Path(c['model_path']) / 'pipeline_config.json'
        if saved.is_file() and json.loads(saved.read_text())['model']['ground']['masks'] != str((final/'road_masks').resolve()):
            raise ValueError('Saved model was trained with different masks')
        sys.path.insert(0, str(TOOLS))
        from prepare_full_road_config import prepare
        audited = case / 'config.full_road.json'
        prepare(config, audited, c['model_path'], final / 'road_masks')
        # Batch consumers read config.json; retain the generator input in config.draft.json.
        if config == case / 'config.json' and config.read_bytes() != audited.read_bytes():
            t = case / '.full_road_config.tmp'; shutil.copyfile(audited, t); t.replace(config)
    return str(audited)


def config_for(path, gpu):
    return path  # The preparation stage already wrote and verified the road masks.
