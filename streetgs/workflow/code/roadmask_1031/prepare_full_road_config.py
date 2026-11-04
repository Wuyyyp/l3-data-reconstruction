"""Prepare a new 4D run using explicit dense road masks; never fall back to LiDAR masks."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np


def prepare(source, dest, output, mask_dir):
    source, dest, output, mask_dir = map(Path, (source, dest, output, mask_dir))
    if mask_dir.name != 'road_masks':
        raise ValueError('Use an explicitly prepared road_masks directory, not ground_masks/closed_masks')
    c = json.loads(source.read_text())
    manifest = json.loads((Path(c['source_path']).parent / 'input_manifest.json').read_text())
    views = [v for v in manifest['views'] if v['cam'] in c['data']['cameras'] and v['frame'] in c['data']['frame_ids']]
    rows = []
    for v in views:
        frame, cam = v['name'].split('_')
        p = mask_dir / ('cam_' + cam) / (frame + '.npy')
        m = np.load(p, allow_pickle=False)
        if m.shape != (900, 1600) or not np.isin(m, [0, 1, 255]).all():
            raise ValueError('Expected a binary 900x1600 road mask: ' + str(p))
        rows.append(dict(image=v['name'], pixels=int(np.count_nonzero(m)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if len(rows) != 707 or len({x['image'] for x in rows}) != 707 or not sum(x['pixels'] for x in rows):
        raise ValueError('Expected 707 unique views with nonempty road supervision')
    c['model']['ground']['masks'] = str(mask_dir.resolve())
    c['model_path'] = str(output)
    if dest.exists() and json.loads(dest.read_text()) != c:
        raise ValueError('Existing configuration differs; use a fresh RUN_TAG')
    audit_path = dest.with_suffix('.mask_audit.json')
    audit = dict(mask_source=str(mask_dir.resolve()), views=len(rows), rows=rows,
                 policy='road & guidance_valid & ~dynamic_bounds; warmup also excludes deferred actors',
                 rgb_not_intersected_with_lidar=True, depth_only_where_valid_depth=True)
    if audit_path.exists() and json.loads(audit_path.read_text()) != audit:
        raise ValueError('Road mask contents changed since configuration creation')
    dest.write_text(json.dumps(c, indent=2))
    audit_path.write_text(json.dumps(audit, indent=2))
    print(json.dumps(dict(config=str(dest), road_masks=str(mask_dir), views=len(rows), pixels=sum(x['pixels'] for x in rows))))


if __name__ == '__main__':
    prepare(*sys.argv[1:])
