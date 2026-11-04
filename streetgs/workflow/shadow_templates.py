"""Build mesh-only local shadow splats for this window in metres."""
import json, sys, tempfile
from pathlib import Path
import numpy as np

def template(actor):
    radii = np.asarray(actor['size'][:2], dtype=float) / 2 + 1
    assert np.isfinite(radii).all() and (radii > 1).all()
    x,y = (np.arange(-r, r + .001, .2) for r in radii)
    xy = np.stack(np.meshgrid(x,y,indexing='ij'), -1).reshape(-1,2)
    xyz = np.zeros((len(xy),3),np.float32)
    xyz[:,:2] = xy
    normals = np.zeros_like(xyz)
    normals[:,2] = 1
    rotation = np.zeros((len(xy),4),np.float32)
    rotation[:,0] = 1
    return dict(xyz=xyz,normals=normals,rotation=rotation,rgb=np.zeros_like(xyz))

def build(case):
    actors = json.loads((case/'data/actors.json').read_text())
    target = case/'shadow_templates'
    expected = {str(a['id']) for a in actors if a.get('source') != 'native_lidar'}
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=case) as temp:
            stage = Path(temp)/'templates'
            stage.mkdir()
            for a in actors:
                if str(a['id']) in expected:
                    np.savez(stage/(str(a['id'])+'.npz'), **template(a))
            stage.rename(target)
    assert {p.stem for p in target.glob('*.npz')} == expected
    print('shadow templates',len(expected),flush=True)

if __name__=='__main__':
    build(Path(sys.argv[1]))
