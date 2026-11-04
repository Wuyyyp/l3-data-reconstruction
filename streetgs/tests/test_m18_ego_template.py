from pathlib import Path
import tempfile, json
import numpy as np
from lib.utils.m18_ego_template import build
R=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(dir=R/'output',prefix='test_ego_builder_') as d:
    D=Path(d); S=D/'source';(S/'ego_pose').mkdir(parents=True);(S/'track').mkdir()
    pc={}
    for i in range(15):
        f=100+i*2
        pc[f]=np.array([[1.02,0,1.22],[2.02,0,1.22],[3+i*.15,0,1.22]],dtype='f4')
        pose=np.eye(4);pose[0,3]=i*2;np.savetxt(S/'ego_pose'/f'{f:06d}.txt',pose)
    np.savez(S/'pointcloud.npz',pointcloud=pc)
    (S/'track/track_info.txt').write_text('header\n'+'\n'.join(f'{f} 1 car 0 0.3 0.3 0.3 2.02 0 1.22 0' for f in pc))
    p=build(S,D/'first',previews=False)
    t=np.load(p);a=json.loads((p.parent/'audit.json').read_text())
    assert a['candidate_points']==15 and len(t['active_cells'])==1 and a['protected_box_points']==15,a
    assert bool(t['cell_only']) and bool(t['protect_track_boxes'])
    try: build(S,D/'first',previews=False)
    except FileExistsError: pass
    else: raise AssertionError('Overwrote existing template')
    p2=build(S,D/'second',previews=False); assert p2!=p
    for f in pc: np.savetxt(S/'ego_pose'/f'{f:06d}.txt',np.eye(4))
    empty=build(S,D/'stationary',previews=False)
    assert len(np.load(empty)['active_cells'])==0
    from scipy.spatial import cKDTree
    cKDTree(np.load(empty)['seed_ego'])
print('PASS: fresh outputs, nonzero/noncontiguous frame IDs, box protection, temporal support, stationary scene retains all, empty template supported')
