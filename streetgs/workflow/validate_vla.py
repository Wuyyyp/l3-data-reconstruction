from pathlib import Path
import sys,json,cv2,numpy as np
from prepare_pose_input import U,CAMERAS
from l3_wys_canvas import VERSION,geometry,render
w=Path(sys.argv[1]);orig=w/'original';src=w/'pose/run/deliverables'
with (src/'result_interpolation_optimized.pkl').open('rb') as f:infos=U(f).load()['infos']
vs=json.loads((orig/'data/input_manifest.json').read_text())['views']
rows=json.loads((orig/'full/eval_checkpoint_030000/results.json').read_text())['rows'];by_key={r['image']:r for r in rows}
assert len(vs)==len(by_key)==len(rows)==707
for v in vs:
    f,c=v['frame'],v['cam'];name=v['name'];assert name==f'{f:06d}_{c:02d}'
    camera=next(k for k,id_ in CAMERAS.items() if id_==c);cam=infos[f]['sensors']['cams'][camera]
    np.testing.assert_allclose(v['K'],cam['cam_intrinsic'],atol=1e-6,rtol=0)
    valid=cv2.imread(str(orig/'data/converted/valid_mask'/(name+'.png')),0)
    assert valid.shape==(900,1600) and 0<by_key[name]['all_pixels']<=np.count_nonzero(valid) and np.isfinite(by_key[name]['affine_all_psnr'])
    g=cam['source_geometry'];assert g==geometry(c, g['source_wh'])
    np.testing.assert_allclose(v['K'],np.array(g['A'])@np.array(cam['original_cam_intrinsic']),rtol=0,atol=1e-6)
    if c<9:assert valid[:,:125].any() and valid[:,1475:].any()
    if f in (0,50,100):
        original=cv2.imread(cam['original_data_path']);ok,enc=cv2.imencode('.jpg',render(original,cam['source_geometry']))
        assert ok and enc.tobytes()==Path(cam['data_path']).read_bytes()
assert json.loads((orig/'config.full_road.mask_audit.json').read_text())['views']==707
(w/'canvas_acceptance.json').write_text(json.dumps(dict(geometry_version=VERSION,original_valid_masks=707,pixel_replays=21,full_road_masks=707)))

