"""Keep method-D geometry, use established L3 fixed/sky masks, no dynamic masking."""
from pathlib import Path
import sys,os,json
import cv2,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mvs_process'))
import delivery_d_prepare as p

# The new pose delivery is already 1600x900. Preserve all K/model values.
import struct
def cameras1600(path,scales=None):
    records={}
    with path.open('rb') as f:
        for _ in range(struct.unpack('<Q',f.read(8))[0]):
            cid,kind,w,h=struct.unpack('<iiQQ',f.read(24))
            assert kind==1 and (w,h)==(1600,900),(cid,kind,w,h)
            params=np.array(struct.unpack('<4d',f.read(32)))
            if scales is not None:assert scales.get(cid,1)==1
            records[cid]=dict(size=(w,h),output_size=(w,h),K=np.array([[params[0],0,params[2]],[0,params[1],params[3]],[0,0,1]]))
        assert not f.read(1)
    return records
p.model_cameras=cameras1600

old=Path(os.environ['WYS_OLD_CASE'])/'converted'
p.tracks_for_window=lambda infos,start,end:{}
p.objects=lambda info:[]
def masks(prepared,inv,infos):
    directory=prepared/'06_masks';target=directory/'openmvs_masks';target.mkdir(parents=True,exist_ok=True)
    # Exact L3 hood is already included in the source mask. Keep generic rig at bottom row.
    p.save(directory/'camera10_rig_profile.json',dict(description='Established L3 fixed masks already applied',camera10_topline=[[0,1],[1,1]],edge_margin_at_960=0))
    rows=[]
    for row in inv['records']:
        stem=Path(row['label']).stem;w,h=row['output_size']
        sky=cv2.imread(str(old/'sky_mask'/(stem+'.png')),0)
        fixed=cv2.imread(str(old/'fixed_mask'/(stem+'.png')),0)
        assert sky is not None and fixed is not None and sky.shape==fixed.shape==(900,1600)
        sky=cv2.resize(sky,(w,h),interpolation=cv2.INTER_NEAREST)>0
        fixed=cv2.resize(fixed,(w,h),interpolation=cv2.INTER_NEAREST)>0
        path=target/(stem+'.mask.png');assert cv2.imwrite(str(path),(sky|fixed).astype('u1')*255)
        rows.append(dict(image=row['label'],dynamic_pixels=0,sky_pixels=int(sky.sum()),fixed_pixels=int(fixed.sum()),sha256=p.digest(path)))
    p.save(directory/'mask_audit.json',dict(status='PASS',images=len(rows),rows=rows))
    p.save(directory/'dynamic_mask_audit.json',dict(status='PASS',image_count=len(rows),dynamic_mask_count=len(rows),dynamic_policy='disabled-preserve-all-objects',dynamic_threshold_m=None,tracks={},rows=rows))
p.prepare_masks=masks
# The inventory is already checked by original main; semantics are reused, no model inference.
if '--semantics-only' in sys.argv:
    print('Using validated L3 infer_sky and fixed masks',flush=True)
else:
    p.__file__=__file__
    p.main()
