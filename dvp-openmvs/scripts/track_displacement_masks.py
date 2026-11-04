"""Mask tracks with >3 m map XY displacement within the reconstruction window."""
import argparse, hashlib, json, sys
from pathlib import Path
import cv2
import numpy as np
REPO = Path(__file__).resolve().parents[1]
DATA = Path('/data/l3data-reconstruction-bingxing')
sys.path[:0] = [str(REPO/'data_preprocess'), str(DATA), str(DATA/'Dongfeng_Pose-main/src')]
from M18proc.dynamic_policy import object_id, select_tracks, VERSION
from extract_original_restored_metadata import NumpyCompatUnpickler
from dongfeng_registration.dynamic_object_compensation import box_corners_ego, BOX_EDGE_INDICES
def objects(info):
    # M18 objects use L,W,H; its gt_boxes use a different size convention.
    if info.get('objects'):
        return info['objects']
    boxes = info.get('gt_boxes', [])
    ids = info.get('ID', info.get('track_id', []))
    velocities = info.get('vels', [])
    names = info.get('gt_names', [])
    return [dict(id=ids[i] if i < len(ids) else None,
                 velocity=velocities[i] if i < len(velocities) else None,
                 type=names[i] if i < len(names) else '',
                 location=list(box[:3]), size=list(box[3:6]), rotation=[0,0,box[6]])
            for i, box in enumerate(boxes)]

def projected_polygon(corners, transform, K, width, height):
    xyz = corners @ transform[:3,:3].T + transform[:3,3]
    assert np.isfinite(xyz).all()
    # Clip the 12 cuboid edges to the camera's near plane before projecting.
    near = 0.1
    visible = [p for p in xyz if p[2] >= near]
    for a, b in BOX_EDGE_INDICES:
        p, q = xyz[a], xyz[b]
        if (p[2] < near) != (q[2] < near):
            visible.append(p + (q-p) * ((near-p[2])/(q[2]-p[2])))
    if len(visible) < 3:
        return None
    uvz = np.asarray(visible) @ K.T
    uv = (uvz[:,:2]/uvz[:,2:]).astype(np.float32)
    hull = cv2.convexHull(uv)
    screen = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], np.float32)
    area, polygon = cv2.intersectConvexConvex(hull, screen)
    return None if area <= 0 or polygon is None else np.rint(polygon).astype(np.int32)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory',type=Path,required=True)
    ap.add_argument('--threshold-m',type=float,default=3.0)
    ap.add_argument('--map-pose-field',default='ego2global_transformation_matrix')
    ap.add_argument('--baseline-masks',type=Path,required=True,help='Sky/ego masks before any old dynamic union')
    ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); cv2.setNumThreads(1)
    if args.output.exists(): raise RuntimeError('Choose a new mask directory; historical masks are preserved')
    inventory=json.loads(args.inventory.read_text())
    pkl=Path(inventory['source_restored_clip'])/'result_interpolation_optimized.pkl'
    with pkl.open('rb') as f: infos=NumpyCompatUnpickler(f).load()['infos']
    frames=sorted({r['original_frame'] for r in inventory['records']})
    selected,policy=select_tracks(((f,objects(infos[f]),infos[f].get(args.map_pose_field)) for f in frames),args.threshold_m)
    policy.update(source_pkl=str(pkl),source_pkl_sha256=hashlib.sha256(pkl.read_bytes()).hexdigest(),
                  map_pose_field=args.map_pose_field,inventory=str(args.inventory))
    dynamic_dir=args.output/'dynamic_mask'; dynamic_dir.mkdir(parents=True)
    merged_dir=args.output/'openmvs_masks'; merged_dir.mkdir()
    rows=[]; kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    for rec in inventory['records']:
        label=Path(rec['label']).stem; frame=rec['original_frame']
        base_path=args.baseline_masks/(label+'.mask.png')
        base=cv2.imread(str(base_path),0)
        if base is None: raise RuntimeError('Missing sky/ego baseline: '+str(base_path))
        spec=inventory['calibrations'][rec['camera']]; h,w=base.shape
        assert (w,h)==(spec['width'],spec['height'])
        K=np.asarray(spec['K'])
        # Preserve delivered image-space annotation geometry; BA must not shift masks.
        ego=np.asarray(inventory['clouds'][rec['frame_index']]['ego_to_world']['after'])
        transform=np.linalg.inv(np.asarray(rec['poses']['after']))@ego
        dynamic=np.zeros_like(base); projected=[]
        for obj in objects(infos[frame]):
            key=object_id(obj)
            if key not in selected: continue
            corners=box_corners_ego(obj)
            if not np.isfinite(corners).all(): continue
            polygon=projected_polygon(corners,transform,K,w,h)
            if polygon is None: continue
            cv2.fillConvexPoly(dynamic,polygon,255)
            projected.append(dict(track_id=key,max_displacement_m=policy['tracks'][key]['max_displacement_m']))
        dynamic=cv2.dilate(dynamic,kernel)
        base=np.where(base>=128,255,0).astype(np.uint8); merged=cv2.bitwise_or(base,dynamic)
        assert cv2.imwrite(str(dynamic_dir/(label+'.png')),dynamic)
        assert cv2.imwrite(str(merged_dir/(label+'.mask.png')),merged)
        assert np.all(merged[base>0]==255)
        rows.append(dict(label=label,original_frame=frame,baseline_pixels=int((base>0).sum()),
                         dynamic_pixels=int((dynamic>0).sum()),added_pixels=int(((dynamic>0)&(base==0)).sum()),
                         ignored_pixels=int((merged>0).sum()),projected_tracks=projected))
        if rec['frame_index'] in (0,50,100):
            image=cv2.imread(str(args.images/rec['label'])); assert image is not None and image.shape[:2]==base.shape
            overlay=image.copy(); overlay[dynamic>0]=(0,0,255)
            preview=cv2.addWeighted(image,.6,overlay,.4,0)
            cv2.imwrite(str(args.output/(label+'_dynamic_preview.jpg')),cv2.resize(preview,(960,round(h*960/w))))
    (args.output/'dynamic_mask_policy.json').write_text(json.dumps(policy,indent=2,allow_nan=False)+'\n')
    report=dict(status='PASS',dynamic_policy=policy['version'],dynamic_threshold_m=args.threshold_m,image_count=len(rows),dynamic_mask_count=len(rows),
                dynamic_dilation_px=5,baseline_masks=str(args.baseline_masks),selected_tracks=len(selected),
                tracks=len(policy['tracks']),dynamic_pixels=sum(r['dynamic_pixels'] for r in rows),
                added_pixels=sum(r['added_pixels'] for r in rows),rows=rows)
    (args.output/'dynamic_mask_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2),flush=True)
if __name__=='__main__': main()
