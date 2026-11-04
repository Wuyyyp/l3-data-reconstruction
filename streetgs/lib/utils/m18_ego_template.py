"""Build a fresh, scene-derived candidate ego-return filter; not a vehicle CAD model."""
from pathlib import Path
import json, time, math
import numpy as np

def cell_keys(p):
    v = np.floor(p / .1).astype('i8')
    return (v[:, 0]+100)*1000000+(v[:, 1]+100)*1000+v[:, 2]+100

def build(source, output, extrinsics='camera_to_ego', min_fraction=.69, min_motion=10., radius=5., z_min=.8, z_max=2.3, previews=True):
    S, O = Path(source).resolve(), Path(output).resolve()
    if not (0 < min_fraction <= 1 and min_motion > 0 and 0 < radius < 9 and -9 < z_min < z_max < 9):
        raise ValueError('Invalid candidate thresholds/ROI')
    if extrinsics not in ('ego_to_camera', 'camera_to_ego'): raise ValueError('Unknown extrinsics')
    start = time.time()
    # Never overwrite/reuse a prior run template.
    O.mkdir(parents=True, exist_ok=False)
    with np.load(S/'pointcloud.npz', allow_pickle=True) as data:
        pc = data['pointcloud'].item()
    frames = sorted(pc)
    if not frames: raise ValueError('No pointcloud frames')
    poses = {f: np.loadtxt(S/'ego_pose'/f'{int(f):06d}.txt')[:3,3] for f in frames}
    if not np.isfinite(list(poses.values())).all(): raise ValueError('Non-finite ego poses')
    boxes = {}
    for line in (S/'track/track_info.txt').read_text().splitlines()[1:]:
        if line.strip():
            row = line.split(); boxes.setdefault(int(row[0]), []).append(row)
    near, support, protected_count = {}, {}, 0
    for f in frames:
        p = pc[f]
        q = p[np.isfinite(p).all(1)&(np.linalg.norm(p[:,:2],axis=1)<radius)&(p[:,2]>z_min)&(p[:,2]<z_max)]
        protected = np.zeros(len(q), bool)
        for row in boxes.get(int(f), []):
            h,w,l = map(float,row[4:7]); yaw=float(row[10]); c,s=np.cos(yaw),np.sin(yaw)
            local=(q-np.array(row[7:10],float))@np.array([[c,-s,0],[s,c,0],[0,0,1]])
            protected |= (np.abs(local)<=np.array([l,w,h])/2+.15).all(1)
        protected_count += int(protected.sum()); q=q[~protected]; near[f]=q
        for k in np.unique(cell_keys(q)): support.setdefault(int(k),[]).append(f)
    # Ordinal thirds support arbitrary/non-zero frame identifiers.
    thirds={f:min(i*3//len(frames),2) for i,f in enumerate(frames)}
    required=max(10,math.ceil(min_fraction*len(frames)))
    active=np.array(sorted(k for k,fs in support.items() if len(fs)>=required and len({thirds[f] for f in fs})==3 and np.linalg.norm(np.ptp([poses[f] for f in fs],axis=0))>=min_motion),dtype='i8')
    selected={f:q[np.isin(cell_keys(q),active)] for f,q in near.items()}
    allp=np.concatenate(list(selected.values()))
    _,ix=np.unique(np.floor(allp/.02).astype('i8'),axis=0,return_index=True); seed=allp[ix]
    template=O/'candidate_template.npz'
    np.savez(template,seed_ego=seed,active_cells=active,cell_only=True,protect_track_boxes=True,roi_radius=radius,roi_z_min=z_min,roi_z_max=z_max)
    with (O/'candidate_ego_points.ply').open('wb') as out:
        out.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n'%len(seed)).encode());seed.astype('<f4').tofile(out)
    projections=[]
    if previews:
        import cv2
        chosen={frames[min(int(len(frames)*x),len(frames)-1)] for x in (.2,.5,.8)}
        for im_path in sorted((S/'images').glob('*.jpg')):
            f,cam=map(int,im_path.stem.split('_'))
            if f not in chosen: continue
            im=cv2.imread(str(im_path))
            if im is None: raise ValueError('Unreadable preview '+str(im_path))
            E=np.loadtxt(S/'extrinsics'/(im_path.stem+'.txt'));K=np.loadtxt(S/'intrinsics'/(im_path.stem+'.txt'))
            if extrinsics=='camera_to_ego': E=np.linalg.inv(E)
            q=selected[f]@E[:3,:3].T+E[:3,3]; uv=q@K.T
            with np.errstate(divide='ignore',invalid='ignore'): uv=uv[:,:2]/uv[:,2:]
            ok=(q[:,2]>.1)&np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<im.shape[1])&(uv[:,1]>=0)&(uv[:,1]<im.shape[0])
            for x,y in uv[ok].astype(int): cv2.circle(im,(x,y),2,(0,0,255),-1)
            cv2.putText(im,f'{im_path.stem} candidate: {int(ok.sum())}',(20,35),0,.8,(0,0,255),2)
            cv2.imwrite(str(O/(im_path.stem+'.jpg')),cv2.resize(im,(960,int(im.shape[0]*960/im.shape[1]))))
            projections.append(dict(image=im_path.stem,projected=int(ok.sum())))
    audit=dict(source=str(S),template=str(template),frames=len(frames),required_frames=required,min_fraction=min_fraction,min_motion_m=min_motion,roi=dict(radius=radius,z_min=z_min,z_max=z_max),candidate_points=len(allp),seed_points=len(seed),active_cells=len(active),protected_box_points=protected_count,per_frame=[dict(frame=int(f),candidates=len(p)) for f,p in selected.items()],projections=projections,seconds=time.time()-start,status='candidates_found' if len(active) else 'insufficient_evidence_no_deletion',caveat='Empirical candidates, not proven echoes. ROI assumes ego coordinates in metres with +Z up. Unlabelled persistent objects may be included; inspect projections. Empty template retains all points.')
    (O/'audit.json').write_text(json.dumps(audit,indent=2))
    return template
