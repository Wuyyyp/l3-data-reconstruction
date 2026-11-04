#!/usr/bin/env python3
"""D sparse-point scale calibration with held-out validation and MoGe support."""
from pathlib import Path
import argparse,json,struct,sys,shutil,os,time
import numpy as np,cv2
cv2.setNumThreads(1)
from finalize_dvp_scene import make_normals,write_camera
def read(p):
 with p.open('rb') as f:
  _,h,w,t=struct.unpack('<4i',f.read(16));a=np.fromfile(f,dtype='<f4')
 return a.reshape((h,w) if t==5 else (h,w,3))
def write(p,a):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(struct.pack('<4i',1,a.shape[0],a.shape[1],5 if a.ndim==2 else 21)+np.asarray(a,dtype='<f4').tobytes())
def medmad(a):
 a=np.asarray(a);m=float(np.median(a));return m,float(1.4826*np.median(np.abs(a-m)))
def summary(a):return {k:float(v) for k,v in zip(['min','p10','median','p90','max'],np.percentile(a,[0,10,50,90,100]))}
def main():
 global SRC,PREP,ROOT,DST
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--result-root',type=Path,required=True);args=ap.parse_args()
 SRC=args.source;PREP=args.prepared;ROOT=args.result_root;DST=ROOT/'09_dvp/scene'
 if SRC.resolve()==(ROOT/'09_dvp').resolve():raise ValueError('Source stage-1 scene must be kept separately')
 if (ROOT/'calibration_report.json').exists():raise RuntimeError('Experiment already prepared')
 ROOT.mkdir(exist_ok=True);DST.mkdir(parents=True,exist_ok=True)
 for name in ['cams','metadata','metric_prior','metric_prior_unbounded','prior_support']:(DST/name).mkdir(exist_ok=True)
 for name in ['images','blocks']:
  p=DST/name
  if not p.exists():p.symlink_to(SRC/'scene'/name,target_is_directory=True)
 shutil.copy2(SRC/'scene/pair.txt',DST/'pair.txt')
 points={}
 for line in (PREP/'07_sparse/triangulated_text/points3D.txt').open():
  if line.startswith('#') or not line.strip():continue
  a=line.split();pid=int(a[0]);err=float(a[7]);track=(len(a)-8)//2
  if err<=2 and track>=3:points[pid]=np.array(list(map(float,a[1:4])))
 lines=[l.strip() for l in (PREP/'07_sparse/triangulated_text/images.txt').open() if not l.startswith('#')];observations={}
 for i in range(0,len(lines),2):
  label=Path(lines[i].split()[9]).stem;a=np.array(lines[i+1].split(),dtype=float).reshape(-1,3);observations[label]=a
 records=[];train_by_cam={};validation_by_cam={}
 for mp in sorted((SRC/'scene/metadata').glob('*.json')):
  m=json.loads(mp.read_text());id=mp.stem;label=m['source_label'];cam=label[-2:];f=int(label[:6]);l=(SRC/'scene/cams'/f'{id}_cam.txt').read_text().splitlines();E=np.array([[float(x) for x in t.split()] for t in l[1:5]]);K=np.array([[float(x) for x in t.split()] for t in l[7:10]])
  oldscale=m['moge_scale_multiplier'];raw=read(SRC/'scene/metric_prior_unbounded'/f'{id}.dmb')/oldscale;keep=cv2.imread(str(SRC/'scene/blocks'/f'mask_{int(id)}.png'),0)>=128;raw[~keep]=0
  obs=observations[label];obs=obs[np.array([int(a[2]) in points for a in obs])]
  ratios=[];validation=[];cells=set()
  if len(obs):
   xyz=np.array([points[int(a[2])] for a in obs]);q=xyz@E[:3,:3].T+E[:3,3];uv=q@K.T;uv=uv[:,:2]/uv[:,2:]
   actual=obs[:,:2]*m['resize_scale']+np.array(m['padding_xy']);good=(q[:,2]>0)&(np.linalg.norm(uv-actual,axis=1)<3)&(uv[:,0]>=2)&(uv[:,0]<958)&(uv[:,1]>=2)&(uv[:,1]<638)
   for j in np.where(good)[0]:
    x,y=np.rint(uv[j]).astype(int);patch=raw[y-1:y+2,x-1:x+2];vals=patch[patch>0]
    if len(vals)<9 or np.percentile(vals,90)/np.percentile(vals,10)>1.15:continue
    ratio=float(np.log(q[j,2]/float(raw[y,x])));pid=int(obs[j,2])
    if pid%5==0:validation.append(ratio)
    else:ratios.append(ratio);cells.add((x//120,y//80))
  train_by_cam.setdefault(cam,[]).extend(ratios);validation_by_cam.setdefault(cam,[]).extend([(v,float(np.log(oldscale))) for v in validation])
  records.append(dict(id=id,label=label,cam=cam,frame=f,meta=m,E=E,K=K,raw=raw,keep=keep,train=ratios,validation=validation,cells=len(cells),oldscale=oldscale))
 print('SPARSE correspondences',len(points),'frames',len(records),flush=True)
 camfit={}
 for cam,ratios in train_by_cam.items():
  if len(ratios)<100:
   camfit[cam]=None
   continue
  med,mad=medmad(ratios);inliers=np.array(ratios)[np.abs(np.array(ratios)-med)<=max(3*mad,np.log(1.2))];med,mad=medmad(inliers);camfit[cam]=(med,mad)
 for r in records:
  if camfit[r['cam']] is None:
   r['fit']=float(np.log(r['oldscale']))
   continue
  med,mad=camfit[r['cam']];a=np.array(r['train']);good=a[np.abs(a-med)<=max(3*mad,np.log(1.3))]
  fit=float(np.median(good)) if len(good)>=30 and r['cells']>=6 else med
  r['fit']=float(np.clip(fit,med-max(3*mad,np.log(1.3)),med+max(3*mad,np.log(1.3))))
 # A camera-level validation gate prevents a sparse correction that clearly worsens alignment.
 camreport={}
 for cam in camfit:
  if camfit[cam] is None:
   camreport[cam]={'validation_count':0,'use_sparse_calibration':False,'training_count':len(train_by_cam[cam]),'fallback':'LiDAR scale; too few BA sparse correspondences'}
   for r in records:
    if r['cam']==cam:r['scale']=r['oldscale'];r['depth']=r.pop('raw')*r['scale'];r['threshold']=float(np.log(1.5))
   continue
  rs=[r for r in records if r['cam']==cam];old=[];new=[]
  for r in rs:
   old.extend(abs(np.array(r['validation'])-np.log(r['oldscale'])));new.extend(abs(np.array(r['validation'])-r['fit']))
  use=len(new)>=30 and np.median(new)<=max(np.median(old)*1.2,np.log(1.05))
  camreport[cam]={'validation_count':len(new),'old_median_abs_log_error':float(np.median(old)) if old else None,'new_median_abs_log_error':float(np.median(new)) if new else None,'use_sparse_calibration':bool(use),'training_count':len(train_by_cam[cam])}
  for r in rs:
   r['scale']=float(np.exp(r['fit'])) if use else r['oldscale'];r['depth']=r.pop('raw')*r['scale'];r['threshold']=float(max(np.log(1.5),3*np.median(new))) if new else float(np.log(1.5))
 print('CALIBRATION',json.dumps(camreport),flush=True)
 bylabel={r['label']:r for r in records};ranges={};audit=[]
 yy,xx=np.mgrid[0:640:4,0:960:4];pixels=np.column_stack([xx.ravel(),yy.ravel(),np.ones(xx.size)])
 for num,r in enumerate(records):
  dep=r['depth'];ds=dep[yy,xx].ravel();q=(pixels@np.linalg.inv(r['K']).T)*ds[:,None];world=(q-r['E'][:3,3])@r['E'][:3,:3];counts=np.zeros(len(ds),np.uint8)
  for shift in [-6,-3,3,6]:
   neighbor=bylabel.get(f"{r['frame']+shift:06d}_{r['cam']}")
   if neighbor is None:continue
   q2=world@neighbor['E'][:3,:3].T+neighbor['E'][:3,3];uv=q2@neighbor['K'].T;uv=np.rint(uv[:,:2]/np.maximum(uv[:,2:],1e-8)).astype(int)
   valid=(ds>0)&(q2[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<640);inds=np.where(valid)[0];x,y=uv[inds].T;other=neighbor['depth'][y,x]
   counts[inds]+=((other>0)&(np.abs(other-q2[inds,2])<=.2*q2[inds,2])).astype(np.uint8)
  support=cv2.resize(counts.reshape(yy.shape),(960,640),interpolation=cv2.INTER_NEAREST);support[~r['keep']]=0
  valid=dep>0;supported=dep[(support>=2)&valid];allvalues=dep[valid]
  oldmin,oldmax=r['meta']['dvp_depth_range_m']
  lo=min(oldmin,.85*float(np.percentile(supported,.5))) if len(supported)>=1000 else oldmin
  hi=max(oldmax,1.2*float(np.percentile(supported,99.5))) if len(supported)>=1000 else oldmax
  # Preserve the unbounded prior; out-of-range samples are separately accounted for.
  ranges[r['label']]={'depth_min_m':lo,'depth_max_m':hi,'source':'sparse_base_plus_cross_view_supported_calibrated_moge','sparse_points':len(r['train'])+len(r['validation'])}
  write_camera(DST/'cams'/f"{r['id']}_cam.txt",r['E'],r['K'],lo,hi)
  normals=make_normals(dep,r['K'],r['E'])
  for folder in ['metric_prior','metric_prior_unbounded']:
   write(DST/folder/f"{r['id']}.dmb",dep);write(DST/folder/f"{r['id']}_normal.dmb",normals)
  cv2.imwrite(str(DST/'prior_support'/f"{r['id']}.png"),support)
  meta=r['meta'].copy();meta.update(moge_scale_multiplier=r['scale'],previous_moge_scale_multiplier=r['oldscale'],scale_source='sparse_robust_fit' if camreport[r['cam']]['use_sparse_calibration'] else 'previous_lidar_validation_fallback',dvp_depth_range_m=[lo,hi],depth_range_source=ranges[r['label']]['source'],prior_log_deviation_threshold=r['threshold'])
  (DST/'metadata'/f"{r['id']}.json").write_text(json.dumps(meta,indent=2))
  audit.append({'label':r['label'],'old_scale':r['oldscale'],'new_scale':r['scale'],'old_max':oldmax,'new_max':hi,'supported_prior_pixels':len(supported),'out_of_range_prior_pixels':int(((dep>hi)|((dep>0)&(dep<lo))).sum()),'training_pairs':len(r['train']),'validation_pairs':len(r['validation'])})
  if (num+1)%50==0:print('PREPARED',num+1,flush=True)
 manifest=json.loads((SRC/'manifest.json').read_text());manifest['images']=[json.loads((DST/'metadata'/f"{r['id']}.json").read_text()) for r in records];manifest['experiment']='sparse_moge_20260924';(ROOT/'09_dvp/manifest.json').write_text(json.dumps(manifest,indent=2))
 range_report={'ranges':ranges,'depth_range_mode':'sparse + calibrated MoGe with multi-view support','depth_min_m_summary':summary([v['depth_min_m'] for v in ranges.values()]),'depth_max_m_summary':summary([v['depth_max_m'] for v in ranges.values()])};(ROOT/'sparse_depth_ranges_openmvs.json').write_text(json.dumps(range_report,indent=2))
 report={'cameras':camreport,'ranges':range_report['depth_max_m_summary'],'rows':audit,'validation_note':'20 percent of point IDs excluded from scale fitting, used for camera-level correction acceptance; not an independent final accuracy test'};(ROOT/'calibration_report.json').write_text(json.dumps(report,indent=2));print('PREPARE COMPLETE',json.dumps(report['ranges']),flush=True)
if __name__=='__main__':main()
