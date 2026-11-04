from pathlib import Path
import argparse,sys,json,os,struct
import numpy as np,cv2
cv2.setNumThreads(1)
from delivery_d_calibrate import read,write
ap=argparse.ArgumentParser();ap.add_argument('--result-root',type=Path,required=True);args=ap.parse_args();ROOT=args.result_root
s=ROOT/'09_dvp/scene';records={};lines=(s/'pair20.txt').read_text().splitlines();pairs={}
for i in range(1,len(lines),2):pairs[int(lines[i])]=[int(v) for v in lines[i+1].split()[1::2]]
for p in (s/'APD').glob('*/depths.dmb'):
 id=int(p.parent.name[-8:]);meta=json.loads((s/'metadata'/f'{id:08d}.json').read_text());l=(s/'cams'/f'{id:08d}_cam.txt').read_text().splitlines();E=np.array([[float(x) for x in t.split()] for t in l[1:5]]);K=np.array([[float(x) for x in t.split()] for t in l[7:10]])
 records[id]={'p':p,'meta':meta,'E':E,'K':K,'depth':read(p)}
yy,xx=np.mgrid[0:640:4,0:960:4];pixels=np.column_stack([xx.ravel(),yy.ravel(),np.ones(xx.size)]);rows=[]
for count,(id,r) in enumerate(sorted(records.items())):
 dep=r['depth'];prior=read(s/'metric_prior'/f'{id:08d}.dmb');ps=cv2.imread(str(s/'prior_support'/f'{id:08d}.png'),0);valid=(dep>0)&(prior>0);delta=np.zeros_like(dep);delta[valid]=np.abs(np.log(dep[valid]/prior[valid]));disagree=valid&(ps>=2)&(delta>r['meta']['prior_log_deviation_threshold'])
 nearcap=dep>=.98*r['meta']['dvp_depth_range_m'][1];geo=np.zeros(len(pixels),np.uint8)
 if disagree.any():
  ys,xs=np.where(disagree);z=dep[ys,xs];pix=np.column_stack([xs,ys,np.ones(len(xs))]);q=(pix@np.linalg.inv(r['K']).T)*z[:,None];world=(q-r['E'][:3,3])@r['E'][:3,:3];supports=np.zeros(len(xs),np.uint8)
  for nid in pairs[id]:
   todo=np.where(supports<2)[0]
   if not len(todo):break
   n=records[nid];q2=world[todo]@n['E'][:3,:3].T+n['E'][:3,3];uv=q2@n['K'].T;uv=np.rint(uv[:,:2]/np.maximum(uv[:,2:],1e-8)).astype(int);ok=(q2[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<640);inds=np.where(ok)[0];x,y=uv[inds].T;d=n['depth'][y,x];supports[todo[inds]]+=((d>0)&(np.abs(d-q2[inds,2])<=.01*q2[inds,2])).astype(np.uint8)
  remove=np.zeros_like(valid);remove[ys,xs]=(supports<2)|nearcap[ys,xs]
 else:remove=np.zeros_like(valid)
 keepmask=cv2.imread(str(s/'blocks'/f'mask_{id}.png'),0)>=128;remove|=~keepmask
 changed=int(((dep>0)&remove).sum());before=ROOT/'09_dvp/APD_before_prior_filter'/r['p'].parent.name;before.mkdir(parents=True,exist_ok=True)
 for name in ['depths.dmb','APD_normals.dmb']:
  p=r['p'].parent/name;backup=before/name
  if not backup.exists():os.link(p,backup)
  a=read(p);a[remove]=0;tmp=p.with_suffix('.masktmp');write(tmp,a);os.replace(tmp,p)
 assert not np.any(read(r['p'])[~keepmask])
 rows.append({'id':id,'label':r['meta']['source_label'],'prior_disagreement_pixels':int(disagree.sum()),'removed_depth_pixels':changed})
 if (count+1)%100==0:print('FILTERED',count+1,flush=True)
report={'images':len(rows),'removed_depth_pixels':sum(r['removed_depth_pixels'] for r in rows),'rule':'reliable cross-view MoGe prior AND log disagreement above calibrated threshold AND (fewer than 2 geometric supports OR depth at 98 percent search cap); mask always enforced','geometric_check_sampling_step':1,'rows':rows};(ROOT/'prior_filter_report.json').write_text(json.dumps(report,indent=2));print('FILTER COMPLETE',report['removed_depth_pixels'],flush=True)
