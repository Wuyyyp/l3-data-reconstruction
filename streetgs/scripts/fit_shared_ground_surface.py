from pathlib import Path
import json,time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from plyfile import PlyData,PlyElement
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
S=Path(args.input);O=Path(args.output);O.mkdir(parents=True,exist_ok=False)
t0=time.time();a=np.load(S);p=a['xyz'].astype(float);rgb=(a['rgb']*255).clip(0,255).astype('uint8');origin=np.zeros(3)
# One robust height observation per XY cell, rather than one plane per raw point.
cell=.15;keys=np.floor(p[:,:2]/cell).astype('int64');unique,inv,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
order=np.argsort(inv);starts=np.r_[0,np.cumsum(count)];z=np.empty(len(unique));spread=np.empty(len(unique));ambiguous=np.zeros(len(unique),bool)
for i in range(len(unique)):
 values=np.sort(p[order[starts[i]:starts[i+1]],2]);z[i]=np.median(values);spread[i]=np.ptp(values)
 # Keep suspected distinct vertical surfaces out of a single-valued road model.
 if len(values)>=4:
  gaps=np.diff(values);j=int(gaps.argmax());ambiguous[i]=gaps[j]>.25 and j+1>=2 and len(values)-j-1>=2
 ambiguous[i]|=spread[i]>.6
xy=(unique+.5)*cell;valid=~ambiguous
centers=np.column_stack([xy[valid],z[valid]]);tree=cKDTree(centers[:,:2]);d,ix=tree.query(centers[:,:2],k=32,workers=4)
outz=np.empty(len(centers));norm=np.empty((len(centers),3));rms=np.empty(len(centers));good=np.empty(len(centers),bool)
for start in range(0,len(centers),10000):
 stop=min(start+10000,len(centers));q=centers[start:stop];nn=centers[ix[start:stop]];dist=d[start:stop];X=np.concatenate([nn[:,:,:2]-q[:,None,:2],np.ones((len(q),32,1))],2);Y=nn[:,:,2]-q[:,None,2];base=np.exp(-.5*(dist/.45)**2)*(dist<=1.0);weights=base.copy()
 for iteration in range(5):
  lhs=np.einsum('nki,nk,nkj->nij',X,weights,X)+np.eye(3)[None]*1e-7;rhs=np.einsum('nki,nk,nk->ni',X,weights,Y);beta=np.linalg.solve(lhs,rhs);res=Y-np.einsum('nki,ni->nk',X,beta);weights=base*np.minimum(1,.04/np.maximum(abs(res),1e-8))
 outz[start:stop]=q[:,2]+beta[:,2];n=np.column_stack([-beta[:,:2],np.ones(len(q))]);n/=np.linalg.norm(n,axis=1)[:,None];norm[start:stop]=n;rms[start:stop]=np.sqrt((weights*res**2).sum(1)/weights.sum(1));good[start:stop]=(n[:,2]>.94)&(rms[start:stop]<.08)&((dist<.75).sum(1)>=8)
surf=np.column_stack([centers[:,:2],outz])[good];norm=norm[good];_,nearest=cKDTree(p).query(surf,workers=4);colors=rgb[nearest]
def ply(path,xyz,col,n=None):
 dtype=[(k,'f4') for k in ['x','y','z']]+[(k,'u1') for k in ['red','green','blue']]+([] if n is None else [(k,'f4') for k in ['nx','ny','nz']]);v=np.empty(len(xyz),dtype=dtype)
 for j,k in enumerate('xyz'):v[k]=xyz[:,j]
 for j,k in enumerate(['red','green','blue']):v[k]=col[:,j]
 if n is not None:
  for j,k in enumerate(['nx','ny','nz']):v[k]=n[:,j]
 PlyData([PlyElement.describe(v,'vertex')]).write(path)
ply(O/'road_surface_training_coordinates_15cm.ply',surf+origin,colors,norm)
xx=np.cross(np.broadcast_to([0.,1.,0.],norm.shape),norm);xx/=np.linalg.norm(xx,axis=1)[:,None];yy=np.cross(norm,xx);quat=Rotation.from_matrix(np.stack([xx,yy,norm],axis=2)).as_quat()[:,[3,0,1,2]]
np.savez_compressed(O/'ground.npz',xyz=surf.astype('f4'),normals=norm.astype('f4'),rotation=quat.astype('f4'),rgb=colors.astype('f4')/255)
dd,ii=cKDTree(surf[:,:2]).query(p[:,:2],workers=4);residual=((p-surf[ii])*norm[ii]).sum(1);supported=dd<.3;rejected=(~supported)|(abs(residual)>.15)
ply(O/'rejected_or_unsupported_training_coordinates.ply',p[rejected]+origin,rgb[rejected])
# Diagnostics relative to fitted model, not independent ground truth.
fig,axes=plt.subplots(2,2,figsize=(14,9));rng=np.random.default_rng(0);ids=rng.choice(len(p),min(100000,len(p)),replace=False)
axes[0,0].scatter(p[ids,0],p[ids,1],c=residual[ids],s=.2,vmin=-.2,vmax=.2,cmap='coolwarm');axes[0,0].set_title('Raw points: signed distance to fitted surface');axes[0,0].set_aspect('equal')
axes[0,1].hist(residual[supported],bins=150,range=(-.3,.3));axes[0,1].set_title('Raw residual to fitted surface (m); not ground truth')
# Sections across travel direction using first-to-last ego translation.
_,_,vh=np.linalg.svd(p[:,:2]-p[:,:2].mean(0),full_matrices=False);forward=vh[0];side=np.array([-forward[1],forward[0]]);along=p[:,:2]@forward;salong=surf[:,:2]@forward
for ax,percent in zip(axes[1],[35,65]):
 at=np.percentile(along,percent);sel=abs(along-at)<.25;ss=abs(salong-at)<.25;ax.scatter(p[sel,:2]@side,p[sel,2],s=2,alpha=.3,label='Patchwork++ merged');ax.scatter(surf[ss,:2]@side,surf[ss,2],s=5,label='shared surface');ax.set_title('Cross section at longitudinal %.2fm'%at);ax.set_xlabel('Lateral (m)');ax.set_ylabel('Height (m)');ax.legend()
fig.tight_layout();fig.savefig(O/'surface_diagnostics.png',dpi=150);plt.close(fig)
assert len(np.unique(np.floor(surf[:,:2]/cell).astype('int64'),axis=0))==len(surf)
assert np.isfinite(surf).all() and np.allclose(np.linalg.norm(norm,axis=1),1)
report=dict(source=str(S),input_points=len(p),output_points=len(surf),xy_cell_m=cell,ambiguous_cells=int(ambiguous.sum()),rejected_fit_cells=int((~good).sum()),raw_supported_points=int(supported.sum()),raw_abs_residual_percentiles_m=np.percentile(abs(residual[supported]),[50,90,95,99]).tolist(),raw_residual_p95_minus_p5_m=float(np.percentile(residual[supported],95)-np.percentile(residual[supported],5)),origin_world=origin.tolist(),seconds=time.time()-t0,method='XY median observations; shared moving robust local planes with 32 neighbors, 1m radius, 4cm Huber transition; one point per 15cm XY cell',limitations=['Single-height road representation; cells with separated height groups or >60cm range are excluded. This does not guarantee detection of all overpasses.','No independent geometry ground truth. A thinner output is imposed by the surface representation, not evidence of improved accuracy.','Existing poses unchanged; no training started.'],validation=dict(finite=True,unit_normals=True,one_point_per_xy_cell=True))
(O/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
