"""4DGF-inspired projection-coverage initialization; no change to RGB/depth losses."""
import math,time,json
from pathlib import Path
import torch
from lib.config import cfg

def seed_layers(points,cameras,center,radius):
 from lib.trainers.improvedgs_dynamic.async_pipeline import masks_for_sky
 begin=time.perf_counter();n=int(cfg.model.sky.candidates_per_layer);multipliers=cfg.model.sky.layer_radius_multipliers
 i=torch.arange(n,device=points.device,dtype=points.dtype);z=(i+.5)/n;a=i*(math.pi*(3-math.sqrt(5)));rr=(1-z*z).sqrt();dirs=torch.stack((rr*a.cos(),rr*a.sin(),z),1)
 radii=radius*torch.tensor(multipliers,device=points.device,dtype=points.dtype);xyz=torch.cat([center+r*dirs for r in radii]);keep=torch.zeros(len(xyz),dtype=torch.bool,device=points.device)
 assert len(radii)==3 and bool((radii[1:]>radii[:-1]).all())
 def project(x,c,K,h,w):
  T=c.world_view_transform.to(x).T;v=x@T[:3,:3].T+T[:3,3];q=v@K.T;uv=q[:,:2]/q[:,2:3].clamp_min(1e-6);pix=uv.round().long();inside=(v[:,2]>1e-3)&torch.isfinite(uv).all(1)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h);return pix,inside
 for ci,c in enumerate(cameras):
  if bool(keep.all()):break
  factor=max(1,int(max(c.image_width,c.image_height)*float(cfg.model.sky.coverage_window_fraction)//2));h=c.image_height//factor;w=c.image_width//factor;K=c.K.to(points).clone();K[:2]/=factor
  pix,inside=project(points,c,K,h,w);occ=torch.zeros((h,w),device=points.device,dtype=torch.bool);p=pix[inside];occ[p[:,1],p[:,0]]=True
  pending=(~keep).nonzero().flatten();p,inside=project(xyz[pending],c,K,h,w);loc=inside.nonzero().flatten();q=p[loc];empty=~occ[q[:,1],q[:,0]]
  # Do not spend initialization points on excluded vehicles/ego regions.
  masks=masks_for_sky(c);valid=c.alpha_mask.bool()
  if not cfg.model.nsg.include_obj:valid=valid&~masks['obj_bound'].bool()
  valid=valid.reshape(c.image_height,c.image_width).to(points.device)
  uv=q*factor;empty &= valid[uv[:,1].clamp_max(c.image_height-1),uv[:,0].clamp_max(c.image_width-1)]
  keep[pending[loc[empty]]]=True
  if ci%100==0:print('FAR_INIT_COVERAGE',ci+1,len(cameras),int(keep.sum()),flush=True)
 counts=[int(keep[j*n:(j+1)*n].sum()) for j in range(len(radii))];total=int(cfg.model.sky.initial_retained_points)
 assert min(counts)>=total//len(radii)+1,('Insufficient coverage candidates per layer',counts,total)
 selected=[]
 for j in range(len(radii)):
  ids=keep[j*n:(j+1)*n].nonzero().flatten()+j*n;k=total//len(radii)+(j<total%len(radii));take=torch.div(torch.arange(k,device=points.device)*len(ids),k,rounding_mode='floor');selected.append(ids[take])
 ids=torch.cat(selected);x=xyz[ids];assert len(x)==total
 info=dict(method='multisphere_projection_coverage',candidates_per_layer=n,candidates=n*len(radii),coverage_retained_per_layer=counts,retained_per_layer=[len(s) for s in selected],retained=len(x),center=center.tolist(),reference_radius=float(radius),layer_radii=radii.tolist(),radius_multipliers=list(multipliers),sky_mask_cameras=0,hemisphere=True,initialization_seconds=time.perf_counter()-begin,coverage_window_fraction=float(cfg.model.sky.coverage_window_fraction),initial_budget_matched=True,static_valid_masks=True,depth_or_occlusion_test=False)
 out=Path(cfg.model_path);out.mkdir(parents=True,exist_ok=True);(out/'far_initialization.json').write_text(json.dumps(info,indent=2));print('FAR_INIT_COMPLETE',json.dumps(info),flush=True)
 return x,radii,info
