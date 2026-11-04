from runtime_setup import ROOT as RUN_ROOT,CONFIG,VARIANT,render,SETTINGS
import os,sys,json,time,random,math,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from PIL import Image
from scene.training_runtime import parse_training_arguments
from scene.methods.initialization_3dgs import build_gaussian_model_3dgs
from scene.training_context import build_training_context,attach_scene_and_gaussians_to_context
from scene.training_loop import update_3dgs_training_schedule,run_3dgs_parameter_update_method
from scene.methods.densification_stage import run_3dgs_densification_method
from scene.methods.pruning_methods import run_3dgs_pruning_method
from scene.dataset_readers import fetchPly

from lib_bilagrid import BilateralGrid,slice
from utils.static_losses import masked_ssim
DATA=Path(SETTINGS['data_cache'])
PLY=SETTINGS['initialization_ply']
PARAMS=['_xyz','_features_dc','_features_rest','_scaling','_rotation','_opacity']
def load_cameras(limit=0):
 rows=sorted(torch.load(DATA/'cameras.pt'),key=lambda x:x['image_name']);rows=rows[:limit] if limit else rows;cameras=[]
 for i,row in enumerate(rows):
  c=SimpleNamespace(**row);c.uid=i
  for k in ['world_view_transform','full_proj_transform','camera_center']:setattr(c,k,getattr(c,k).cuda())
  gt=Image.open(Path(SETTINGS['images'])/(c.image_name+'.png')).convert('RGB');assert gt.size==(c.image_width,c.image_height)
  c.original_image=torch.from_numpy(np.array(gt).copy()).permute(2,0,1).float()/255
  d=torch.load(DATA/(c.image_name+'.pt'));c.alpha_mask=d['valid'];c.lidar_depth=d['depth'];c.depth_mask=d['depth_valid'];c.sky=d['sky'];cameras.append(c)
 return cameras
class BridgeScene:
 def __init__(self,cameras,model,path):self.cameras=cameras;self.gaussians=model;self.cameras_extent=20.;self.model_path=str(path)
 def getTrainCameras(self):return self.cameras
 def getTestCameras(self):return []
def apply_grid(grid,c,rgb,xy):
 shape=(c.image_height,c.image_width)
 if shape not in xy:
  h,w=shape;y,x=torch.meshgrid((torch.arange(h,device='cuda')+.5)/h,(torch.arange(w,device='cuda')+.5)/w,indexing='ij');xy[shape]=torch.stack([x,y],-1)
 return slice(grid,xy[shape],rgb.permute(1,2,0),torch.tensor([c.uid],device='cuda'))['rgb'].permute(2,0,1)
def main():
 lp,op,pp,args=parse_training_arguments(sys.argv[1:]);dataset=lp.extract(args);opt=op.extract(args);pipe=pp.extract(args);out=Path(dataset.model_path);out.mkdir(parents=True,exist_ok=True)
 (out/'config.json').write_text(json.dumps(vars(args),indent=2));random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
 context=build_training_context(dataset,opt,pipe,args);cams=load_cameras(int(os.environ.get('CAMERA_LIMIT','0')));model=build_gaussian_model_3dgs(dataset,opt);model.create_from_pcd(fetchPly(PLY),cams,20.);model.training_setup(opt)
 initial={k:hashlib.sha256(getattr(model,k).detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k in PARAMS}
 (out/'initialization.json').write_text(json.dumps(dict(count=len(model.get_xyz),hashes=initial,camera_names=[c.image_name for c in cams]),indent=2))
 scene=BridgeScene(cams,model,out);context=attach_scene_and_gaussians_to_context(context,scene,model)
 # EAS weights must exclude all masked pixels, including their edge neighborhoods.
 for i,c in enumerate(cams) if context.runtime_state['edge_maps'] else []:
  invalid=torch.nn.functional.max_pool2d((~c.alpha_mask).float()[None],3,1,1)[0,0].bool();context.runtime_state['edge_maps'][i]*=~invalid
 context.runtime_state['edge_map_pool']=context.runtime_state['edge_maps'].copy()
 grid_state=context.runtime_state['bilateral_grid'];grid=grid_state.model;grid_opt=grid_state.optimizer;rng=random.Random(0);stack=[];start=0;xy={};bg=torch.zeros(3,device='cuda')
 sky=None
 if CONFIG.get('with_sky'):
  from sky_bridge import NativeSky
  sky=NativeSky(cams,model.get_xyz)
 resume=os.environ.get('RESUME','')
 if resume:
  audit=torch.load(resume,map_location='cpu')
  assert bool(audit.get('sky') is not None)==bool(sky is not None), 'Checkpoint sky mode differs'
  assert audit['camera_names']==[c.image_name for c in cams], 'Checkpoint cameras differ'
  history=out/'curve.jsonl'
  if history.exists():
   previous=[json.loads(line)['iteration'] for line in history.read_text().splitlines()]
   assert previous==list(range(1,audit['iteration']+1)), 'Curve/checkpoint mismatch; preserve history and recover explicitly'
  del audit
 if resume:
  ck=torch.load(resume);assert ck['camera_names']==[c.image_name for c in cams], 'Checkpoint cameras differ';model.restore(ck['model'],opt);grid_state.restore(ck['bilateral_state']);start=ck['iteration'];stack=ck['stack'];rng.setstate(ck['camera_rng']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
  for k,g in ck['grads'].items():getattr(model,k).grad=g
  context.runtime_state.update(ck['runtime']);ids=ck['edge_pool'];context.runtime_state['edge_camera_pool']=[cams[i] for i in ids];context.runtime_state['edge_map_pool']=[context.runtime_state['edge_maps'][i] for i in ids];
  if sky is not None:sky.model.load_state_dict(ck['sky'])
  if 'exposure' in ck:
   model._exposure.data.copy_(ck['exposure']);model.exposure_optimizer.load_state_dict(ck['exposure_optimizer'])
  (out/'resume_validation.json').write_text(json.dumps(dict(iteration=start,pending_gradients=list(ck['grads']),rap_resets=context.runtime_state['rap_reset_count'],edge_pool=len(ids),camera_stack=len(stack))))
  del ck
 elif (out/'curve.jsonl').exists():raise FileExistsError('Resume existing curve explicitly')
 stop=int(os.environ.get('STOP_AFTER',opt.iterations));torch.cuda.reset_peak_memory_stats();began=time.time()
 print('INITIALIZED',len(cams),len(model.get_xyz),'threshold',opt.densify_grad_threshold,flush=True)
 for it in range(start+1,stop+1):
  torch.cuda.synchronize();t=time.perf_counter();update_3dgs_training_schedule(context,model,it)
  if not stack:stack=list(range(len(cams)))
  c=cams[stack.pop(rng.randrange(len(stack)))];pkg=render(c,model,pipe,bg);raw=pkg['render']
  if sky is not None:
   sky.model.update_learning_rate(it);raw=raw+(1-pkg['acc'])*sky.render(c)
  pred=grid_state.correct(raw,c.image_name);gt=c.original_image.cuda();mask=c.alpha_mask.cuda().bool()
  l1=(pred-gt).abs()[mask.expand_as(pred)].mean();ssim=masked_ssim(pred if CONFIG.get('grid_ssim') else raw,gt,mask);valid=c.depth_mask.cuda().bool();depth=c.lidar_depth.cuda();invgt=1/depth.clamp_min(.1)
  dl=(pkg['depth'][valid]-invgt[valid]).abs().mean() if valid.any() else pkg['depth'].sum()*0
  if CONFIG.get('metric_depth'):
   valid=mask&~c.sky.cuda().bool()&torch.isfinite(depth)&(depth>=.1)&(depth<=200)
   err=(pkg['depth']/pkg['acc'].clamp_min(1e-6)-depth).abs()[valid];dl=err.topk(max(1,int(.95*len(err))),largest=False).values.mean() if len(err) else pkg['depth'].sum()*0
  weight=opt.depth_l1_weight_init*((opt.depth_l1_weight_final/opt.depth_l1_weight_init)**(it/opt.iterations));weight=.1 if CONFIG.get('metric_depth') else weight;tv=10*grid.tv_loss();loss=(1-opt.lambda_dssim)*l1+opt.lambda_dssim*(1-ssim)+weight*dl+tv
  if sky is not None:
   from sky_bridge import sky_loss
   pkg['scene_alpha']=pkg['acc'];loss=loss+sky_loss(c,pkg,sky)
  assert torch.isfinite(loss);loss.backward()
  if sky is not None:
   assert sky.model._features_dc.grad is not None and torch.isfinite(sky.model._features_dc.grad).all()
  if it==start+1:
   assert grid.grids.grad is not None and torch.isfinite(grid.grids.grad).all();assert model._xyz.grad is not None and torch.isfinite(model._xyz.grad).all();print('GRADIENTS_FINITE',float(grid.grids.grad.abs().sum()),int(valid.sum()),flush=True)
  state=dict(bg=bg,viewpoint_camera=c,viewspace_point_tensor=pkg['viewspace_points'],visibility_filter=pkg['visibility_filter'],radii=pkg['radii'])
  with torch.no_grad():
   run_3dgs_parameter_update_method(context,model,it,pkg['radii'])
   run_3dgs_densification_method(context,model,it,state);run_3dgs_pruning_method(context,model,it)
   if sky is not None:sky.model.update_optimizer()
  torch.cuda.synchronize();row=dict(iteration=it,camera=c.image_name,loss=float(loss),l1=float(l1),ssim=float(ssim),depth_loss=float(dl),depth_weight=weight,depth_pixels=int(valid.sum()),grid_tv=float(tv),points=len(model.get_xyz),train_ms=(time.perf_counter()-t)*1000,elapsed=time.time()-began,peak_allocated_bytes=torch.cuda.max_memory_allocated(),sky_points=len(sky.model.get_xyz) if sky is not None else 0,rap_reset_count=context.runtime_state['rap_reset_count'],edge_pool=len(context.runtime_state['edge_camera_pool']),variant=VARIANT,actors=0)
  with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  if it%10==0 or it==start+1:print(json.dumps(row),flush=True)
  if it%1000==0 or it==stop:
   ck=dict(exposure=model._exposure.detach().clone(),exposure_optimizer=model.exposure_optimizer.state_dict(),sky=sky.capture() if sky is not None else None,variant=VARIANT,bilateral_state=grid_state.capture(),iteration=it,model=model.capture(),grid=grid.state_dict(),grid_optimizer=grid_opt.state_dict(),camera_names=[c.image_name for c in cams],stack=stack,camera_rng=rng.getstate(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),grads={k:getattr(model,k).grad for k in PARAMS if getattr(model,k).grad is not None},runtime={k:context.runtime_state[k] for k in ['rap_reset_count','rap_trigger_iterations']},edge_pool=[c.uid for c in context.runtime_state['edge_camera_pool']] if context.runtime_state['edge_maps'] else []);dest=out/('checkpoint_%06d.pth'%it);torch.save(ck,str(dest)+'.tmp');os.replace(str(dest)+'.tmp',dest);del ck
  (out/'status.json').write_text(json.dumps(row)) if it%100==0 else None
  del loss,pkg,pred,raw,l1,ssim,dl,tv,state
 # Full checkpoint retains all state.
 print('TRAIN_COMPLETE',stop,flush=True)
if __name__=='__main__':main()
