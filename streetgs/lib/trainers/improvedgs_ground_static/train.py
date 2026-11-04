import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'deps'))
from ground_extension import *
from scene.training_runtime import parse_training_arguments
from scene.methods.initialization_3dgs import build_gaussian_model_3dgs
from scene.training_context import build_training_context,attach_scene_and_gaussians_to_context
from scene.training_loop import update_3dgs_training_schedule,run_3dgs_parameter_update_method
from scene.methods.densification_stage import run_3dgs_densification_method
from scene.methods.pruning_methods import run_3dgs_pruning_method
import scene.methods.densification_stage as ds_stage
import resource
from utils.static_losses import masked_ssim

def main():
 start_wall=time.time();lp,op,pp,args=parse_training_arguments(sys.argv[1:]);data=lp.extract(args);opt=op.extract(args);pipe=pp.extract(args);out=Path(data.model_path);out.mkdir(parents=True,exist_ok=True)
 random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
 stamp=time.time();data_set,cams=dataset();dataset_seconds=time.time()-stamp
 context=build_training_context(data,opt,pipe,args);bg=build_gaussian_model_3dgs(data,opt);bg.create_from_pcd(data_set.scene_info.point_cloud,cams,20.);bg.training_setup(opt)
 scene=DynamicScene(data_set,cams,bg,out);sky=scene.sg.sky_cubemap
 smoke=os.environ.get('SMOKE')=='1'
 initial=scene.hashes();(out/('resume_initialization.json' if os.environ.get('RESUME') else 'initialization.json')).write_text(json.dumps(dict(hashes=initial,camera_names=[c.image_name for c in cams],actors=len(scene.sg.obj_list)),indent=2));del initial
 context.method_config['use_eas']=False;attach_scene_and_gaussians_to_context(context,scene,bg);context.method_config['use_eas']=True
 # Identical EAS pool and per-view operator, but materialize each edge map on demand.
 context.runtime_state['edge_maps']=list(range(len(cams)));context.runtime_state['edge_map_pool']=list(range(len(cams)));context.runtime_state['edge_camera_pool']=cams.copy();ds_stage.render=scene.render
 grid_state=context.runtime_state.get('bilateral_grid');grid=grid_state.model if grid_state else None
 rng=random.Random(0);stack=[];start=0;resume=os.environ.get('RESUME')
 if resume:
  ck=torch.load(resume);assert ck['camera_names']==[c.image_name for c in cams]
  if (out/'curve.jsonl').exists():assert [json.loads(z)['iteration'] for z in (out/'curve.jsonl').read_text().splitlines()]==list(range(1,ck['iteration']+1))
  bg.restore(ck['model'],opt);scene.restore(ck['dynamic']);grid_state.restore(ck['bilateral_state']) if grid_state else None;start=ck['iteration'];stack=ck['stack'];rng.setstate(ck['camera_rng']);context.runtime_state.update(ck['runtime']);ids=ck['edge_pool'];context.runtime_state['edge_camera_pool']=[cams[i] for i in ids];context.runtime_state['edge_map_pool']=ids.copy()
  for k,g in ck['grads'].items():getattr(bg,k).grad=g
  bg._exposure.data.copy_(ck['exposure']);bg.exposure_optimizer.load_state_dict(ck['exposure_optimizer'])
  random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
  if smoke:
   def exact(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b), 'Restored tensor mismatch'
    elif isinstance(a,dict):
     assert set(a)==set(b)
     for k in a:exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
     assert len(a)==len(b)
     for x,y in zip(a,b):exact(x,y)
    else:assert a==b,(a,b)
   exact(ck['dynamic'],scene.capture())
  (out/'resume_validation.json').write_text(json.dumps(dict(iteration=start,pending_grads=list(ck['grads']),actor_count=scene.actor_count(),sky_count=0)));del ck
 elif (out/'curve.jsonl').exists():raise FileExistsError('Existing curve requires explicit resume')
 stop=int(os.environ.get('STOP_AFTER','25000'));background=torch.zeros(3,device='cuda');torch.cuda.reset_peak_memory_stats();loader=getattr(cams[0],'_payload_loader',None)
 (out/'startup.json').write_text(json.dumps(dict(dataset_seconds=dataset_seconds,ready_before_first=loader.stats() if loader else dict(ready=len(cams),total=len(cams)),initialization_seconds=time.time()-start_wall)))
 prefetch=Prefetch(cams)
 began=time.time();save_seconds=0.;grad_verified=False;minimal=out.name=='full';previous_checkpoint=Path(resume) if resume else None
 curve_stream=(out/'curve.jsonl').open('a',buffering=1024*1024)
 for it in range(start+1,stop+1):
  torch.cuda.synchronize();t=time.perf_counter();update_3dgs_training_schedule(context,bg,it);scene.update_lr(it)
  if not stack:stack=list(range(len(cams)))
  c=cams[stack.pop(rng.randrange(len(stack)))];prefetch.activate(c);prefetch.next(rng,stack,cams,True);w=time.perf_counter();gt=c.original_image.cuda();guidance=c.guidance;data_wait=time.perf_counter()-w
  mask=c.alpha_mask.cuda();mask=mask & ~guidance['obj_bound'].cuda().bool() if not cfg.model.nsg.include_obj else mask;sm=guidance['sky_mask'].cuda().bool();depth=guidance['lidar_depth'].cuda();valid=mask&~sm&torch.isfinite(depth)&(depth>=.1)&(depth<=200)
  assert not torch.any(mask&guidance['obj_bound'].cuda().bool())
  assert scene.actor_count()==0
  if it==start+1:(out/'static_validation.json').write_text(json.dumps(dict(passed=True,actors=0,static_pixels=int(mask.sum()),excluded_pixels=int(guidance['obj_bound'].sum()),mesh_depth_loss=False)))
  t_data=time.perf_counter();ev=[torch.cuda.Event(enable_timing=True) for _ in range(5)];ev[0].record()
  pkg=scene.render(c,bg,pipe,background,metric=True);raw=pkg['render']+(1-pkg['acc'])*sky(c);pred=grid_state.correct(raw,c.image_name) if grid_state else (scene.sg.color_correction(c,raw) if scene.sg.color_correction is not None else raw)
  ev[1].record();t_render=time.perf_counter()
  l1=(pred-gt).abs()[mask.expand_as(pred)].mean();ssim=masked_ssim(pred if scene.sg.color_correction is not None else raw,gt,mask);metric_depth=pkg['depth']/pkg['acc'].clamp_min(1e-10);dl=(metric_depth[valid]-depth[valid]).abs().mean() if valid.any() else pkg['depth'].sum()*0
  ml=metric_depth.sum()*0  # Vehicle mesh depth is excluded from static reconstruction.
  weight=opt.depth_l1_weight_init*(opt.depth_l1_weight_final/opt.depth_l1_weight_init)**(it/opt.iterations);tv=10*grid.tv_loss() if grid is not None else raw.sum()*0;loss=(1-opt.lambda_dssim)*l1+opt.lambda_dssim*(1-ssim)+.1*dl+.05*ml+tv+sky_loss(c,pkg,sky)
  ground_loss,ground_metrics=scene.ground_loss(c,pipe,background,gt);loss=ground_loss if scene.warmup() else loss+ground_loss
  ev[2].record();t_loss=time.perf_counter()
  assert torch.isfinite(loss);loss.backward()
  if scene.sg.color_correction is not None and not scene.warmup():
   g=scene.sg.color_correction.affine_trans.grad;assert g is not None and torch.isfinite(g).all()
  ev[3].record();t_backward=time.perf_counter()
  if not scene.warmup() and not grad_verified and not cfg.model.nsg.include_obj:
   assert grid is None or (grid.grids.grad is not None and torch.isfinite(grid.grids.grad).all() and grid.grids.grad.abs().sum()>0)
   assert sky.sky_cube_map.grad is not None and torch.isfinite(sky.sky_cube_map.grad).all() and sky.sky_cube_map.grad.abs().sum()>0
   (out/"gradient_validation.json").write_text(json.dumps(dict(passed=True,iteration=it,static=True)));grad_verified=True
  if not scene.warmup() and not grad_verified and scene.sg.graph_obj_list:
   grads={}
   for n in scene.sg.graph_obj_list:
    m=getattr(scene.sg,n)
    for k in ['_xyz','_features_dc']:
     g=getattr(m,k).grad
     assert g is not None and torch.isfinite(g).all();grads[n+k]=float(g.abs().sum())
   pose=scene.sg.actor_pose;pg=torch.zeros(1,device='cuda');rg=torch.zeros(1,device='cuda')
   assert pg is not None and rg is not None and torch.isfinite(pg).all() and torch.isfinite(rg).all()
   if sum(grads.values())>0:
    assert grid is None or (grid.grids.grad is not None and float(grid.grids.grad.abs().sum())>0 and torch.isfinite(grid.grids.grad).all())
    assert sky.sky_cube_map.grad is not None and torch.isfinite(sky.sky_cube_map.grad).all() and float(sky.sky_cube_map.grad.abs().sum())>0
    (out/'gradient_validation.json').write_text(json.dumps(dict(passed=True,iteration=it,actor_grads=grads,fixed_supplied_affine=True,pose_translation_grad=float(pg.abs().sum()),pose_rotation_grad=float(rg.abs().sum()),grid_grad=float(grid.grids.grad.abs().sum()) if grid is not None else None,sky_grad=float(sky.sky_cube_map.grad.abs().sum()))));grad_verified=True
  scene.audit_finite(gradients=True)
  state=dict(bg=background,viewpoint_camera=c,viewspace_point_tensor=pkg['viewspace_points'],visibility_filter=pkg['visibility_filter'],radii=pkg['radii'])
  with torch.no_grad():
   scene.accumulate_ground(pkg);scene.step_ground();scene.densify_ground(it,smoke)
   if not scene.warmup():scene.update_actors(it,pkg,smoke);run_3dgs_parameter_update_method(context,bg,it,pkg['radii']);run_3dgs_densification_method(context,bg,it,state);run_3dgs_pruning_method(context,bg,it)
  ev[4].record();torch.cuda.synchronize();t_end=time.perf_counter();row=dict(iteration=it,camera=c.image_name,loss=float(loss),l1=float(l1),ssim=float(ssim),depth_loss=float(dl),mesh_depth_loss=float(ml),points=len(bg.get_xyz),actor_points=scene.actor_count(),actors=len(scene.sg.obj_list),visible_actors=len(pkg['actor_ranges'])-1,sky_points=0,sky_type='cubemap',train_ms=(time.perf_counter()-t)*1000,data_wait_ms=data_wait*1000,ready=loader.stats()['ready'] if loader else len(cams),peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed=time.time()-began)
  row.update(ground_metrics);row['ground_points']=len(scene.ground.initial);row['ground_split_events']=len(scene.ground.events);row['stage']='ground' if scene.warmup() else 'joint'
  row.update(data_stage_ms=(t_data-t)*1000,render_submit_ms=(t_render-t_data)*1000,loss_submit_ms=(t_loss-t_render)*1000,backward_submit_ms=(t_backward-t_loss)*1000,update_and_sync_ms=(t_end-t_backward)*1000,cuda_timeline_ms=[ev[j].elapsed_time(ev[j+1]) for j in range(4)],major_faults=resource.getrusage(resource.RUSAGE_SELF).ru_majflt)
  curve_stream.write(json.dumps(row)+'\n')
  if not minimal:curve_stream.flush()
  if it%(1000 if minimal else 10)==0 or it==start+1:print(json.dumps(row),flush=True)
  if it%1000==0 or it==stop:
   curve_stream.flush();scene.audit_finite(optimizer=True)
   s=time.time();ck=dict(iteration=it,model=bg.capture(),dynamic=scene.capture(),bilateral_state=grid_state.capture() if grid_state else None,grid=grid.state_dict() if grid else None,camera_names=[c.image_name for c in cams],stack=stack,camera_rng=rng.getstate(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),grads={k:getattr(bg,k).grad for k in PARAMS if getattr(bg,k).grad is not None},runtime={k:context.runtime_state[k] for k in ['rap_reset_count','rap_trigger_iterations']},edge_pool=[c.uid for c in context.runtime_state['edge_camera_pool']],exposure=bg._exposure.detach().clone(),exposure_optimizer=bg.exposure_optimizer.state_dict());dest=out/('checkpoint_%06d.pth'%it);torch.save(ck,str(dest)+'.tmp');os.replace(str(dest)+'.tmp',dest);del ck
   if minimal and previous_checkpoint is not None and previous_checkpoint!=dest:
    assert previous_checkpoint.resolve().parent==out.resolve()
    previous_checkpoint.unlink(missing_ok=True)
   previous_checkpoint=dest;save_seconds+=time.time()-s
  if it%(1000 if minimal else 100)==0:(out/'status.json').write_text(json.dumps(row))
  del loss,ground_loss,pkg,pred,raw,l1,ssim,dl,tv,state
 curve_stream.flush();curve_stream.close()
 assert grad_verified,'No nonzero supervised gradient observed'
 prefetch.close();(out/'prefetch.json').write_text(json.dumps(prefetch.stats()))
 # Finish materializing and hash every payload after training, not on its critical start path.
 training_wall_seconds=time.time()-start_wall;hash_started=time.time()
 hashes={}
 for c in cams:
  vals={'image':c.original_image,**c.guidance};hashes[c.image_name]={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in vals.items()}
 (out/'payload_hashes.json').write_text(json.dumps(hashes))
 (out/'timing.json').write_text(json.dumps(dict(wall_seconds=time.time()-start_wall,training_wall_seconds=training_wall_seconds,payload_audit_seconds=time.time()-hash_started,checkpoint_seconds=save_seconds,loader=loader.stats() if loader else None)))
 if loader is not None:loader.close()
 print('TRAIN_COMPLETE',stop,flush=True)
if __name__=='__main__':main()
