"""New 1031 pipeline, stage receipts, clip-wide pose-QC rejection, no old jobs."""
import re
import argparse,concurrent.futures,csv,json,os,signal,subprocess,time
from pathlib import Path
from task_queue import ROOT,write_json
ADAPTER=Path(__file__).resolve().parent
def check_quality(work,clip,unused):
 if not (work/'accepted_delivery.json').is_file():raise RuntimeError('Window delivery not admitted')
REPO=Path(__file__).resolve().parents[1]
PY=__import__('sys').executable
POSEPY=os.environ.get('POSE_PYTHON',PY)
ENGINE=REPO/'engine/affine_shadow'
TEMPLATE=ROOT/'train_template.json'

class AnnotationConflict(RuntimeError):
 pass

def annotation_conflict(log,start):
 with log.open('rb') as stream:
  stream.seek(0,2)
  stream.seek(max(start,stream.tell()-2048))
  tail=stream.read().rstrip().splitlines()
 return bool(tail and re.fullmatch(rb'ValueError: Conflicting 2D prompts for track -?[0-9]+ in [a-zA-Z0-9_]+',tail[-1]))

class Job:
 def __init__(self,work,row,gpu):
  self.work=Path(work);self.work.mkdir(parents=True,exist_ok=True);self.row=row;self.gpu=str(gpu)
  self.env=dict(os.environ,CUDA_VISIBLE_DEVICES=self.gpu,EGL_DEVICE_ID=self.gpu,LD_LIBRARY_PATH=os.environ.get('EGL_LIB_DIR','')+':'+os.environ.get('LD_LIBRARY_PATH',''),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
  self.prep_env=dict(self.env,PYTHONPATH=os.pathsep.join(x for x in [str(ROOT),os.environ.get('PREPARATION_DEPS',''),os.environ.get('PYTHONPATH','')] if x))
  (self.work/'logs').mkdir(exist_ok=True);(self.work/'stages').mkdir(exist_ok=True)
  write_json(self.work/'job_input.json',row)
 def run(self,stage,command,env=None,cwd=None,receipt=True):
  marker=self.work/'stages'/(stage+'.json')
  check_quality(self.work,self.row['clip_name'],ROOT)
  if receipt and marker.exists():return
  started=time.time();write_json(self.work/'status.json',dict(stage=stage,status='running',started=started,gpu=self.gpu))
  with (self.work/'logs'/(stage+'.log')).open('a') as f:
   log_start=f.tell()
   process=subprocess.Popen([str(x) for x in command],env=env or self.env,cwd=cwd or self.work,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    while process.poll() is None:
     time.sleep(10);check_quality(self.work,self.row['clip_name'],ROOT)
   except BaseException:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
    raise
  check_quality(self.work,self.row['clip_name'],ROOT)
  if process.returncode:
   log=self.work/'logs'/(stage+'.log')
   msg=stage+' failed; see '+str(log)
   if process.returncode==1 and stage in ('mesh','mesh_preflight') and annotation_conflict(log,log_start):
    raise AnnotationConflict(msg)
   raise RuntimeError(msg)
  if receipt:write_json(marker,dict(stage=stage,started=started,finished=time.time(),seconds=time.time()-started))
 def script(self,stage,file,*args,env=None,python=PY):
  if env is None:
   env=self.prep_env
   if python==POSEPY:
    env=dict(self.env,PYTHONNOUSERSITE='1')
    env.pop('PYTHONPATH',None)
  self.run(stage,[python,ROOT/file,*args],env)
 def mesh(self):
  if self.row.get('no_mesh'):
   self.script('native_vehicles','native_vehicles.py',self.work)
   return
  driver=Path(os.environ['MESH_ROOT'])/'register_sample.py'
  cmd=[os.environ['MESH_PYTHON'],driver,self.work,self.gpu]
  activation=os.environ.get('MESH_ENV_ACTIVATE')
  if activation:
   cmd=['bash','-c','source "$1"; shift; unset PYTHONPATH; exec "$@"','mesh',activation,*cmd]
  self.run('mesh',cmd)
 def prepare_geometry(self):
  w=self.work
  src=Path(self.row['source'])
  self.script('window_input',ADAPTER/'window_input.py',w,src)
  skyenv=dict(self.env)
  skyenv['PYTHONPATH']=os.pathsep.join([str(Path(os.environ['POSE_ROOT'])/'src'),str(ROOT),os.environ.get('POSE_LOMA_SRC','')])
  for key in ('HF_HOME','TORCH_HOME'):
   if os.environ.get('POSE_'+key):skyenv[key]=os.environ['POSE_'+key]
  self.script('sky',ADAPTER/'sky.py',w,python=POSEPY,env=skyenv)
  self.script('geometry',ADAPTER/'geometry_vla.py','--work',w,'--sample',self.row['sample_name'])
 def prepare(self):
  self.prepare_geometry()
  w=self.work
  c=json.loads(TEMPLATE.read_text());c.update(source_path=str(w/'geometry/converted'),model_path=str(w/'geometry/model'));c['data']['type']='M18';c['data']['regenerate_pcd']=True
  c['workspace']=str(REPO)
  c['model'].pop('ground',None);c.pop('improvedgs',None);c.pop('full_background_source',None)
  write_json(w/'native_config.yaml',c)
  def dvp():
   if self.row.get('no_far'):return
   if self.row.get('far_ply'):
    write_json(w/'dvp_result.json',{'ply':self.row['far_ply'],'far_only':True})
    return
   self.script('dvp_delivery',ADAPTER/'dvp_input.py',w)
   self.script('dvp','run_dvp.py',w,self.gpu,python=os.environ['DVP_PYTHON'],env=self.env)
  with concurrent.futures.ThreadPoolExecutor(3) as pool:
   mesh=pool.submit(self.mesh)
   far=pool.submit(dvp)
   native=pool.submit(self.script,'native','prepare_native.py',w)
   far.result();native.result()
   self.script('merge','merge_background.py',w,*(['--no-far'] if self.row.get('no_far') else []))
   mesh.result()
  shared=json.loads((w/'shared_mesh_acceptance.json').read_text())
  assert (self.row.get('no_mesh') or shared['mesh_version']=='sam3-shared-vehicle-pose-1102') and shared['shared_camera_pose'] and shared['source_fields_preserved']
  assert shared['identical_camera_keys']==7*shared['track_frames']
  meshroot=w/'mesh_inputs';write_json(meshroot/'case.json',dict(baseline=str(w/'geometry')))
  self.run('mesh_prepared',[PY,ROOT/'code/preparation/prepare_mesh.py',meshroot],self.prep_env)
  protected=w/'protected';protected.mkdir(exist_ok=True);write_json(protected/'case.json',dict(baseline=str(w/'geometry'),mesh_reference=str(meshroot)))
  c.update(source_path=str(protected/'converted'),model_path=str(protected/'model'));write_json(protected/'train.yaml',c)
  self.run('protected',[PY,ROOT/'code/preparation/prepare_preserve.py',protected],self.prep_env)
  original=w/'original';original.mkdir(exist_ok=True)
  import sys;sys.path.insert(0,str(REPO/'scripts'));from run_ground_pipeline import DEFAULTS
  recipe=dict(DEFAULTS,source=str(w/'geometry/converted'),output=str(original),ego_mask=str(w/'pose/source_fixed_masks'),frames=list(range(101)),cameras=[0,1,2,3,4,9,10],scene_mode='dynamic',sky_mode='hemisphere',reuse_prepared_background=True,split='train-all',mesh_source=str(protected),pose_csv=str(w/'pose/run/deliverables/colmap-ba/image_camera_to_world_after_ba.csv'),origin_frame=0,dynamic_frame=50)
  write_json(original/'preprocess_config.json',recipe)
  for name,code,target in [('extract','extract.py','preprocess/extraction'),('separate','prepared_background.py','preprocess/separation'),('refine','refine.py','preprocess/refined'),('prepare','prepare.py','data')]:
   (original/target).parent.mkdir(parents=True,exist_ok=True)
   env=dict(self.prep_env,ROAD_PIPELINE_CONFIG=str(original/'preprocess_config.json'),ROAD_STAGE_OUTPUT=str(original/target))
   self.run('ground_'+name,[PY,ROOT/'code/ground'/code],env)
  if not (w/'stages/road_masks.json').is_file():self.configure(original)
  with concurrent.futures.ThreadPoolExecutor(4) as pool:
   fs=[
    pool.submit(self.run,'surface',[PY,REPO/'scripts/fit_shared_ground_surface.py','--input',original/'data/prepared/ground.npz','--output',original/'surface'],self.prep_env),
    pool.submit(self.run,'anchors',[PY,ROOT/'prepare_anchors_case.py',original]),
    pool.submit(self.script,'shadow_templates','shadow_templates.py',original),
    pool.submit(self.script,'road_masks',ADAPTER/'road_vla.py',original,self.gpu,env=self.env)]
   for f in fs:f.result()
  self.launch('original_validate',original,'validate')
 def configure(self,out):
  c=json.loads(TEMPLATE.read_text());c.update(source_path=str(out/'data/converted'),model_path=str(out/'full'),record_dir=str(out/'full/record'),point_cloud_dir=str(out/'full/point_cloud'),trained_model_dir=str(out/'full/trained_model'),full_background_source=str(self.work/'protected'))
  c['train'].update(checkpoint_iterations=[15000,30000],iterations=30000,save_iterations=[],start_checkpoint=None)
  c['train'].pop('keep_latest_only',None)
  # Preserve an existing scene's documented budget; unseen scenes use repository recipe default.
  import sys
  sys.path.insert(0,str(REPO/'scripts'))
  from run_ground_pipeline import DEFAULTS
  budget_table=json.loads((ROOT/'scene_budgets.json').read_text())
  clip=self.row['clip_name']
  budget=budget_table.get(clip,DEFAULTS['background_budget'])
  c['improvedgs']['background_budget']=budget
  c['workspace']=str(REPO)
  c['model']['ground']['sh_degree']=0
  c['model']['ground'].update(init=str(out/'surface/ground.npz'),background_ply=str(out/'data/prepared/nonground.ply'),masks=str(out/'data/ground_masks'),actor_ply_dir=str(out/'data/prepared/actors'))
  for name in ['config.json','config.draft.json']:write_json(out/name,c)
  write_json(out/'budget_source.json',dict(points=budget,source='scene_budgets.json' if clip in budget_table else 'run_ground_pipeline.DEFAULTS'))
 def launch(self,label,out,stage,*extra,config=None):
  if stage=='train' and not (self.work/'stages'/(label+'.json')).exists():
   resume_ckpt=out/'full/checkpoint_015000.pth'
   if resume_ckpt.is_file():extra=(*extra,'--resume',str(resume_ckpt))
  self.run(label,[PY,ENGINE/'launch.py','--config',config or out/'config.json','--output-root',out,'--gpu',self.gpu,'--allow-share','--stage',stage,*extra])
 def execute(self,prepare_only=False):
  self.prepare()
  if prepare_only:return
  w=self.work;original=w/'original';shifted=w/'shifted'

  self.launch('train_original',original,'train','--stop-after','30000')
  self.launch('evaluate_original',original,'evaluate','--checkpoint',original/'full/checkpoint_030000.pth','--no-images')
  self.script('canvas_acceptance',ADAPTER/'validate_vla.py',w)
  result=json.loads((original/'full/eval_checkpoint_030000/results.json').read_text())
  assert result['images']==707 and len(result['rows'])==707
  assert all((original/'full'/f'checkpoint_{i:06d}.pth').stat().st_size>0 for i in (15000,30000))
  write_json(w/'complete.json',dict(sample=self.row['sample_name'],clip=self.row['clip_name'],source=self.row['source'],finished=time.time(),iterations=30000,images=707,metrics=result['summary'],mesh_version='sam3-shared-vehicle-pose-1102',far_only=not self.row.get('no_far',False),original_only=True,vehicle_mesh=not self.row.get('no_mesh',False)))

