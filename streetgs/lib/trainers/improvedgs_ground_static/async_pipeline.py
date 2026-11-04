"""Fresh-run masks-only preload, CPU RGB/depth streaming and bounded CUDA prefetch."""
import os,time,queue,threading,concurrent.futures,copy,random,atexit
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from lib.config import cfg
from lib.utils import streaming_data as sd
from lib.utils.camera_utils import Camera,loadguidance,prepareCam
from collections.abc import Mapping
class Guide(Mapping):
 def __init__(self,masks,loader,i):self.masks=masks;self.loader=loader;self.i=i
 def __iter__(self):return iter(list(self.masks)+(['lidar_depth','mesh_depth'] if 'lidar_depth' in self.loader.infos[self.i].guidance else []))
 def __len__(self):return len(list(iter(self)))
 def __getitem__(self,k):return self.masks[k] if k in self.masks else self.loader.get(self.i)[1][k]
class Loader(sd.BackgroundPayloads):
 def _worker(self):
  while not self.stopping.is_set():
   _,i=self.queue.get()
   if i==-1 or self.stopping.is_set():return
   with self.lock:
    if i in self.claimed:continue
    self.claimed.add(i)
   try:
    c=self.infos[i];base=Path(cfg.source_path);a=np.load(base/'lidar_depth'/(c.image_name+'.npy'),allow_pickle=True).item();depth=np.zeros((c.height,c.width),np.float32);depth[a['mask']]=a['value'];guid={'lidar_depth':depth}
    with Image.open(c.image_path) as im:p=prepareCam(c._replace(image=im,guidance=guid,metadata=copy.deepcopy(c.metadata)),self.scale)
    p['guidance'].update(self.masks[i]);p['guidance']['mesh_depth']=torch.zeros((1,c.height,c.width),dtype=torch.float32);self.futures[i].set_result((p['image'].clamp(0,1),p['guidance']))
   except BaseException as e:self.futures[i].set_exception(e)
   with self.lock:
    self.finished+=1
    if self.finished==len(self.infos):self.done_seconds=time.perf_counter()-self.start
class AsyncCamera(Camera):
 def __getattribute__(self,name):
  if name in ('original_image','guidance'):
   d=object.__getattribute__(self,'__dict__');gpu=d.get('_gpu_payload')
   if gpu is not None:return gpu[0 if name=='original_image' else 1]
   source=d.get('_cpu_source')
   if source is not None:
    loader,i=source
    return loader.get(i)[0] if name=='original_image' else d['_cpu_guide']
  return super().__getattribute__(name)
def cpu_payload(c):
 source=c.__dict__.get('_cpu_source')
 return source[0].get(source[1]) if source is not None else (c.original_image,c.guidance)
def masks_for_sky(c):
 return c.__dict__.get('_mask_preload',c.guidance)
def make_cameras(infos,scale,workers,budget):
 t=time.perf_counter()
 def mask_one(info):
  f=min(1,1600/info.width)/scale;res=(int(info.width*f),int(info.height*f))
  return loadguidance({k:info.guidance[k] for k in info.guidance if k!='lidar_depth'},res)
 with concurrent.futures.ThreadPoolExecutor(workers) as pool:masks=list(pool.map(mask_one,infos))
 cfg.data.streaming_defer_start=True;loader=Loader(infos,scale,workers,budget);loader.masks=masks;loader.mask_preload_seconds=time.perf_counter()-t;cams=[]
 for i,info in enumerate(infos):
  f=min(1,1600/info.width)/scale;w,h=int(info.width*f),int(info.height*f);K=info.K.copy();K[:2]*=f
  c=AsyncCamera(id=info.uid,R=info.R,T=info.T,FoVx=info.FovX,FoVy=info.FovY,K=K,image=torch.empty((3,h,w),device='meta'),image_name=info.image_name,metadata=copy.deepcopy(info.metadata),guidance={})
  c._cpu_source=(loader,i);c._cpu_guide=Guide(masks[i],loader,i);c._mask_preload=masks[i];c._payload_loader=loader;cams.append(c)
 atexit.register(loader.close)
 return cams
sd.make_streaming_cameras=make_cameras
class Prefetch:
 def __init__(self,cams):
  self.enabled=bool(cfg.data.get('streaming_load',False)) and bool(cfg.data.get('cuda_prefetch',True));self.cams=cams;self.futures={};self.current=None;self.wait_seconds=0.;self.bytes=0;self.copies=0;self.exact_payload_checks=0
  self.executor=concurrent.futures.ThreadPoolExecutor(1) if self.enabled else None
  self.stream=torch.cuda.Stream() if self.enabled else None
  if self.enabled:atexit.register(self.close)
 def _copy(self,c):
  im,guide=cpu_payload(c);host=[im.pin_memory(),{k:v.pin_memory() for k,v in guide.items()}]
  with torch.cuda.stream(self.stream):
   gpu=[host[0].to('cuda',non_blocking=True),{k:v.to('cuda',non_blocking=True) for k,v in host[1].items()}];event=torch.cuda.Event();event.record(self.stream)
  self.copies+=1;self.bytes+=sum(v.numel()*v.element_size() for v in [host[0],*host[1].values()]);return gpu,event,host
 def submit(self,c):
  if self.enabled and c.image_name not in self.futures:
   assert len(self.futures)<2,'Prefetch queue exceeded two slots';self.futures[c.image_name]=self.executor.submit(self._copy,c)
 def activate(self,c):
  if not self.enabled:return
  if self.current is not None:self.current.__dict__.pop('_gpu_payload',None)
  self.submit(c);t=time.perf_counter();gpu,event,host=self.futures.pop(c.image_name).result();torch.cuda.current_stream().wait_event(event)
  for v in [gpu[0],*gpu[1].values()]:v.record_stream(torch.cuda.current_stream())
  # Keep pinned buffers alive until the non-blocking copies complete.
  if cfg.data.get('audit_gpu_payload',False):
   assert torch.equal(gpu[0].cpu(),host[0])
   for k in gpu[1]:assert torch.equal(gpu[1][k].cpu(),host[1][k]),k
   self.exact_payload_checks+=1
  self.host=host;self.event=event;c._gpu_payload=gpu;self.current=c;self.wait_seconds+=time.perf_counter()-t
 def next(self,rng,stack,cams,indices=False):
  if not self.enabled:return
  r=random.Random();r.setstate(rng.getstate());s=stack if stack else (list(range(len(cams))) if indices else cams);item=s[r.randrange(len(s))];self.submit(cams[item] if indices else item)
 def close(self):
  if self.executor is None:return
  self.executor.shutdown(wait=True);self.stream.synchronize()
  if self.current is not None:self.current.__dict__.pop('_gpu_payload',None)
  self.futures.clear();self.executor=None
 def stats(self):return dict(enabled=self.enabled,exact_payload_checks=self.exact_payload_checks,copies=self.copies,bytes=self.bytes,host_wait_seconds=self.wait_seconds,pinned_slots_max=2)
