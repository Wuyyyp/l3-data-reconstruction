"""Opt-in background CPU payload loading. No CUDA or RNG work in workers."""
import concurrent.futures,queue,threading,time,copy
from collections.abc import Mapping
import numpy as np
import cv2
from PIL import Image
import torch

class DeferredGuidance(Mapping):
    def __init__(self, factories):self.factories=factories
    def __iter__(self):return iter(self.factories)
    def __len__(self):return len(self.factories)
    def __getitem__(self,key):return self.factories[key]()

def read_bool_mask(path):
    a=cv2.imread(path)
    if a is None:raise FileNotFoundError(path)
    return Image.fromarray(a[...,0]>0)

def read_depth(path):
    d=np.load(path,allow_pickle=True).item()
    result=np.zeros_like(d['mask']).astype(np.float32);result[d['mask']]=d['value'];return result

class BackgroundPayloads:
    def __init__(self, infos, scale, workers, budget_bytes):
        from lib.utils.camera_utils import prepareCam
        self.infos=infos;self.scale=scale;self.prepare=prepareCam
        # All loaded payloads remain CPU-cached for reuse. Refuse over-budget
        # datasets instead of silently allocating unbounded host memory.
        estimate=sum(int(c.width*min(1,1600/c.width)/scale)*int(c.height*min(1,1600/c.width)/scale)*24 for c in infos)
        if estimate>budget_bytes:raise ValueError('Streaming CPU payload estimate exceeds configured cache budget')
        self.estimated_bytes=estimate;self.futures=[concurrent.futures.Future() for _ in infos]
        self.queue=queue.PriorityQueue();self.lock=threading.Lock();self.claimed=set();self.finished=0;self.wait_seconds=0.;self.start=time.perf_counter();self.done_seconds=None
        for i in range(len(infos)):self.queue.put((1,i))
        self.threads=[];self.workers=max(1,workers);self.stopping=threading.Event()
        from lib.config import cfg
        if not cfg.data.get('streaming_defer_start',False):self.start_workers()
    def start_workers(self):
        if self.threads:return
        for _ in range(self.workers):
            t=threading.Thread(target=self._worker,daemon=True);t.start();self.threads.append(t)
    def close(self):
        self.stopping.set()
        for _ in self.threads:self.queue.put((-1,-1))
        for t in self.threads:t.join()
    def _worker(self):
        while not self.stopping.is_set():
            _,i=self.queue.get()
            if i==-1 or self.stopping.is_set():return
            with self.lock:
                if i in self.claimed:continue
                self.claimed.add(i)
            try:
                c=self.infos[i]
                # Open independently: no shared PIL file handles between scales.
                with Image.open(c.image_path) as image:
                    payload=self.prepare(c._replace(image=image,metadata=copy.deepcopy(c.metadata)),self.scale)
                self.futures[i].set_result((payload['image'].clamp(0,1),payload['guidance']))
            except BaseException as exc:self.futures[i].set_exception(exc)
            with self.lock:
                self.finished+=1
                if self.finished==len(self.infos):self.done_seconds=time.perf_counter()-self.start
    def get(self,i):
        # Prioritize the requested frame without changing training sampling.
        self.start_workers();self.queue.put((0,i));t=time.perf_counter();result=self.futures[i].result();self.wait_seconds+=time.perf_counter()-t;return result
    def stats(self):return dict(ready=self.finished,total=len(self.infos),wait_seconds=self.wait_seconds,done_seconds=self.done_seconds,estimated_bytes=self.estimated_bytes)

def make_streaming_cameras(infos,scale,workers,budget_bytes):
    from lib.utils.camera_utils import Camera
    class LazyCamera(Camera):
        def __getattribute__(self,name):
            if name in ('original_image','guidance'):
                state=object.__getattribute__(self,'__dict__')
                lazy=state.get('_payload_source')
                if lazy is not None:
                    loader,i=lazy;image,guidance=loader.get(i)
                    self.original_image=image;self.guidance=guidance;self._payload_source=None
            return super().__getattribute__(name)
    loader=BackgroundPayloads(infos,scale,workers,budget_bytes);cameras=[]
    for i,info in enumerate(infos):
        factor=min(1,1600/info.width)/scale;w,h=int(info.width*factor),int(info.height*factor);K=info.K.copy();K[:2]*=factor
        c=LazyCamera(id=info.uid,R=info.R,T=info.T,FoVx=info.FovX,FoVy=info.FovY,K=K,image=torch.empty((3,h,w),device='meta'),image_name=info.image_name,metadata=copy.deepcopy(info.metadata),guidance={})
        c._payload_source=(loader,i);c._payload_loader=loader;cameras.append(c)
    return cameras
