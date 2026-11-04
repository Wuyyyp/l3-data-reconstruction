"""Bounded CPU preparation and opt-in camera transfer; preserve sampling order."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque, OrderedDict
import copy
import torch

def ordered_prepare(fn, items, workers):
    """CPU-only fn; ordered delivery, at most 2*workers tasks in flight."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        source=iter(items);pending=deque()
        for _ in range(2*workers):
            try:pending.append(pool.submit(fn,next(source)))
            except StopIteration:break
        while pending:
            yield pending.popleft().result()
            try:pending.append(pool.submit(fn,next(source)))
            except StopIteration:pass

class CameraTransfer:
    """One-camera GPU prefetch with a bounded pinned CPU tensor cache.

    Predictions never consume RNG. A wrong prediction falls back to the actual
    selected camera. Cached tensors are snapshots; call invalidate after edits.
    Camera transforms/learnable metadata are never cached or replaced.
    """
    def __init__(self, max_pinned_bytes=512*1024**2):
        self.stream=torch.cuda.Stream()
        self.cache=OrderedDict();self.bytes=0;self.limit=max_pinned_bytes
        self.pending=None;self.hits=0;self.misses=0
    def invalidate(self):
        self.stream.synchronize();self.pending=None;self.cache.clear();self.bytes=0
    def _host(self, camera):
        key=id(camera)
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);return self.cache[key]
        self.misses+=1
        items={'original_image':camera.original_image}
        items.update({'guidance/'+k:v for k,v in camera.guidance.items() if torch.is_tensor(v)})
        host={k:v.detach().pin_memory() if v.device.type=='cpu' and not v.is_pinned() else v for k,v in items.items()}
        size=sum(t.numel()*t.element_size() for t in host.values() if t.device.type=='cpu')
        while self.cache and self.bytes+size>self.limit:
            _,old=self.cache.popitem(last=False);self.bytes-=sum(t.numel()*t.element_size() for t in old.values() if t.device.type=='cpu')
        if size<=self.limit:self.cache[key]=host;self.bytes+=size
        return host
    def prefetch(self,camera):
        if self.pending is not None:
            if self.pending[0] is camera:return
            # Bound live buffers and retain host source until DMA completes.
            self.pending[3].synchronize();self.pending=None
        host=self._host(camera)
        with torch.cuda.stream(self.stream):
            gpu={k:v.to('cuda',non_blocking=True) for k,v in host.items()}
            done=torch.cuda.Event();done.record(self.stream)
        self.pending=(camera,host,gpu,done)
    def acquire(self,camera):
        self.prefetch(camera)
        _,host,gpu,done=self.pending
        torch.cuda.current_stream().wait_event(done)
        result=copy.copy(camera);result.guidance=dict(camera.guidance)
        for k,v in gpu.items():
            v.record_stream(torch.cuda.current_stream())
            if k=='original_image':result.original_image=v
            else:result.guidance[k[len('guidance/'):]]=v
        # Synchronizing CPU here is unnecessary: pinned allocator tracks DMA;
        # cache or this pending host retains buffers until the next prefetch.
        self.pending=None
        return result
