"""Buffered scalar telemetry: one host transfer per interval, optional CUDA timing."""
import json
import torch

class Metrics:
    def __init__(self,stream,interval=100):
        self.stream=stream;self.interval=max(1,int(interval));self.pending=[]
    def add(self,row,events=None):
        self.pending.append((row,events))
        if len(self.pending)>=self.interval:self.flush()
    def flush(self):
        if not self.pending:return
        tensors=[];slots=[]
        for row,events in self.pending:
            for key,value in row.items():
                if torch.is_tensor(value):
                    slots.append((row,key));tensors.append(value.detach().float().reshape(()))
        if tensors:
            for (row,key),value in zip(slots,torch.stack(tensors).cpu().tolist()):
                row[key]=int(value) if key.endswith('_pixels') else value
        # CUDA timestamps are read only after the bounded batch transfer has completed.
        if any(events is not None for row,events in self.pending):
            torch.cuda.current_stream().synchronize()
        for row,events in self.pending:
            if events is not None:
                row['train_ms']=events[0].elapsed_time(events[-1])
                row['cuda_timeline_ms']=[events[j].elapsed_time(events[j+1]) for j in range(len(events)-1)]
            self.stream.write(json.dumps(row)+'\n')
        self.stream.flush();self.pending.clear()
