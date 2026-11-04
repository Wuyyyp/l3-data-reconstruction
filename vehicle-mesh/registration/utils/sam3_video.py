"""Minimal SAM3 video predictor adapter used by this standalone task."""
from pathlib import Path
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

def checkpoint():
    value=os.environ.get('SAM3_CHECKPOINT')
    if not value or not Path(value).is_file():raise FileNotFoundError('Set SAM3_CHECKPOINT to a local SAM3 checkpoint file')
    return Path(value).resolve()

def match_mask_indices(masks,boxes,min_iou):
    if not len(masks) or not boxes:return {}
    h,w=masks.shape[1:];areas=masks.sum((1,2));scores=np.zeros((len(boxes),len(masks)),np.float32)
    for ri,(_, (x,y,bw,bh)) in enumerate(boxes):
        x0,y0=max(0,x),max(0,y);x1,y1=min(w,x+bw),min(h,y+bh)
        if x1<=x0 or y1<=y0:continue
        intersections=masks[:,y0:y1,x0:x1].sum((1,2));unions=areas+(x1-x0)*(y1-y0)-intersections
        scores[ri]=np.divide(intersections,unions,out=np.zeros(len(masks)),where=unions>0)
    rows,cols=linear_sum_assignment(-scores)
    return {boxes[r][0]:int(c) for r,c in zip(rows,cols) if scores[r,c]>=min_iou}

class Sam3VideoTracker:
    def __init__(self):self.predictor=None
    def infer(self,video_path,checkpoint_path,prompt,frame_count,shape,*,frame_consumer=None):
        from pose_mesh_overlap import replay
        if replay(video_path,checkpoint_path,prompt,frame_count,shape,frame_consumer):return
        if self.predictor is None:
            from sam3.model_builder import build_sam3_video_predictor
            self.predictor=build_sam3_video_predictor(checkpoint_path=str(checkpoint_path),gpus_to_use=[0],video_loader_type='cv2',compile=False)
        sid=None;seen=set()
        try:
            sid=self.predictor.handle_request({'type':'start_session','resource_path':str(video_path),'offload_video_to_cpu':True,'offload_state_to_cpu':True})['session_id']
            self.predictor.handle_request({'type':'add_prompt','session_id':sid,'frame_index':0,'text':prompt})
            req={'type':'propagate_in_video','session_id':sid,'propagation_direction':'forward','start_frame_index':0,'max_frame_num_to_track':frame_count,'evict_cached_frame_outputs':True}
            for response in self.predictor.handle_stream_request(req):
                i=int(response['frame_index'])
                if i in seen or not 0<=i<frame_count:raise ValueError(f'SAM3 returned invalid frame {i}')
                seen.add(i);frame_consumer(i,response['outputs'])
            if len(seen)!=frame_count:raise ValueError(f'SAM3 returned {len(seen)} of {frame_count} frames')
        finally:
            if sid is not None:self.predictor.handle_request({'type':'close_session','session_id':sid})
    def close(self):
        if self.predictor is not None:self.predictor.shutdown();self.predictor=None
