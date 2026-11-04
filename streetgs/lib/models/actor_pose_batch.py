"""Batch actor interpolation; caches immutable lookup tensors, never trainable poses.

Preserves the scalar path's rotation indexing/composition semantics intentionally.
"""
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import roma
from lib.config import cfg
from lib.utils.general_utils import quaternion_raw_multiply_theta

def _slerp(q0, q1, step):
    q0 = F.normalize(q0, dim=-1)[..., [1, 2, 3, 0]]
    q1 = F.normalize(q1, dim=-1)[..., [1, 2, 3, 0]]
    if cfg.mode == 'train':
        rel = roma.utils.quat_product(roma.utils.quat_conjugation(q0), q1)
        rv = roma.mappings.unitquat_to_rotvec(rel, shortest_arc=True)
        rot = roma.mappings.rotvec_to_unitquat(step[:, None] * rv)
        q = roma.utils.quat_product(q0, rot)
    else:
        cos = (q0 * q1).sum(-1)
        q1 = torch.where((cos < 0)[:, None], -q1, q1)
        cos = cos.abs()
        near = cos > 1 - 1e-3
        omega = torch.acos(cos)
        alpha = torch.where(near, 1-step, torch.sin((1-step)*omega))
        beta = torch.where(near, step, torch.sin(step*omega))
        q = roma.utils.quat_normalize(alpha[:, None]*q0 + beta[:, None]*q1)
    return q[..., [3, 0, 1, 2]]

def _lookup(pose, track_ids, times):
    if not hasattr(pose, '_batch_lookup'):pose._batch_lookup=OrderedDict()
    key=(tuple(track_ids),tuple(float(t) for t in times),str(pose.input_trans.device))
    cache=pose._batch_lookup
    if key not in cache:
        rows=[];weights=[]
        for tid,t in zip(track_ids,times):
            # CPU metadata lookup; same nearest-pair/tie ordering as scalar path.
            if not hasattr(pose,'_timing_cpu'):pose.clear_timing_cache()
            if tid not in pose._timing_cpu:
                ix=pose.obj_info[tid]['track_idx'].detach().cpu().numpy()
                pose._timing_cpu[tid]=(ix,np.asarray(pose.timestamps)[ix[:,0]])
            ix,ts=pose._timing_cpu[tid];nearest=np.argsort(np.abs(ts-t))[:2]
            assert len(nearest)==2
            (f1,c1),(f2,c2)=ix[nearest];t1,t2=ts[nearest]
            rows.append([f1,c1,f2,c2]);weights.append([t2-t,t-t1,t2-t1,(t-t1)/(t2-t1)])
        device=pose.input_trans.device
        cache[key]=(torch.tensor(rows,device=device,dtype=torch.long).unbind(-1),torch.tensor(weights,device=device,dtype=pose.input_trans.dtype).unbind(-1))
        if len(cache)>4096:cache.popitem(last=False)
    cache.move_to_end(key)
    return cache[key]

def _at(pose, ids, times):
    (f1,c1,f2,c2),(w1,w2,den,r)=_lookup(pose,ids,times)
    t1,t2=pose.input_trans[f1,c1],pose.input_trans[f2,c2]
    q1,q2=pose.input_rots[f1,c1],pose.input_rots[f2,c2]
    if pose.opt_track:
        t1=t1+pose.opt_trans[f1,c1];t2=t2+pose.opt_trans[f2,c2]
        # Scalar helper takes theta shape (1,) and returns (1,4); batched theta is (N,).
        q1=quaternion_raw_multiply_theta(q1,pose.opt_rots[f1,c1,0])
        q2=quaternion_raw_multiply_theta(q1,pose.opt_rots[f1,c2,0])
    trans=(t1*w1[:,None]+t2*w2[:,None])*den.reciprocal()[:,None]
    return _slerp(q1,q2,r),trans

def get_actor_poses(pose, track_ids, camera):
    t=camera.meta['timestamp']
    if pose.opt_track and camera.meta['is_val']:
        # Preserve double interpolation for held-out camera timestamps.
        q=[];tr=[]
        for tid in track_ids:
            a,b=pose.find_closest_camera_timestamps(tid,camera)
            if a is None:rot,pos=_at(pose,[tid],[t])
            else:
                rs,ts=_at(pose,[tid,tid],[a,b]);w=(t-a)/(b-a)
                rot=_slerp(rs[:1],rs[1:],rs.new_tensor([w]))
                pos=(ts[:1]*(b-t)+ts[1:]*(t-a))/(b-a)
            q.append(rot);tr.append(pos)
        return torch.cat(q),torch.cat(tr)
    return _at(pose,track_ids,[t]*len(track_ids))
