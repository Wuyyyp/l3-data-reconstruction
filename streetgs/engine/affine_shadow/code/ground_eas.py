"""ImprovedGS EAS adapted to fixed-normal tangent ground splitting.
Uses the existing Laplacian operator and occlusion-weighted rasterizer scores.
Does not change RGB loss, gradient threshold, split geometry, or actor policy.
"""
import json,time,random
import torch
import torch.nn.functional as F
from scene.methods.densification_methods import compute_edge_map,normalize_to_unit_range

def select_candidates(ids,gradient,importance,k,iteration):
    assert importance.shape==gradient.shape and torch.isfinite(importance).all()
    weight=importance[ids].clamp_min(0)
    positive=weight>0
    gen=torch.Generator(device=ids.device).manual_seed(905300+iteration)
    count=min(k,int(positive.sum()))
    selected=ids[torch.multinomial(weight,count,replacement=False,generator=gen)] if count else ids[:0]
    # Sampled views may not see all eligible points. Explicit gradient fallback
    # fills the unchanged per-event budget, rather than starving unseen areas.
    if count<k:
        remaining=ids[~torch.isin(ids,selected)]
        selected=torch.cat([selected,remaining[gradient[remaining].topk(k-count).indices]])
    return selected,count

@torch.no_grad()
def prepare(scene,current,pipe,background,it,smoke):
    due=(it==3) if smoke else (500<=it<=10000 and it%100==0)
    if not due:return
    from bridge import cpu_payload
    begin=time.perf_counter()
    views=[current]
    others=[c for c in scene.cameras if c.uid!=current.uid]
    views+=random.Random(817000+it).sample(others,min(9,len(others)))
    scores=torch.zeros(len(scene.ground.initial),device='cuda')
    edge_pixels=0
    for c in views:
        im,_=cpu_payload(c)
        mask=scene.ground_mask(c).bool()
        # Edge operator reads original RGB; remove 1px invalid-mask neighbourhood.
        interior=F.max_pool2d(F.pad((~mask).float()[None],(1,1,1,1),value=1),3,1)[0,0]==0
        edge=compute_edge_map(im.cuda())*interior
        edge_pixels+=int((edge>0).sum())
        scene.ground_eas_scoring=True
        try:pkg=scene.render(c,scene.gaussians,pipe,background,track_gradients=False,pixel_weights=edge)
        finally:scene.ground_eas_scoring=False
        n=len(scores);s=pkg['accum_weights'][:n]
        assert s.shape==scores.shape and torch.isfinite(s).all() and (s>=0).all()
        scores+=normalize_to_unit_range(s)/len(views)
        if c.image_name=='000000_00' and not (scene.audit_out/'edge_000000_00.png').exists():
            from PIL import Image
            Image.fromarray((edge.cpu().numpy()*255).astype('uint8')).save(scene.audit_out/'edge_000000_00.png')
    scene.ground._eas_score=scores
    scene.ground._eas_audit=dict(iteration=it,views=[c.image_name for c in views],positive_ground=int((scores>0).sum()),edge_pixels=edge_pixels,score_max=float(scores.max()),score_seconds=time.perf_counter()-begin)
    assert edge_pixels>0 and (scores>0).any(),'No valid ground edge contribution'
