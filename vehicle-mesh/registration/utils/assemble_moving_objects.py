"""Export whole traffic participants: a rider and cycle share one instance ID."""
from __future__ import annotations
import argparse,hashlib,json,multiprocessing as mp,os,pickle,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import cv2,numpy as np
from motion_mask_utils import annotation_objects,annotation_world_tracks,camera_matrices,category_group,motion_decision,pack_mask,unpack_mask
from segment_moving_objects import atomic_json,encoder
from box_projection import project_track_box
from sam3_video import checkpoint

PROCESSOR=None
GPU=None
CYCLE_PROMPTS={'bicycle','motorcycle','electric scooter','electric bicycle','tricycle'}

def initialize(queue):
    global GPU
    GPU=queue.get();os.environ['CUDA_VISIBLE_DEVICES']=str(GPU)
    import torch
    torch.set_num_threads(4);cv2.setNumThreads(1)

def entity_type(category,prompt):
    group=category_group(category) if category else None
    if group=='two_wheeler' or prompt in CYCLE_PROMPTS:return 'rider_vehicle'
    if group=='person' or prompt=='person':return 'person'
    return 'vehicle'

def group_tracks(tracks,world,threshold):
    entities={};mapping={}
    for key,t in tracks.items():
        identity=t['motion'].get('annotation_id')
        name=f'obj_{identity}' if identity is not None else key
        category=world.get(identity,{}).get('category')
        kind=entity_type(category,t['prompt'])
        motion=motion_decision(world[identity]['positions'],threshold) if identity is not None else dict(t['motion'])
        if identity is not None:motion['annotation_id']=identity
        row=entities.setdefault(name,dict(entity_type=kind,prompt=kind,category=category,motion=motion,members=[],observations=[]))
        row['members'].append(key);mapping[key]=name
    # Source riders are independent of the mesh selection and may need recovery.
    for identity,t in world.items():
        if category_group(t['category'])!='two_wheeler':continue
        motion=motion_decision(t['positions'],threshold);motion['annotation_id']=identity
        if motion['state']=='moving':entities.setdefault(f'obj_{identity}',dict(entity_type='rider_vehicle',prompt='rider_vehicle',category=t['category'],motion=motion,members=[],observations=[]))
    return entities,mapping

def box_iou(mask,box):
    x,y,w,h=map(int,box);H,W=mask.shape;x0=max(0,x);y0=max(0,y);x1=min(W,x+w);y1=min(H,y+h)
    inter=int(mask[y0:y1,x0:x1].sum()) if x1>x0 and y1>y0 else 0
    return inter/max(1,int(mask.sum())+max(0,x1-x0)*max(0,y1-y0)-inter)

def select_mask(masks,scores,box,other_boxes,seed):
    best=None
    for mask,score in zip(masks,scores):
        if float(score)<.35 or mask.sum()<64:continue
        overlap=box_iou(mask,box)
        if overlap<.15 or any(box_iou(mask,b)>overlap+.05 for b in other_boxes):continue
        coverage=float((mask&seed).sum()/max(1,seed.sum()))
        rank=overlap+.05*coverage+.05*float(score)
        if best is None or rank>best[0]:best=(rank,mask,float(score))
    return best

def refine(frame,box,seed,category,other_boxes):
    global PROCESSOR
    x,y,w,h=map(int,box)
    if w*h<64 or min(w,h)<5:return None
    margin=max(20,round(max(w,h)*.15));H,W=frame.shape[:2]
    x0=max(0,x-margin);y0=max(0,y-margin);x1=min(W,x+w+margin);y1=min(H,y+h+margin)
    if x1<=x0 or y1<=y0:return None
    import torch
    from PIL import Image
    if PROCESSOR is None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        PROCESSOR=Sam3Processor(build_sam3_image_model(checkpoint_path=str(checkpoint()),load_from_HF=False),confidence_threshold=.35)
    word='tricycle' if category and 'tricycle' in category.lower() else 'bicycle' if category and 'bicycle' in category.lower() else 'motorcycle'
    prompts=[f'person riding a {word}',f'{word} and rider']
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=PROCESSOR.set_image(Image.fromarray(cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)))
        for prompt in prompts:
            PROCESSOR.reset_all_prompts(state);PROCESSOR.set_text_prompt(prompt,state)
            state=PROCESSOR.add_geometric_prompt([(x+w/2-x0)/(x1-x0),(y+h/2-y0)/(y1-y0),w/(x1-x0),h/(y1-y0)],True,state)
            masks=state['masks'].squeeze(1).cpu().numpy().astype(bool);scores=state['scores'].float().cpu().numpy()
            full=[]
            for mask in masks:
                image=np.zeros((H,W),bool);image[y0:y1,x0:x1]=mask;full.append(image)
            best=select_mask(full,scores,box,other_boxes,seed)
            if best is not None:return best[1],dict(source='SAM3 whole rider + cycle box/text prompt',text_prompt=prompt,confidence=best[2])
    return None

def merge_frame(parts,entities,mapping):
    merged={}
    for key,mask in parts.items():
        name=mapping[key]
        if entities[name]['motion']['state']!='moving':continue
        if name not in merged:merged[name]=mask.copy()
        else:merged[name]|=mask
    return merged

def remove_component_duplicates(instances,entities):
    # A confirmed whole cycle consumes contained unassociated person/cycle parts.
    cycles=[k for k in instances if entities[k]['entity_type']=='rider_vehicle']
    consumed={}
    for key in list(instances):
        row=entities[key]
        if row['motion'].get('annotation_id') is not None:continue
        for whole in cycles:
            if key==whole or whole not in instances:continue
            if entities[whole]['motion'].get('annotation_id') is None and row['entity_type']=='rider_vehicle' and key<whole:continue
            if (instances[key]&instances[whole]).sum()/max(1,instances[key].sum())>=.8:
                consumed[key]=whole;instances.pop(key);break
    return consumed

def reject_occluded_riders(instances,entities,boxes,depth_spans,static_names):
    removed={}
    for key in list(instances):
        if entities[key]['entity_type']!='rider_vehicle' or key not in depth_spans:continue
        mask=instances[key];area=int(mask.sum());H,W=mask.shape
        for static in static_names:
            if static not in boxes or static not in depth_spans:continue
            # A well-separated nearer static object can hide an entire projected
            # moving track; do not label the visible foreground silhouette as it.
            if depth_spans[static][1]+.3>=depth_spans[key][0]:continue
            x,y,w,h=map(int,boxes[static]);covered=mask[max(0,y):min(H,y+h),max(0,x):min(W,x+w)].sum()/max(1,area)
            if covered>=.85:removed[key]=static;instances.pop(key);break
    return removed

def cycle_geometry(info,camera,shape,world,threshold):
    K,C,T=camera_matrices(info,camera);E=np.linalg.inv(C)@T;boxes={};spans={};static=set()
    for identity,category,b in annotation_objects(info):
        if category_group(category)!='two_wheeler':continue
        key=f'obj_{identity}';p=project_track_box(length=b[3],width=b[4],height=b[5],center_xyz=b[:3],heading=b[6],intrinsic=K,extrinsic=E,image_width=shape[1],image_height=shape[0])
        if not p.visible_bbox_xywh:continue
        boxes[key]=p.visible_bbox_xywh;c,s=np.cos(b[6]),np.sin(b[6]);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        center=(E@np.r_[b[:3],1.])[2];extent=np.abs((E[:3,:3]@rotation)[2])@(b[3:6]/2);spans[key]=(float(center-extent),float(center+extent))
        if motion_decision(world[identity]['positions'],threshold)['state']=='static':static.add(key)
    return boxes,spans,static

def run_camera(payload):
    item,cfg=payload;root=Path(cfg['output']);raw=Path(cfg['candidates']);source=Path(cfg['input']);camera=item['camera'];shape=(item['height'],item['width'])
    marker=root/'masks'/camera/'assignments.json'
    if marker.exists():return json.load(marker.open())['summary']
    started=time.time();infos=pickle.load((source/'result_interpolation_optimized_first101.pkl').open('rb'))['infos'];world=annotation_world_tracks(infos)
    tracks={}
    for pi in range(len(cfg['prompts'])):tracks.update(json.load((raw/'candidates'/camera/f'p{pi}'/'tracks.json').open()))
    entities,mapping=group_tracks(tracks,world,cfg['displacement_m']);folder=marker.parent;folder.mkdir(parents=True,exist_ok=True)
    uniondir=root/'union-masks'/camera;uniondir.mkdir(parents=True,exist_ok=True);counts={};refined=0;fallback=0;consumed_count=0
    cap=cv2.VideoCapture(str(source/item['file']))
    with (root/'logs'/f'{camera}.log').open('w') as log:
        writers=[encoder(root/'videos'/f'{camera}.mp4',shape[1],shape[0],cfg['fps'],'gray',log),encoder(root/'overlays'/f'{camera}.mp4',shape[1],shape[0],cfg['fps'],'bgr24',log)]
        try:
            for fi,info in enumerate(infos):
                ok,frame=cap.read();assert ok,(camera,fi)
                parts={}
                for pi in range(len(cfg['prompts'])):
                    with np.load(raw/'candidates'/camera/f'p{pi}'/f'{fi:06d}.npz') as data:
                        for key in data.files:
                            if key.endswith('_bbox'):parts[key[:-5]]=unpack_mask(data[key],data[key[:-5]+'_bits'],shape)
                instances=merge_frame(parts,entities,mapping);boxes,spans,static=cycle_geometry(info,camera,shape,world,cfg['displacement_m'])
                evidence={}
                for name,row in entities.items():
                    if row['entity_type']!='rider_vehicle' or row['motion']['state']!='moving':continue
                    seed=instances.get(name,np.zeros(shape,bool));box=boxes.get(name)
                    if box is None:
                        if not seed.any():continue
                        if row['motion'].get('annotation_id') is not None:continue
                        box=pack_mask(seed)[0].tolist()
                        # Include a nearby unassociated upper-body candidate in a joint proposal.
                        x,y,w,h=box
                        for key,person in list(instances.items()):
                            pr=entities[key]
                            if pr['entity_type']!='person' or pr['motion'].get('annotation_id') is not None:continue
                            px,py,pw,ph=pack_mask(person)[0]
                            if x+.1*w<px+pw/2<x+.9*w and py+ph>=y-.1*h and py+ph/2<y+h/2:
                                seed=seed|person;box=pack_mask(seed)[0].tolist()
                    result=refine(frame,box,seed,row['category'],[b for k,b in boxes.items() if k!=name])
                    if result:
                        instances[name],evidence[name]=result;refined+=1
                    elif seed.any():instances[name]=seed;evidence[name]=dict(source='associated component union; whole refinement unavailable');fallback+=1
                reject_occluded_riders(instances,entities,boxes,spans,static)
                consumed=remove_component_duplicates(instances,entities);consumed_count+=len(consumed)
                arrays={};union=np.zeros(shape,bool);overlay=frame.copy()
                for key,mask in instances.items():
                    if not mask.any():continue
                    row=entities[key];bbox,bits=pack_mask(mask);arrays[key+'_bbox']=bbox;arrays[key+'_bits']=bits;union|=mask
                    kind=row['entity_type'];counts[kind]=counts.get(kind,0)+1
                    row['observations'].append(dict(frame=fi,pixels=int(mask.sum()),bbox=bbox.tolist(),**evidence.get(key,dict(source='merged SAM3 video candidates'))))
                    color=np.array({'rider_vehicle':[210,80,220],'person':[75,255,85],'vehicle':[41,190,255]}[kind]);overlay[mask]=(overlay[mask]*.5+color*.5).astype(np.uint8)
                np.savez_compressed(folder/f'{fi:06d}.npz',**arrays);assert cv2.imwrite(str(uniondir/f'{fi:06d}.png'),union.astype('uint8')*255)
                cv2.putText(overlay,f'WHOLE OBJECTS > {cfg["displacement_m"]:g} m | {camera} | {fi}',(12,28),0,.6,(0,255,255),2)
                writers[0].stdin.write((union.astype('uint8')*255).tobytes());writers[1].stdin.write(overlay.tobytes())
                if fi in {0,len(infos)//2,len(infos)-1}:cv2.imwrite(str(root/'review'/f'{camera}_{fi:06d}.jpg'),overlay)
                if fi%20==0:print('WHOLE',camera,fi,'refined',refined,'fallback',fallback,flush=True)
            for writer in writers:writer.stdin.close()
            for writer in writers:assert writer.wait()==0
        finally:
            cap.release()
            for writer in writers:
                if writer.poll() is None:writer.terminate();writer.wait()
    for key,row in entities.items():
        row['camera']=camera
        if row['motion']['state']!='moving':row['observations']=[dict(frame=fi) for fi in sorted({o['frame'] for m in row['members'] for o in tracks[m]['observations']})]
    entities={k:v for k,v in entities.items() if v['observations']}
    summary=dict(camera=camera,frames=len(infos),gpu=GPU,seconds=time.time()-started,instance_frames=sum(counts.values()),per_prompt_instance_frames=counts,track_states={s:sum(t['motion']['state']==s for t in entities.values()) for s in ['moving','static','uncertain']},whole_refined_instance_frames=refined,union_fallback_instance_frames=fallback,duplicate_components_consumed=consumed_count)
    atomic_json(marker,dict(summary=summary,tracks=entities));print('WHOLE CAMERA DONE',json.dumps(summary),flush=True);return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--candidates',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--gpus',type=int,nargs='+',required=True);a=parser.parse_args()
    raw=a.candidates.resolve();root=a.output.resolve();cfg=json.load((raw/'config.json').open());assert json.load((raw/'manifest.json').open())['complete']
    if raw==root or raw in root.parents:parser.error('whole output must be separate from candidates')
    cfg.update(candidates=str(raw),output=str(root),entity_mode='whole_rider_vehicle_v1')
    if (root/'config.json').exists():assert json.load((root/'config.json').open())==cfg,'use a fresh output for changed inputs'
    elif root.exists() and any(root.iterdir()):parser.error('output is not empty')
    for name in ['logs','review','masks','union-masks','videos','overlays']:(root/name).mkdir(parents=True,exist_ok=True)
    atomic_json(root/'config.json',cfg);started=time.time();context=mp.get_context('spawn')
    with context.Manager() as manager:
        queue=manager.Queue()
        for gpu in a.gpus:queue.put(gpu)
        with ProcessPoolExecutor(max_workers=len(a.gpus),mp_context=context,initializer=initialize,initargs=(queue,)) as pool:results=list(pool.map(run_camera,[(v,cfg) for v in cfg['manifest']['videos']]))
    atomic_json(root/'manifest.json',dict(complete=True,mode='whole_actual_motion',threshold_m=cfg['displacement_m'],frames=cfg['frames'],fps=cfg['fps'],prompts=cfg['prompts'],cameras=results,seconds=time.time()-started,identity_semantics='One instance per whole rider + cycle; independent pedestrians remain persons; matched source IDs are shared across prompts.',motion_rule='max pairwise world XYZ distance > threshold',candidates=str(raw),selected_vehicle_masks_unchanged=True))
    print('WHOLE MOVING MASKS COMPLETE',flush=True)

if __name__=='__main__':main()
