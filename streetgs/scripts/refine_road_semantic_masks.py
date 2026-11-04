from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.ndimage import distance_transform_edt,minimum_filter,label,binary_fill_holes,binary_closing
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
from plyfile import PlyData

p=argparse.ArgumentParser(description='Refine road mask using local ground and obstacle geometry; compose paired dynamic masks.')
p.add_argument('--source',type=Path,required=True);p.add_argument('--semantic',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mesh-instance-masks',type=Path);p.add_argument('--old-ground-masks',type=Path);p.add_argument('--trial',action='store_true');args=p.parse_args()
S=args.source;A=args.semantic;O=args.output;O.mkdir(exist_ok=True)
for d in ['final_mask','masked_rgb','geometry_support','rejected_geometry','low_confidence','previews','dynamic_mask','combined_mask','combined_masked_rgb','old_combined_mask','old_combined_masked_rgb','mesh36_mask','combined_mesh36_mask','combined_mesh36_masked_rgb','comparison']:(O/d).mkdir(exist_ok=True)
m=json.loads((S/'input_manifest.json').read_text());origin=np.array(m['world_origin']);g=np.load(S/'prepared/ground.npz');xyz=g['xyz'].astype(float)+origin;normal=g['normals'];tree=cKDTree(xyz[:,:2])
bg=PlyData.read(S/'prepared/nonground.ply')['vertex'].data;bxyz=np.column_stack([bg[k] for k in 'xyz']).astype(float)+origin
bd,bi=tree.query(bxyz[:,:2],workers=4)
bheight=np.abs(((bxyz-xyz[bi])*normal[bi]).sum(-1))
bnon=(bd<2.0)&(bheight>.20)
def project(points,W,K,w,h,indices=False):
 q=points@W[:3,:3].T+W[:3,3];ids=np.flatnonzero(q[:,2]>.1);q=q[ids];uv=q@K.T;uv=uv[:,:2]/uv[:,2:];ok=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);uv=uv[ok].astype(int);q=q[ok];ids=ids[ok];pix=uv[:,1]*w+uv[:,0]
 z=np.full(w*h,np.inf,np.float32);np.minimum.at(z,pix,q[:,2]);z=z.reshape(h,w)
 if not indices:return z
 ix=np.full(w*h,-1,np.int32);order=np.argsort(q[:,2])[::-1];ix[pix[order]]=ids[order];return z,ix.reshape(h,w)
rows=[];started=time.time()
views=[v for v in m['views'] if not args.trial or v['name']=='000050_10']
def process(v):
 name=v['name'];w,h=v['width'],v['height'];C=np.array(v['c2w']);W=np.linalg.inv(C);K=np.array(v['K']);lw,lh=w//4,h//4;L=K.copy();L[0]*=lw/w;L[1]*=lh/h
 depth,idx=project(xyz,W,L,lw,lh,True)
 distance,nearest=distance_transform_edt(idx<0,return_indices=True);near=idx[tuple(nearest)];safe=near.clip(0)
 scene,si=project(bxyz,W,L,lw,lh,True)
 sd,snearest=distance_transform_edt(si<0,return_indices=True);si=si[tuple(snearest)]
 obstacle=(si>=0)&(sd<=2)&bnon[si.clip(0)]
 # Limit semantic extension locally without requiring a point at each pixel.
 support=(near>=0)&(distance<=40)&~obstacle
 def up(a):return np.array(Image.fromarray(a.astype('uint8')).resize((w,h),Image.Resampling.NEAREST)).astype(bool)
 support=up(support);obstacle=up(obstacle)
 semantic=np.array(Image.open(A/'semantic_road_mask'/f'{name}.png'))>0
 obstacle &= distance_transform_edt(semantic)<=40
 support=up((near>=0)&(distance<=40))&~obstacle
 candidate=semantic&support
 candidate=binary_fill_holes(candidate)&semantic
 candidate=binary_fill_holes(binary_closing(candidate,iterations=8))&semantic
 cc,n=label(candidate);counts=np.bincount(cc.ravel());seed=np.array(Image.open(A/'strict_mask'/f'{name}.png'))>0
 seeds=np.bincount(cc[seed],minlength=n+1);keep=(counts>=128)&(seeds>=32);keep[0]=False;final=keep[cc]
 rgb=np.array(Image.open(S/'converted/images'/f'{name}.jpg').convert('RGB'))
 valid=np.array(Image.open(S/'converted/valid_mask'/f'{name}.png').convert('L'))>0
 dynamic=np.array(Image.open(S/'converted/target_mask'/f'{name}.png').convert('L'))>0
 extra=S/'static_excluded'/f'{name}.png'
 if extra.exists():dynamic|=np.array(Image.open(extra).convert('L'))>0
 dynamic&=valid;final&=valid&~dynamic
 fid,cam=name.split('_');oldroot=args.old_ground_masks or S/'ground_masks'
 previous=np.load(oldroot/f'cam_{cam}'/f'{fid}.npy').astype(bool)&valid&~dynamic
 union=final|dynamic;old_union=previous|dynamic
 mesh=np.array(Image.open(args.mesh_instance_masks/f'{name}.png').convert('L'))>0 if args.mesh_instance_masks else dynamic.copy()
 mesh&=valid;mesh_union=final|mesh
 assert not (final&dynamic).any() and not (union&~valid).any() and np.array_equal(union&dynamic,old_union&dynamic)
 for folder,mm in [('dynamic_mask',dynamic),('combined_mask',union),('old_combined_mask',old_union),('mesh36_mask',mesh),('combined_mesh36_mask',mesh_union)]:Image.fromarray(mm.astype('u1')*255).save(O/folder/f'{name}.png')
 for folder,mm in [('combined_masked_rgb',union),('old_combined_masked_rgb',old_union),('combined_mesh36_masked_rgb',mesh_union)]:Image.fromarray(rgb*mm[:,:,None]).save(O/folder/f'{name}.jpg',quality=95)

 for folder,mask in [('final_mask',final),('geometry_support',support),('rejected_geometry',semantic&obstacle),('low_confidence',semantic&~final)]:Image.fromarray(mask.astype('uint8')*255).save(O/folder/f'{name}.png')
 Image.fromarray(rgb*final[:,:,None]).save(O/'masked_rgb'/f'{name}.png')
 old=np.array(Image.open(A/'final_mask'/f'{name}.png'))>0
 row=dict(name=name,width=w,height=h,old=int(old.sum()),final=int(final.sum()),rejected_obstacle=int((semantic&obstacle).sum()),low_confidence=int((semantic&~final).sum()),old_ground=int(previous.sum()),dynamic=int(dynamic.sum()),combined=int(union.sum()),old_combined=int(old_union.sum()),mesh36=int(mesh.sum()),combined_mesh36=int(mesh_union.sum()),road_added=int((final&~previous).sum()),road_removed=int((previous&~final).sum()))
 if v['frame'] in [0,50,100]:
  canvas=Image.new('RGB',(1920,450),'#181818')
  for j,(title,arr) in enumerate([('RGB',rgb),('V1: connected-region expansion',rgb*old[:,:,None]),('V2: local geometry boundary cleanup',rgb*final[:,:,None])]):
   im=Image.fromarray(arr);im.thumbnail((640,410));canvas.paste(im,(j*640,35));ImageDraw.Draw(canvas).text((j*640+8,10),title,fill='white')
  canvas.save(O/'previews'/f'{name}.jpg',quality=92)
  comp=Image.new('RGB',(1600,960),'#181818');dd=ImageDraw.Draw(comp)
  for j,(title,arr) in enumerate([('Full original RGB',rgb),('Old LiDAR ground + SAME dynamic mask',rgb*old_union[:,:,None]),('New semantic/local road + SAME dynamic mask',rgb*union[:,:,None]),('New road + mesh36 instance masks (different dynamic scope)',rgb*mesh_union[:,:,None])]):
   x=(j%2)*800;y=(j//2)*480;comp.paste(Image.fromarray(arr).resize((800,450)),(x,y+30));dd.text((x+8,y+8),name+' | '+title,fill='white')
  comp.save(O/'comparison'/f'{name}.jpg',quality=95)

 return row
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as pool:
 for row in pool.map(process,views):
  rows.append(row)
  if len(rows)%20==0 or len(rows)==len(views):
   status=dict(done=len(rows),total=len(views),seconds=time.time()-started,last=row);(O/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
report=dict(source=str(S),previous=str(A),views=len(rows),seconds=time.time()-started,method='Local projection neighborhood plus frontmost background points classified by height relative to nearest ground plane; remove boundary obstacles, close small gaps and fill interior holes within road semantic mask; require ground seeds',parameters=dict(geometry_grid_scale=4,background_ground_xy_radius_m=2.0,background_plane_height_m=.20,max_seed_pixel_distance=160,boundary_band_pixels=40,closing_iterations=8,minimum_component_pixels=128,minimum_seed_pixels=32),limitations=['Sparse geometry and approximate ego/dynamic masks can still leave boundary errors or reject valid road.','Local ground-plane approximation; not a ground-truth segmentation.'],rows=rows,totals={k:sum(r[k] for r in rows) for k in ['old','final','rejected_obstacle','low_confidence','old_ground','dynamic','combined','old_combined','mesh36','combined_mesh36','road_added','road_removed']})
(O/('trial_report.json' if args.trial else 'report.json')).write_text(json.dumps(report,indent=2));print('COMPLETE',flush=True)
