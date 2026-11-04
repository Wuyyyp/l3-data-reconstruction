import torch,itertools
from mesh_backend import camera_matrices
@torch.no_grad()
def exclusion(scene,c):
 if c.image_name in scene.early_masks:return scene.early_masks[c.image_name].cuda()
 import bridge as br
 out=torch.zeros((1,c.image_height,c.image_width),device='cuda',dtype=torch.bool)
 corners=torch.tensor(list(itertools.product([-.5,.5],repeat=3)),device='cuda')
 vm,K=camera_matrices(c,corners)
 for a in br._ACTORS:
  if a['id'] not in scene.native_ids:continue
  key=br.affine_key(a['id'],c)
  if key not in br._AFFINE:continue
  A,t=br._AFFINE[key];meta=a['native_meta'];size=corners.new_tensor([meta[k] for k in ['length','width','height']])
  # _AFFINE maps canonical physical object coordinates to the training world.
  xyz=(corners*size)@A.T+t;q=xyz@vm[0,:3,:3].T+vm[0,:3,3]
  points=[q[q[:,2]>=.2]]
  for i in range(8):
   for j in range(i+1,8):
    if int((corners[i]!=corners[j]).sum())!=1:continue
    if bool((q[i,2]<.2)!=(q[j,2]<.2)):points.append((q[i]+(.2-q[i,2])/(q[j,2]-q[i,2])*(q[j]-q[i]))[None])
  q=torch.cat(points)
  if not len(q):continue
  uv=q@K[0].T;uv=uv[:,:2]/uv[:,2:];lo=uv.min(0).values.floor().long();hi=uv.max(0).values.ceil().long()
  x0,y0=lo.tolist();x1,y1=hi.tolist();x0=max(0,x0);y0=max(0,y0);x1=min(c.image_width,x1+1);y1=min(c.image_height,y1+1)
  if x1>x0 and y1>y0:out[:,y0:y1,x0:x1]=True
 scene.early_masks[c.image_name]=out.cpu();return out
