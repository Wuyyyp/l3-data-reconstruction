"""Both source conventions describe the same nonidentity calibrated camera."""
from pathlib import Path
import tempfile,json
import numpy as np
from PIL import Image
from lib.utils.m18_preprocess import prepare
R=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(dir=R/'output') as tmp:
 root=Path(tmp);E=np.eye(4);E[:3,:3]=[[0,-1,0],[1,0,0],[0,0,1]];E[:3,3]=[1.,2.,3.]
 outputs=[]
 for convention in ('ego_to_camera','camera_to_ego'):
  s=root/convention;s.mkdir()
  for name in ('images','intrinsics','extrinsics','track','lidar_depth'):(s/name).mkdir()
  Image.new('RGB',(16,16)).save(s/'images/000000_09.jpg')
  np.savetxt(s/'extrinsics/000000_09.txt',E if convention=='ego_to_camera' else np.linalg.inv(E))
  np.savetxt(s/'intrinsics/000000_09.txt',[[8,0,8],[0,8,8],[0,0,1]])
  points=np.array([[0.,0.,10.]],dtype='f4');np.savez(s/'pointcloud.npz',pointcloud={0:points},camera_projection={0:np.zeros((1,6))})
  mask=np.zeros((16,16),bool);mask[8,8]=True;np.save(s/'lidar_depth/000000_09.npy',dict(mask=mask,value=np.array([13.],dtype='f4')))
  (s/'track/track_info.txt').write_text('header\n0 1 car 0 2 2 2 0 0 10 0\n');(s/'track/track_camera_vis.json').write_text('{"1":{"0":[]}}')
  original=(s/'extrinsics/000000_09.txt').read_bytes()
  d=prepare(s,root/('out_'+convention),None,convention);outputs.append(d)
  assert np.allclose(np.loadtxt(d/'extrinsics/000000_09.txt'),np.linalg.inv(E))
  assert (s/'extrinsics/000000_09.txt').read_bytes()==original
 assert (outputs[0]/'track/track_camera_vis.json').read_bytes()==(outputs[1]/'track/track_camera_vis.json').read_bytes()
 assert json.loads((outputs[0]/'track/track_camera_vis.json').read_text())['1']['0']==[9]
print('PASS nonidentity source conventions, same camera visibility, source unchanged')
