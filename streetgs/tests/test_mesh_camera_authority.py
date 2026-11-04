import csv,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('mesh_inputs',Path(__file__).resolve().parents[1]/'lib/preprocessing/ground_pipeline/mesh_inputs.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class CameraContract(unittest.TestCase):
 def test_exact_authority_and_reject_small_offset(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'poses.csv';fields=['camera','frame']+['T_cw_%d%d'%(i,j) for i in range(4) for j in range(4)]
   with p.open('w') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(dict(camera='left_front_camera',frame=1,**{'T_cw_%d%d'%(i,j):float(i==j) for i in range(4) for j in range(4)}))
   v=dict(name='000001_00',cam=0,frame=1,c2w=np.eye(4).tolist())
   self.assertEqual(m.verify_camera_authority([v],p)['max_absolute_error'],0.)
   v['c2w'][0][3]=.001
   with self.assertRaisesRegex(ValueError,'differs'):m.verify_camera_authority([v],p)
   v['frame']=0
   with self.assertRaisesRegex(ValueError,'absent'):m.verify_camera_authority([v],p)
 def test_origin_shift_preserves_affine_world_geometry(self):
  T=np.eye(4);T[:3,:3]=[[2,.1,0],[0,.5,0],[0,0,1.2]];T[:3,3]=[3,4,5]
  a=dict(id=1,frames=[1],transforms=[T.tolist()],centers=[T[:3,3].tolist()],rotations=[[1,0,0,0]],size=[4,2,1])
  a['view_transforms']={'000001_00':T.tolist()}
  b=m.shifted_actors([a],np.array([100,20,3]),np.array([90,10,1]),[1])[0]
  np.testing.assert_allclose(np.array(b['view_transforms']['000001_00'])[:3,3]+[90,10,1],T[:3,3]+[100,20,3])
  np.testing.assert_array_equal(np.array(b['transforms'])[0,:3,:3],T[:3,:3])
  np.testing.assert_allclose(np.array(b['centers'])[0]+[90,10,1],T[:3,3]+[100,20,3])
 def test_readable_parent_symlink(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'real').mkdir();(p/'real/x').write_text('test');(p/'link').symlink_to(p/'real',target_is_directory=True)
   self.assertEqual(m.readable(p/'link/x'),p/'real/x')
if __name__=='__main__':unittest.main()
