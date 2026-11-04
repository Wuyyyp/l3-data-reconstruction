from pathlib import Path
import numpy as np,json,tempfile,sys
from PIL import Image
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from lib.utils.m18_preprocess import prepare
with tempfile.TemporaryDirectory(dir=R/'cache') as tmp:
 root=Path(tmp);s=root/'source';s.mkdir()
 for n in ['images','extrinsics','intrinsics','track','lidar_depth']: (s/n).mkdir()
 Image.new('RGB',(8,8)).save(s/'images/000000_09.jpg');np.savetxt(s/'extrinsics/000000_09.txt',np.eye(4));K=np.array([[8,0,4],[0,8,4],[0,0,1]]);np.savetxt(s/'intrinsics/000000_09.txt',K)
 pts=np.array([[0,0,1],[0,0,10]],dtype='f4');np.savez(s/'pointcloud.npz',pointcloud={0:pts},camera_projection={0:np.array([[9,4,4,9,4,4],[9,4,4,9,4,4]])})
 mask=np.zeros((8,8),bool);mask[4,4]=True;np.save(s/'lidar_depth/000000_09.npy',dict(mask=mask,value=np.array([1.],dtype='f4')))
 (s/'track/track_info.txt').write_text('frame_id track_id object_class alpha h w l x y z yaw speed\n0 2 Car -10 2 2 2 0 0 1 0 2\n');(s/'track/track_camera_vis.json').write_text('{"2":{"0":[]}}')
 t=root/'template.npz';np.savez(t,seed_ego=pts[:1],active_cells=np.array([100*1000000+100*1000+110]),protect_track_boxes=np.array(True),cell_only=np.array(True))
 out=prepare(s,root/'run',t,'ego_to_camera');assert json.loads((out/'track/track_camera_vis.json').read_text())['2']['0']==[9]
 assert np.array_equal(np.load(out/'pointcloud.npz',allow_pickle=True)['pointcloud'].item()[0],pts);assert np.load(out/'lidar_depth/000000_09.npy',allow_pickle=True).item()['mask'].any()
 assert len(np.load(s/'pointcloud.npz',allow_pickle=True)['pointcloud'].item()[0])==2
 try:prepare(s,root/'run',t,'ego_to_camera')
 except AssertionError:pass
 else:raise AssertionError('Existing run reused')
print('PASS protected-box case: point/projection alignment, depth filtering, empty info2d recovery, source immutability and no cross-run reuse')

