"""Write separate, standard 3DGS PLY layers in their model coordinate frames."""
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement

def save_ground_ply(ground, path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    def cpu(x):return x.detach().cpu().float().numpy()
    xyz=cpu(ground.get_xyz)
    arrays=[xyz,np.zeros_like(xyz),cpu(ground.get_features_dc.transpose(1,2)).reshape(len(xyz),-1),
            cpu(ground.get_features_rest.transpose(1,2)).reshape(len(xyz),-1),
            cpu(ground.opacity),np.log(cpu(ground.get_scaling)),cpu(ground.get_rotation)]
    names=['x','y','z','nx','ny','nz']
    names += ['f_dc_'+str(i) for i in range(arrays[2].shape[1])]
    names += ['f_rest_'+str(i) for i in range(arrays[3].shape[1])]
    names += ['opacity']+['scale_'+str(i) for i in range(3)]+['rot_'+str(i) for i in range(4)]
    data=np.concatenate(arrays,axis=1)
    vertices=np.empty(len(xyz),dtype=[(n,'f4') for n in names])
    for i,name in enumerate(names):vertices[name]=data[:,i]
    PlyData([PlyElement.describe(vertices,'vertex')],text=False).write(str(path))
