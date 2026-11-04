"""Create a fresh read-only-source adapter; regenerate ego poses from optimized delivery."""
import argparse,json,pickle,sys,csv
from pathlib import Path
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'lib/preprocessing/ground_pipeline'))
from mesh_inputs import bundle,readable,CAMERA_NAMES
class U(pickle.Unpickler):
 def find_class(self,m,n):return super().find_class(m.replace('numpy._core','numpy.core'),n)
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['mesh-source','optimized-delivery','lidar-converted','output']:p.add_argument('--'+key,required=True)
 p.add_argument('--frames',required=True,help='Explicit ids, e.g. 1,2,4 or 0:100 inclusive')
 p.add_argument('--allow-first-frame-pkl',action='store_true',help='Explicitly use optimized PKL rig pose only for missing frame zero; recorded in audit')
 a=p.parse_args();frames=[]
 for part in a.frames.split(','):
  frames.extend(range(int(part.split(':')[0]),int(part.split(':')[1])+1) if ':' in part else [int(part)])
 frames=sorted(set(frames));out=Path(a.output).absolute()
 if not out.resolve().is_relative_to(R/'output'):raise ValueError('Only repo output allowed')
 if out.exists():raise FileExistsError(out)
 source=Path(a.lidar_converted);delivery=Path(a.optimized_delivery)
 report=json.loads(readable(source/'conversion.complete.json').read_text())
 field='ego2global_transformation_matrix_camera_reference_offset_optimized'
 if report['pose_field']!=field or readable(report['source_pkl'])!=readable(delivery/'result_interpolation_optimized.pkl'):raise ValueError('LiDAR conversion not from requested optimized delivery')
 with readable(delivery/'result_interpolation_optimized.pkl').open('rb') as f:infos=U(f).load()['infos']
 mapping={x['output_frame']:x['source_frame'] for x in report['frame_mapping']}
 out.mkdir(parents=True);(out/'ego_pose').mkdir()
 with readable(delivery/'image_extrinsics/image_camera_to_world_after.csv').open() as f:authority_rows=list(csv.DictReader(f))
 keys={(int(v['frame']),v['camera']) for v in authority_rows};fallbacks=[]
 manifest=json.loads(readable(Path(a.mesh_source)/'input_manifest.json').read_text())
 fields=['camera','frame']+['T_cw_%d%d'%(i,j) for i in range(4) for j in range(4)]
 for v in manifest['views']:
  if v['frame'] not in frames:continue
  frame=v.get('original_frame',v['frame']);camera=CAMERA_NAMES[v['cam']]
  if (frame,camera) in keys:continue
  if frame!=0 or not a.allow_first_frame_pkl:raise ValueError('No authoritative pose for '+v['name']+'; frame zero requires explicit --allow-first-frame-pkl')
  info=infos[frame];T=np.asarray(info[field])@np.linalg.inv(info['sensors']['cams'][camera]['extrinsic'])
  authority_rows.append(dict(camera=camera,frame=frame,**{'T_cw_%d%d'%(i,j):float(T[i,j]) for i in range(4) for j in range(4)}));fallbacks.append(v['name'])
 with (out/'camera_authority.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(authority_rows)
 b=bundle(a.mesh_source,frames=frames,pose_csv=out/'camera_authority.csv')
 for folder in ['images','lidar_depth','sky_mask','dynamic_mask','fixed_mask']:(out/folder).mkdir()
 for name in ['pointcloud.npz','track','timestamps.json','timestamps_specific.json']:
  src=source/name
  if src.is_dir():(out/name).symlink_to(src)
  else:(out/name).symlink_to(readable(src))
 records=[];calibrations={};errors=[]
 for fid in frames:
  E=np.asarray(infos[mapping[fid]][field],dtype=float)
  if E.shape!=(4,4) or not np.isfinite(E).all():raise ValueError('Invalid optimized ego')
  old=np.loadtxt(readable(source/'ego_pose'/('%06d.txt'%fid)));np.testing.assert_allclose(old,E,atol=1e-8,rtol=0)
  errors.append(float(abs(old-E).max()));np.savetxt(out/'ego_pose'/('%06d.txt'%fid),E,fmt='%.17g')
 for v in b['views']:
  name=v['name'];cam=CAMERA_NAMES[v['cam']];image=readable(source/'images'/(name+'.jpg'))
  with Image.open(image) as im:w,h=im.size
  original_size=b['manifest']['resolution'];K=np.asarray(v['K'],float).copy();K[0]*=w/original_size[0];K[1]*=h/original_size[1]
  cal=dict(K=K.tolist(),width=w,height=h)
  if cam in calibrations and calibrations[cam]!=cal:raise ValueError('Per-view intrinsics need separate calibration keys')
  calibrations[cam]=cal;records.append(dict(label=name+'.jpg',camera=cam,camera_to_world=v['c2w']))
  (out/'images'/(name+'.jpg')).symlink_to(image)
  for folder,srcfolder,ext in [('lidar_depth','lidar_depth','.npy'),('sky_mask','sky_mask','.png'),('fixed_mask','fixed_mask','.png')]:
   (out/folder/(name+ext)).symlink_to(readable(source/srcfolder/(name+ext)))
  (out/'dynamic_mask'/(name+'.png')).symlink_to(readable(b['root']/'converted/target_mask'/(name+'.png')))
 (out/'input_manifest.json').write_text(json.dumps(dict(records=records,calibrations=calibrations,pose_authority=b['audit']['camera_authority'],source=str(source)),indent=2))
 (out/'optimized_source_audit.json').write_text(json.dumps(dict(mesh=b['audit'],ego_field=field,ego_max_difference=max(errors),lidar_source=str(readable(source/'pointcloud.npz')),ground_reused=False,frames=frames,first_frame_optimized_pkl_fallbacks=fallbacks,first_frame_not_colmap_ba=bool(fallbacks)),indent=2))
 print(out)
if __name__=='__main__':main()
