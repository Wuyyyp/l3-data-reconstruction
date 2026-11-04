from pathlib import Path
import csv,json,pickle,sys,copy
from prepare_pose_input import U,CAMERAS
root=Path(sys.argv[1]);pose=root/'pose';source=pose/'run/deliverables';ba=source/'colmap-ba'
out=root/'delivery'/json.loads((pose/'input_audit.json').read_text())['sample'];(out/'colmap/poses').mkdir(parents=True)
(out/'colmap/sparse_ba').symlink_to(ba/'sparse_after',target_is_directory=True)
# The BA file carries the 16 optimized matrix entries; join only file identity
# and time from its paired pre-BA pose export. Never substitute a pre-BA pose.
source_rows={(r['frame'],r['camera'],r['image_filename']):r for r in csv.DictReader((source/'image_extrinsics/image_camera_to_world_after.csv').open())}
rows=list(csv.DictReader((ba/'image_camera_to_world_after_ba.csv').open()));assert len(rows)==len(source_rows)==707
for row in rows:
    peer=source_rows[row['frame'],row['camera'],row['image_filename']];row.update(timestamp_ns=peer['timestamp_ns'],image_path=peer['image_path'])
    assert Path(row['image_path']).is_file()
with (out/'colmap/poses/image_camera_to_world_after_ba.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
data=U((root/'geometry/mesh_input/result_interpolation_optimized_first101.pkl').open('rb')).load()
# DVP expects delivery-relative perception references; raw points are left unchanged.

(out/'compensated_perception').mkdir(exist_ok=True)
perception_sources={}
for info in data['infos']:
    lidar=info['sensors']['lidar']['perception'];path=Path(lidar['data_path'])
    link=out/'compensated_perception'/path.name
    if not path.is_file():raise FileNotFoundError(path)
    if not link.exists():link.symlink_to(path)
    lidar['data_path']=lidar['aws_path']=str(link.relative_to(out))
    assert lidar['dynamic_compensated']
    for camera in CAMERAS:
        cam=info['sensors']['cams'][camera];cam['data_path']=cam['aws_path']=str(pose/'canonical/images'/camera/(str(int(cam['timestamp']))+'.jpg'))
with (out/'result_interpolation_optimized.pkl').open('wb') as f:pickle.dump(data,f,protocol=4)
(out/'adapter.json').write_text(json.dumps(dict(dynamic_compensation=True,perception_directory='delivered compensated perception',ba_source=str(ba),camera_authority='after_ba',images=707),indent=2))
print(out)
