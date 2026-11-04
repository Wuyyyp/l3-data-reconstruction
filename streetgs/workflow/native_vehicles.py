"""Supply an empty mesh placement list; the preserve stage then uses native actors."""
import json, shutil, sys
from pathlib import Path
w=Path(sys.argv[1])
dest=w/'mesh_inputs/mesh_registration'
(dest/'registration/mask-videos/masks').mkdir(parents=True,exist_ok=True)
shutil.copy2(w/'geometry/mesh_input/result_interpolation_optimized_first101.pkl',dest/'result_interpolation_optimized_first101_with_meshes.pkl')
for key,value in [('view_transforms',{}),('asset_matches',{}),('placements',{'frames':[{'frame':f,'objects':[]} for f in range(101)]})]:
    (dest/(key+'.json')).write_text(json.dumps(value))
(w/'shared_mesh_acceptance.json').write_text(json.dumps(dict(mesh_version='native',shared_camera_pose=True,source_fields_preserved=True,identical_camera_keys=0,track_frames=0)))
