#!/usr/bin/env python3
# Only model subset/export; never performs optimization.
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'_runtime'))
import pycolmap
p=argparse.ArgumentParser()
p.add_argument('command',choices=['image_deleter','model_converter'])
p.add_argument('--input_path',required=True);p.add_argument('--output_path',required=True)
p.add_argument('--image_names_path');p.add_argument('--output_type')
a=p.parse_args();r=pycolmap.Reconstruction(a.input_path)
if a.command=='image_deleter':
    names=set(Path(a.image_names_path).read_text().splitlines())
    frame_ids={im.frame_id for im in r.images.values() if im.name in names}
    for fid in frame_ids:
        registered=[im.name for im in r.images.values() if im.frame_id==fid]
        assert all(name in names for name in registered),('Mixed keep/delete frame',fid,registered)
    for fid in frame_ids:r.deregister_frame(fid)
    r.write_binary(a.output_path)
else:
    assert a.output_type=='TXT'
    r.write_text(a.output_path)
print('COLMAP model I/O complete',a.command,r.num_reg_images(),flush=True)
