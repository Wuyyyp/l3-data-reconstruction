"""Validate all five depth maps and a nonempty fused point cloud."""
import json,struct,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); result={}
for i in range(5):
 p=root/'APD'/f'{i:08d}'/'depths.dmb'
 with p.open('rb') as f:
  version,h,w,typ=struct.unpack('4i',f.read(16));d=np.frombuffer(f.read(),np.float32).reshape(h,w)
 assert version==1 and typ==5 and np.isfinite(d).all()
 valid=d[d>0];assert valid.size>0
 result[str(i)]={'shape':[h,w],'valid_fraction':float(valid.size/d.size),'median_depth':float(np.median(valid)),'median_abs_error_to_plane_z4':float(np.median(np.abs(valid-4)))}
p=root/'APD'/'APD.ply'
with p.open('rb') as f:
 header=[]
 while True:
  line=f.readline().decode('ascii').strip();header.append(line)
  if line=='end_header':break
  assert line
n=int(next(l.split()[-1] for l in header if l.startswith('element vertex')));assert n>0
result['ply_vertices']=n
print(json.dumps(result,indent=2))
