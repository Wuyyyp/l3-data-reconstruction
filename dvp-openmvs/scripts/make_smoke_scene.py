"""Deterministic calibrated textured plane; tests execution, not benchmark accuracy."""
from pathlib import Path
import argparse
import cv2
import numpy as np
p = argparse.ArgumentParser(); p.add_argument('output'); p.add_argument('--width', type=int, default=320); a=p.parse_args()
out=Path(a.output); w=a.width; h=240; f=300.; z=4.
for d in ('images','cams'): (out/d).mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(7)
texture=rng.integers(0,256,(h+100,w+300,3),dtype=np.uint8)
texture=cv2.GaussianBlur(texture,(5,5),0)
lines=['5']
for i in range(5):
    cx=(i-2)*.08
    shift=int(round(f*cx/z))
    im=texture[50:50+h,150+shift:150+shift+w]
    cv2.imwrite(str(out/'images'/f'{i:08d}.jpg'),im)
    (out/'cams'/f'{i:08d}_cam.txt').write_text(f'extrinsic\n1 0 0 {-cx}\n0 1 0 0\n0 0 1 0\n0 0 0 1\n\nintrinsic\n{f} 0 {w/2}\n0 {f} {h/2}\n0 0 1\n\n2 0.01 400 6\n')
    lines += [str(i),'4 '+' '.join(f'{j} 10' for j in range(5) if j!=i)]
(out/'pair.txt').write_text('\n'.join(lines)+'\n')
print(out)
