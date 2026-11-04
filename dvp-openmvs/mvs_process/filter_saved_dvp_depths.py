#!/usr/bin/env python3
"""Create an isolated, mask-filtered copy of saved DVP depths (no MVS rerun)."""
import argparse,json,struct,shutil
from pathlib import Path
import cv2,numpy as np
from mask_policy import resize_ignore

def read_dmb(p):
    with p.open('rb') as f:
        v,h,w,t=struct.unpack('<4i',f.read(16));a=np.fromfile(f,dtype='<f4')
    if v!=1 or t not in (5,21):raise ValueError(str(p))
    return a.reshape((h,w) if t==5 else (h,w,3))
def write_dmb(p,a):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_bytes(struct.pack('<4i',1,a.shape[0],a.shape[1],5 if a.ndim==2 else 21)+np.asarray(a,dtype='<f4').tobytes())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--masks',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise ValueError('Choose a new output directory; never overwrite saved results')
    src=args.run/'scene';dst=args.output/'scene';dst.mkdir(parents=True)
    for name in ['blocks','images','metadata','cams','APD','metric_prior','metric_prior_unbounded']: (dst/name).mkdir()
    for name in ['cams','metadata']:
        for p in (src/name).iterdir():shutil.copy2(p,dst/name/p.name)
    shutil.copy2(src/'pair.txt',dst/'pair.txt')
    rows=[]
    for mp in sorted((src/'metadata').glob('*.json')):
        m=json.loads(mp.read_text());id=mp.stem;label=m['source_label'];cw,ch=m['content_size'];px,py=m['padding_xy']
        ignore=cv2.imread(str(args.masks/(label+'.mask.png')),0)
        if ignore is None:raise ValueError(label)
        keep=np.zeros((640,960),np.uint8);keep[py:py+ch,px:px+cw]=(~resize_ignore(ignore,(cw,ch))).astype(np.uint8)*255
        assert cv2.imwrite(str(dst/'blocks'/f'mask_{int(id)}.png'),keep)
        im=cv2.imread(str(src/'images'/f'{id}.jpg'));im[keep==0]=0;assert cv2.imwrite(str(dst/'images'/f'{id}.jpg'),im)
        dp=next((src/'APD').glob('*'+id+'*/depths.dmb'));dep=read_dmb(dp);removed=int(((dep>0)&(keep==0)).sum());dep[keep==0]=0
        write_dmb(dst/'APD'/dp.parent.name/'depths.dmb',dep)
        for p in dp.parent.glob('*normal*.dmb'):
            a=read_dmb(p);a[keep==0]=0;write_dmb(dst/'APD'/dp.parent.name/p.name,a)
        for folder in ['metric_prior','metric_prior_unbounded']:
            for p in (src/folder).glob(id+'*.dmb'):
                a=read_dmb(p);a[keep==0]=0;write_dmb(dst/folder/p.name,a)
        check=read_dmb(dst/'APD'/dp.parent.name/'depths.dmb');assert not np.any(check[keep==0])
        rows.append({'label':label,'removed_depth_pixels':removed,'remaining_depth_pixels':int((dep>0).sum())})
    shutil.copy2(args.run/'manifest.json',args.output/'manifest.json')
    report={'source_run':str(args.run),'mask_audit':str(args.masks/'mask_audit.json'),'operation':'postfilter saved depths and normals; no new DVP estimation or point-cloud fusion','images':len(rows),'removed_depth_pixels':sum(r['removed_depth_pixels'] for r in rows),'masked_depth_nonzero':0,'rows':rows}
    (args.output/'mask_filter_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if __name__=='__main__':main()
