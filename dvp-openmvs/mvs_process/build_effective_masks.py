#!/usr/bin/env python3
"""Union supplied semantic/dynamic masks, calibrated ego rig and boundary padding."""
import argparse, hashlib, json
from pathlib import Path
import cv2
import numpy as np
from mask_policy import VERSION, rig_mask

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--input-masks',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--sky-masks',type=Path)
    ap.add_argument('--rig-profile',type=Path,required=True)
    args=ap.parse_args()
    cv2.setNumThreads(1)
    profile=json.loads(args.rig_profile.read_text())
    digest=hashlib.sha256(VERSION.encode()+args.rig_profile.read_bytes())
    pending=[];rows=[];previews=[]
    for p in sorted(args.images.glob('*.jpg')):
        image=cv2.imread(str(p)); raw=(args.input_masks/(p.stem+'.mask.png')).read_bytes()
        base=cv2.imdecode(np.frombuffer(raw,np.uint8),0)
        if image is None or base is None or base.shape!=image.shape[:2]: raise ValueError(str(p))
        h,w=base.shape; ignore=(base>=128).astype(np.uint8)*255
        if args.sky_masks:
            sky=cv2.imread(str(args.sky_masks/(p.stem+'.png')),0)
            if sky is None or sky.shape!=base.shape: raise ValueError('Missing/mismatched sky mask: '+p.stem)
            ignore[sky>=128]=255
        camera=p.stem.rsplit('_',1)[1]
        rig=np.zeros_like(ignore)
        if camera=='10': rig=rig_mask(base.shape,profile['camera10_topline'])
        # Only black components attached to an image border are padding candidates;
        # do not remove interior black cars, windows, shadows etc by RGB alone.
        black=(image.max(axis=2)<=2).astype(np.uint8)
        _,labels=cv2.connectedComponents(black,8)
        border=np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]))
        border=border[border!=0]; padding=np.isin(labels,border).astype(np.uint8)*255
        merged=cv2.bitwise_or(cv2.bitwise_or(ignore,rig),padding)
        radius=int(round(profile.get('edge_margin_at_960',2)*w/960))
        if radius: merged=cv2.dilate(merged,np.ones((2*radius+1,2*radius+1),np.uint8))
        ok,encoded=cv2.imencode('.png',merged);assert ok
        data=encoded.tobytes();digest.update(p.name.encode());digest.update(data)
        pending.append((p.stem+'.mask.png',data))
        rows.append({'label':p.stem,'old_ignored':int((ignore>0).sum()),'new_ignored':int((merged>0).sum()),'rig_pixels':int((rig>0).sum())})
        if camera=='10' and int(p.stem.split('_')[0]) in [0,25,50,75,84,100]:
            overlay=image.copy();overlay[merged>0]=(0,0,255);overlay=cv2.addWeighted(image,.6,overlay,.4,0)
            previews.append((p.stem,cv2.resize(overlay,(960,round(h*960/w)))))
    if not pending: raise ValueError('No input images')
    fingerprint=digest.hexdigest();audit=args.output/'mask_audit.json'
    if audit.exists() and json.loads(audit.read_text())['sha256']!=fingerprint:
        raise RuntimeError('Masks changed: use a new output/result directory; cached depths must not be reused')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,data in pending:
        target=args.output/name
        if not target.exists() or target.read_bytes()!=data: target.write_bytes(data)
    for name,image in previews: cv2.imwrite(str(args.output/(name+'_preview.jpg')),image)
    report={'version':VERSION,'sha256':fingerprint,'image_count':len(rows),'semantics':'255=ignore,0=keep','profile':profile,'input_masks':str(args.input_masks),'rows':rows}
    audit.write_text(json.dumps(report,indent=2)+'\n')
    print(fingerprint)
if __name__=='__main__': main()
