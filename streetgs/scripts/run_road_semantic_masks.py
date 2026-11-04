from pathlib import Path
import json,time
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor,SegformerForSemanticSegmentation
from scipy.ndimage import label,binary_fill_holes

import argparse
parser=argparse.ArgumentParser(description='Road semantics seeded by projected ground; excludes ego and dynamic targets, no sky-mask use.')
parser.add_argument('--source',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--weights',type=Path,required=True)
args=parser.parse_args();S=args.source;O=args.output;W=args.weights
O.mkdir(exist_ok=True)
for d in ['semantic_road_mask','final_mask','masked_rgb','semantic_masked_rgb','evidence','previews','strict_mask']:(O/d).mkdir(exist_ok=True)
torch.set_num_threads(4)
processor=AutoImageProcessor.from_pretrained(W,local_files_only=True)
model=SegformerForSemanticSegmentation.from_pretrained(W,local_files_only=True).cuda().eval()
assert model.config.id2label[0].lower()=='road'
views=json.loads((S/'input_manifest.json').read_text())['views']
rows=[];started=time.time()
for i,v in enumerate(views):
 name=v['name'];fid,cam=name.split('_')
 im=Image.open(S/'converted/images'/f'{name}.jpg').convert('RGB');rgb=np.array(im);w,h=im.size
 inp=processor(images=im,size={'height':768,'width':1152},return_tensors='pt').to('cuda')
 with torch.inference_mode():
  logits=F.interpolate(model(**inp).logits,size=(h,w),mode='bilinear',align_corners=False)
  prob=logits.softmax(1);labels=logits.argmax(1)[0].cpu().numpy();confidence=prob[0,0].cpu().numpy()
 del inp,logits,prob
 def mask(folder):return np.array(Image.open(S/'converted'/folder/f'{name}.png').convert('L'))>0
 excluded=mask('target_mask')
 extra=S/'static_excluded'/f'{name}.png'
 if extra.exists():excluded|=np.array(Image.open(extra).convert('L'))>0
 valid=mask('valid_mask')&~excluded
 projected=np.load(S/'ground_masks'/f'cam_{cam}'/f'{fid}.npy').astype(bool)
 assert projected.shape==(h,w)
 semantic=(labels==0)&(confidence>=.6)&valid
 strict=semantic&projected
 # Grow within semantic-road connected regions, never across explicit non-road.
 cc,ncc=label(semantic);sizes=np.bincount(cc.ravel(),minlength=ncc+1);support=np.bincount(cc[strict],minlength=ncc+1)
 accepted=(sizes>=128)&(support>=32);accepted[0]=False
 final=accepted[cc]
 # Fill only tiny uncertain holes; preserve confident non-road, ego and objects.
 holes=binary_fill_holes(final)&~final
 hc,nh=label(holes);hs=np.bincount(hc.ravel(),minlength=nh+1)
 fill=(hc>0)&(hs[hc]<=64)&(labels==0)&(confidence>=.35)&valid
 final|=fill
 evidence=np.zeros((h,w),np.uint8);evidence[final&~projected]=1;evidence[projected&~final&valid]=2;evidence[final&projected]=3;evidence[semantic&~final]=4;evidence[~valid]=255
 for folder,m in [('semantic_road_mask',semantic),('final_mask',final),('strict_mask',strict)]:Image.fromarray(m.astype('uint8')*255).save(O/folder/f'{name}.png')
 Image.fromarray(evidence).save(O/'evidence'/f'{name}.png')
 Image.fromarray(rgb*final[:,:,None]).save(O/'masked_rgb'/f'{name}.png')
 Image.fromarray(rgb*semantic[:,:,None]).save(O/'semantic_masked_rgb'/f'{name}.jpg',quality=95)
 assert not (final&~valid).any() and not (final&(labels!=0)).any()
 row=dict(name=name,width=w,height=h,projected=int(projected.sum()),semantic=int(semantic.sum()),final=int(final.sum()),removed=int((projected&~final).sum()),expanded=int((final&~projected).sum()),supported=int((final&projected).sum()))
 rows.append(row)
 if v['frame'] in [0,50,100]:
  parts=[]
  for title,arr in [('RGB',rgb),('Previous sparse projected ground',rgb*projected[:,:,None]),('Final dense road (point-supported region)',rgb*final[:,:,None])]:
   tile=Image.new('RGB',(640,450),'#181818');pic=Image.fromarray(arr);pic.thumbnail((640,410));tile.paste(pic,(0,35));ImageDraw.Draw(tile).text((10,10),title,fill='white');parts.append(tile)
  canvas=Image.new('RGB',(1920,450));
  for j,p in enumerate(parts):canvas.paste(p,(640*j,0))
  canvas.save(O/'previews'/f'{name}.jpg',quality=90)
 if i%20==0 or i+1==len(views):
  status=dict(done=i+1,total=len(views),seconds=time.time()-started,last=row)
  (O/'status.json').write_text(json.dumps(status));print(json.dumps(status),flush=True)
report=dict(source=str(S),model=str(W),classes=model.config.id2label,inference_size=[1152,768],road_probability_threshold=.6,views=len(rows),seconds=time.time()-started,method='Keep semantic road components >=128 pixels with >=32 projected visible ground seeds; fill <=64-pixel holes only where argmax is road and probability>=.35. Apply ego and dynamic exclusions; do not read sky masks.',evidence_codes={'0':'not selected','1':'semantic expansion, no direct point support','2':'rejected projected ground','3':'final road with point support','4':'semantic road without accepted seeded component','255':'excluded'},limitations=['Component support does not verify per-pixel geometry of expanded regions.','Use paired source ego-valid and dynamic exclusion masks.','No point-cloud changes or training.'],totals={k:sum(r[k] for r in rows) for k in ['projected','semantic','final','removed','expanded','supported']},rows=rows,validation='All final masks contain only road-class valid pixels; RGB outside final mask exactly zero.')
(O/'report.json').write_text(json.dumps(report,indent=2))
print('COMPLETE',flush=True)
