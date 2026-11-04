import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
root=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');base=root/'output/gsplat_stage67_fixed15k_20261006'
out=base/'old50k_comparison/visuals';out.mkdir(exist_ok=True)
r=json.loads((base/'old50k_comparison/results.json').read_text())
prev=json.loads((root/'output/sample_00380_pose_ba_25000_20261001_172042/pose_comparison/results.json').read_text())
filenames={Path(x['image']).stem:x['image'] for x in prev['rows']}
rows=r['rows'];groups=[]
def choose(rs):
    mean=sum(x['psnr']['gsplat'] for x in rs)/len(rs)
    return [('low',min(rs,key=lambda x:x['psnr']['gsplat'])),('mean',min(rs,key=lambda x:abs(x['psnr']['gsplat']-mean))),('high',max(rs,key=lambda x:x['psnr']['gsplat']))]
groups.append(dict(id='overall',examples=choose(rows)))
for c in sorted(r['cameras']):groups.append(dict(id='camera_'+c,examples=choose([x for x in rows if x['camera']==c])))
delta=lambda x:x['psnr']['gsplat']-x['psnr']['old50k']
groups.append(dict(id='difference',examples=[('largest_gain',max(rows,key=delta)),('largest_drop',min(rows,key=delta))]))
try:font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
except OSError:font=ImageFont.load_default()
manifest=[]
for group in groups:
    examples=[]
    for label,row in group['examples']:
        stem=row['image'];filename=filenames[stem];dest=out/(stem+'.jpg')
        if not dest.exists():
            paths=[Path(prev['new_source'])/'images'/filename,Path(prev['old_render_dir'])/filename,base/'workers/gsplat/gsplat/images_25000'/(stem+'.png')]
            ims=[Image.open(x).convert('RGB') for x in paths];size=ims[2].size
            ims=[im.resize(size,Image.BILINEAR) if im.size!=size else im for im in ims]
            w=800;h=round(size[1]*w/size[0]);header=70
            canvas=Image.new('RGB',(3*w,h+header),'white');d=ImageDraw.Draw(canvas)
            titles=['GT','StreetGS 50000 | PSNR {:.3f} dB'.format(row['psnr']['old50k']),'gsplat 25000 | PSNR {:.3f} dB'.format(row['psnr']['gsplat'])]
            for i,(title,im) in enumerate(zip(titles,ims)):
                d.text((i*w+12,8),title,font=font,fill='#17263c');d.text((i*w+12,37),stem,font=font,fill='#53647b')
                canvas.paste(im.resize((w,h),Image.LANCZOS),(i*w,header))
            canvas.save(dest,quality=95,subsampling=0)
        examples.append(dict(category=label,image=stem,camera=row['camera'],old_psnr=row['psnr']['old50k'],gsplat_psnr=row['psnr']['gsplat'],delta_psnr=delta(row),file=dest.name))
    manifest.append(dict(id=group['id'],examples=examples))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(groups=len(groups),unique_images=len(list(out.glob('*.jpg'))),overall=manifest[0],difference=manifest[-1]),indent=2))
