"""Final comparison: existing matched cubemap baseline versus independently rendered sky GS."""
import json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def build(root,out):
    ref=root/'output/gsplat_stage67_fixed15k_20261006/workers/gsplat/gsplat'
    cand=out/'gsplat';a=json.loads((ref/'eval_25000.json').read_text());b=json.loads((cand/'eval_25000.json').read_text());aa={r['image']:r for r in a['rows']};bb={r['image']:r for r in b['rows']}
    assert aa.keys()==bb.keys() and len(aa)==707
    summary=json.loads((out/'comparison.json').read_text());summary['per_camera']={};summary['rows']=[]
    # Same production masks for the scalar eval; regional stats below come from saved 8-bit images.
    for cam in ['00','01','02','03','04','09','10']:
        keys=sorted(k for k in aa if k.endswith('_'+cam));summary['per_camera'][cam]={m:{'cubemap':float(np.mean([aa[k][m] for k in keys])),'sphere':float(np.mean([bb[k][m] for k in keys]))} for m in ['psnr','ssim']}
        for k in keys:summary['rows'].append(dict(image=k,cubemap_psnr=aa[k]['psnr'],sphere_psnr=bb[k]['psnr'],delta_psnr=bb[k]['psnr']-aa[k]['psnr'],cubemap_ssim=aa[k]['ssim'],sphere_ssim=bb[k]['ssim']))
    summary['regional_8bit_comparison']={}
    for region in ['sky','non_sky','boundary']:
        rr=[v['regions_8bit'][region] for v in b['rows'] if region in v.get('regions_8bit',{})]
        summary['regional_8bit_comparison'][region]={key:float(np.mean([v[key] for v in rr])) for key in ['cubemap_psnr','sphere_psnr']} if rr else {}
    source=Path('/mnt/nuplan/l3data-reconstruction-bingxing/pose_ba_streetgs_runs/sample_00380_clip_M18-2_07_20251230153515_DF_f184_284_right/run_20261001/streetgauss/converted')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    parts=['<!doctype html><meta charset="utf-8"><title>Sky sphere vs cubemap</title><style>body{background:#142033;color:white;font:18px sans-serif;margin:30px}img{width:100%;max-width:1920px}td,th{padding:10px;border:1px solid #567}</style><h1>Sky sphere vs cubemap / gsplat / 25000 steps</h1><p>Same initial scene and 707 training views. Production masks; not held-out quality. Sky sphere topology stops at step 15000; parameters continue training.</p>','<table><tr><th>Camera</th><th>Cubemap PSNR</th><th>Sphere PSNR</th><th>Cubemap SSIM</th><th>Sphere SSIM</th></tr>']
    for cam,v in summary['per_camera'].items():parts.append('<tr><td>%s</td><td>%.3f</td><td>%.3f</td><td>%.5f</td><td>%.5f</td></tr>'%(cam,v['psnr']['cubemap'],v['psnr']['sphere'],v['ssim']['cubemap'],v['ssim']['sphere']))
    parts.append('</table><h2>Region PSNR (both variants quantized to 8-bit; 3px boundary band)</h2><pre>'+json.dumps(summary['regional_8bit_comparison'],indent=2)+'</pre>')
    for cam in summary['per_camera']:
        rows=[v for v in summary['rows'] if v['image'].endswith('_'+cam)];ordered=sorted(rows,key=lambda r:r['delta_psnr']);choices=[('largest_drop',ordered[0]),('median_change',ordered[len(ordered)//2]),('largest_gain',ordered[-1])]
        for label,row in choices:
            key=row['image'];paths=[source/'images'/(key+'.jpg'),ref/'images_25000'/(key+'.png'),cand/'images_25000'/(key+'.png')];ims=[Image.open(p).convert('RGB') for p in paths];h=round(640*ims[0].height/ims[0].width);sheet=Image.new('RGB',(1920,h+64),'#192c44');d=ImageDraw.Draw(sheet)
            for i,(im,title) in enumerate(zip(ims,['GT','Cubemap %.2f dB'%row['cubemap_psnr'],'Sky sphere %.2f dB'%row['sphere_psnr']])):sheet.paste(im.resize((640,h)),(i*640,64));d.text((i*640+10,35),title,font=font,fill='white')
            d.text((10,8),key+' '+label+' delta %.3f dB'%row['delta_psnr'],font=font,fill='white');name='comparison_'+cam+'_'+label+'.jpg';sheet.save(out/name,quality=93);parts.append('<h3>%s %s</h3><img src="%s">'%(cam,label,name))
    (out/'comparison.json').write_text(json.dumps(summary,indent=2));(out/'report.html').write_text('\n'.join(parts))
