"""Read-only training audit, writes standalone reports into the experiment directory."""
import json,html,time,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/gsplat_stage67'
def read(p,default=None):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def build():
    status=read(OUT/'status.json',{})
    lines=['# StreetGS / gsplat 第六、七阶段训练对比','',
        '状态：`'+status.get('state','unknown')+'`；更新：'+time.strftime('%Y-%m-%d %H:%M:%S %z')+'。','',
        '通过标准：平均 PSNR 降幅不超过 0.15 dB、SSIM 降幅不超过 0.003，且无数值异常。单项优化还需实测降低时间或显存。','',
        '固定同一初始化、相机顺序、损失和训练 mask；保留 25000 步学习率及增密计划。4000 步通过后从该 checkpoint 继续。',
        '评估为 0/40/99 帧 × 7 个视角，共 21 张训练视图，不代表独立测试集泛化能力。','',
        '|配置|步数|平均 PSNR (dB)|平均 SSIM|高斯数|', '|---|---:|---:|---:|---:|']
    evaluations={}
    for name in ['legacy','gsplat','packed','radius1']:
        for path in sorted((OUT/name).glob('eval_*.json'),key=lambda p:int(p.stem.split('_')[1])):
            e=read(path)
            if e:
                evaluations[(name,e['step'])]=e
                lines.append('|{}|{}|{:.5f}|{:.6f}|{}|'.format(name,e['step'],e['psnr'],e['ssim'],sum(e['points'].values())))
    lines+=['','## 验收','', '|配置|参考|步数|ΔPSNR|ΔSSIM|质量达标|性能收益|进入下一阶段|','|---|---|---:|---:|---:|---|---|---|']
    for p in sorted(OUT.glob('gate_*.json')):
        g=read(p)
        if g:lines.append('|{}|{}|{}|{:+.5f}|{:+.6f}|{}|{}|{}|'.format(g['candidate'],g['reference'],g['step'],g['delta_psnr'],g['delta_ssim'],g['quality_pass'],g['performance_benefit'],g['passed']))
    lines+=['','## 训练事件及时间','',
        '常规步耗时排除前 100 步及每 100 步的检查/增密事件。瞬时显存为峰值减去该步开始前已分配显存，不包含常驻模型/优化器。',
        '单项优化的性能验收另外采用同一 checkpoint、轮换执行顺序的配对测试，避免不同点数或串行执行造成偏差。','',
        '|配置|已完成步数|常规步中位数 ms|瞬时显存中位数 MiB|点数改变次数|最后轨迹平移变化|',
        '|---|---:|---:|---:|---:|---:|']
    curves={}
    for name in ['legacy','gsplat','packed','radius1']:
        rows=[]
        try:
            for line in (OUT/name/'curve.jsonl').read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
        except FileNotFoundError:pass
        if not rows:continue
        curves[name]=rows;regular=[r for r in rows if r['step']>100 and r['step']%100]
        pose=[r['track_translation_change'] for r in rows if 'track_translation_change' in r]
        lines.append('|{}|{}|{}|{}|{}|{}|'.format(name,rows[-1]['step'],round(float(np.median([r['total_ms'] for r in regular])),3) if regular else '-',round(float(np.median([r['peak_increment_mb'] for r in regular])),2) if regular else '-',sum(r['points']!=r['points_before'] for r in rows),pose[-1] if pose else '-'))
    for step in [4000,25000]:
        a=evaluations.get(('legacy',step));b=evaluations.get(('gsplat',step))
        if not a or not b:continue
        lines+=['','## {} 步逐视图指标'.format(step),'','|图像|legacy PSNR|gsplat PSNR|ΔPSNR|legacy SSIM|gsplat SSIM|','|---|---:|---:|---:|---:|---:|']
        for ra,rb in zip(a['rows'],b['rows']):
            assert ra['image']==rb['image']
            lines.append('|{}|{:.4f}|{:.4f}|{:+.4f}|{:.6f}|{:.6f}|'.format(ra['image'],ra['psnr'],rb['psnr'],rb['psnr']-ra['psnr'],ra['ssim'],rb['ssim']))
        chosen=[min(b['rows'],key=lambda r:r['psnr']),min(b['rows'],key=lambda r:abs(r['psnr']-b['psnr'])),max(b['rows'],key=lambda r:r['psnr'])]
        for label,row in zip(['low','mean','high'],chosen):
            paths=[OUT/n/f'images_{step}'/(row['image']+'.png') for n in ['legacy','gsplat']]
            if not all(p.exists() for p in paths):continue
            target=OUT/f'example_{step}_{label}.jpg'
            if not target.exists():
                images=[Image.open(p).convert('RGB') for p in paths]
                diff=np.abs(np.asarray(images[0]).astype(float)-np.asarray(images[1]).astype(float))*10
                images.append(Image.fromarray(diff.clip(0,255).astype('uint8')))
                w=640;h=int(images[0].height*w/images[0].width);canvas=Image.new('RGB',(w*3,h+40),'white');d=ImageDraw.Draw(canvas)
                for i,(title,im) in enumerate(zip(['Legacy','gsplat','Absolute difference x10'],images)):
                    canvas.paste(im.resize((w,h)),(i*w,40));d.text((i*w+10,10),title,fill='black')
                canvas.save(target,quality=90)
            lines+=['','{} PSNR 示例：{}，{:.3f} dB。'.format(label,row['image'],row['psnr']),'','![{}]({})'.format(label,target.name)]
    if status.get('error'):lines+=['','## 失败详情','','```',status['error'],'```']
    lines+=['','packed=True 与 radius_clip=1 分开实验；没有将它们叠加。没有执行 Git push。']
    text='\n'.join(lines)+'\n'
    tmp=OUT/'REPORT.md.tmp';tmp.write_text(text);tmp.replace(OUT/'REPORT.md')
    # Keep rendering self-contained without optional Markdown dependencies.
    body=[]
    for line in lines:
        if line.startswith('!['):
            name=line.split('](')[1][:-1];body.append('<img style="width:100%" src="'+html.escape(name)+'">')
        else:body.append(html.escape(line))
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>StreetGS / gsplat</title><body style="font:16px monospace;background:#f4f6fa;padding:28px"><pre style="white-space:pre-wrap">'+'\n'.join(body)+'</pre></body>')
    print('REPORT',status.get('state'),flush=True)
if __name__=='__main__':
    if '--watch' not in sys.argv:build()
    else:
        while True:
            build()
            if read(OUT/'status.json',{}).get('state') in ['complete','failed','needs_review','smoke_complete']:break
            pidfile=OUT/'pipeline.pid'
            if pidfile.exists():
                proc=Path('/proc')/pidfile.read_text().strip()/'stat'
                if not proc.exists() or proc.read_text().split(') ')[1].startswith('Z'):
                    state=read(OUT/'status.json',{});state.update(state='failed',error='Training process exited before a terminal status; inspect pipeline.log')
                    (OUT/'status.json').write_text(json.dumps(state,indent=2));build();break
            time.sleep(120)
