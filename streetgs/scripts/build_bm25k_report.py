import json,math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');out=ROOT/'output/bm25k_20261007';names=['baseline','bilateral','mcmc'];results={};audit={}
for n in names:
 w=out/'workers'/n;ev=json.loads((w/n/'eval_25000.json').read_text());comp=json.loads((w/'completion.json').read_text());curve=[json.loads(l) for l in (w/n/'curve.jsonl').read_text().splitlines()];fixed=[r for r in curve if r['step']>=15000]
 assert ev['image_count']==707 and len(curve)==25000 and len(fixed)==10001 and len({r['points'] for r in fixed})==1 and not any(r['opacity_reset'] or r['mcmc_active'] for r in fixed)
 assert all(math.isfinite(r['loss']) for r in curve)
 results[n]=dict(psnr=ev['psnr'],ssim=ev['ssim'],points=sum(ev['points'].values()),**comp)
 if n=='bilateral':results[n].update(raw_psnr=ev['raw_psnr'],raw_ssim=ev['raw_ssim'])
 results[n]['per_camera']={c:dict(psnr=float(np.mean([r['psnr'] for r in ev['rows'] if r['image'].endswith('_'+c)])),ssim=float(np.mean([r['ssim'] for r in ev['rows'] if r['image'].endswith('_'+c)]))) for c in ['00','01','02','03','04','09','10']}
 audit[n]=dict(fixed_points_verified=True,rows=len(curve),fixed_rows=len(fixed),numerical_loss_finite=True)
base=results['baseline'];comparisons={n:dict(delta_psnr=results[n]['psnr']-base['psnr'],delta_ssim=results[n]['ssim']-base['ssim'],step_time_ratio=results[n]['step_seconds']/base['step_seconds']) for n in names[1:]}
(out/'summary.json').write_text(json.dumps(dict(results=results,comparisons=comparisons),indent=2));(out/'audit.json').write_text(json.dumps(audit,indent=2))
lookup={n:{r['image']:r for r in json.loads((out/'workers'/n/n/'eval_25000.json').read_text())['rows']} for n in names};rows=list(lookup['baseline'].values());mean=np.mean([r['psnr'] for r in rows]);selected=[('low',min(rows,key=lambda r:r['psnr'])),('mean',min(rows,key=lambda r:abs(r['psnr']-mean))),('high',max(rows,key=lambda r:r['psnr']))]
prev=json.loads((ROOT/'output/sample_00380_pose_ba_25000_20261001_172042/pose_comparison/results.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
for label,r in selected:
 name=r['image'];paths=[Path(prev['new_source'])/'images'/(name+'.jpg')]+[out/'workers'/n/n/'images_25000'/(name+'.png') for n in names];ims=[Image.open(p).convert('RGB') for p in paths];w,h=ims[1].size;hh=round(h*640/w);canvas=Image.new('RGB',(2560,hh+65),'white');d=ImageDraw.Draw(canvas)
 for i,(im,title) in enumerate(zip(ims,['GT '+name]+[n+' %.3f dB'%lookup[n][name]['psnr'] for n in names])):d.text((i*640+8,15),title,font=font,fill='black');canvas.paste(im.resize((640,hh)),(i*640,65))
 canvas.save(out/('comparison_'+label+'.jpg'),quality=95)
body='<h1>Bilateral Grid / MCMC 25000步对比</h1><p>同一初始化、相机序列、GPU1串行、packed=True。15000步之后固定点集，MCMC也停止位置噪声。车辆区域loss保持25000步启动。707张训练图，非新视角测试。</p><table><tr><th>组别</th><th>PSNR</th><th>SSIM</th><th>点数</th><th>训练步累计秒</th><th>循环墙钟秒</th><th>任务墙钟秒</th><th>峰值allocated MiB</th></tr>'
for n,r in results.items():body+=f"<tr><td>{n}</td><td>{r['psnr']:.5f}</td><td>{r['ssim']:.6f}</td><td>{r['points']:,}</td><td>{r['step_seconds']:.1f}</td><td>{r['loop_wall_seconds']:.1f}</td><td>{r['job_wall_seconds']:.1f}</td><td>{r['peak_mb']:.1f}</td></tr>"
body+='</table><p>循环墙钟包含中途评估与保存；任务墙钟还包含加载及起点评估。若resumed=true，墙钟只表示最后一次进程段，须结合日志单独汇总，不能直接比较。</p>'
r=results['bilateral'];body+=f"<p>Bilateral使用每相机共享8×8×8网格与高斯联合优化，TV=10，identity=0.001，grid LR=0.002；保留原有颜色校正。网格校正前：PSNR {r['raw_psnr']:.5f}，SSIM {r['raw_ssim']:.6f}。校正后指标包含外观拟合收益，不等同于几何改善。</p>"
body+='<p>MCMC使用逐对象重定位、5%增长和位置噪声(noise_lr=5e5)，点数预算参考上一轮packed各对象最终点数且不低于初始点数。附加opacity/scale正则各0.01；无原始克隆/分裂/opacity reset。小对象每次至少增长1点至预算。没有跨背景/车辆搬点。此为算法组合对比，MCMC损失及点数路径不同，不能把变化只归因于某一个操作。</p><h2>各视角</h2><table><tr><th>相机</th><th>baseline PSNR/SSIM</th><th>bilateral PSNR/SSIM</th><th>mcmc PSNR/SSIM</th></tr>'
for c in ['00','01','02','03','04','09','10']:body+='<tr><td>'+c+'</td>'+''.join(f"<td>{results[n]['per_camera'][c]['psnr']:.4f} / {results[n]['per_camera'][c]['ssim']:.5f}</td>" for n in names)+'</tr>'
body+='</table>'+''.join('<h2>'+label+'</h2><img src="comparison_'+label+'.jpg">' for label in ['low','mean','high'])
(out/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{font:17px/1.7 sans-serif;margin:30px;max-width:1600px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}img{width:100%}</style>'+body+'</html>')
print(json.dumps(comparisons,indent=2))
