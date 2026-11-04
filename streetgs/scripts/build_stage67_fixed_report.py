import json,time,html,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/gsplat_stage67_fixed15k_20261006'
NAMES=['legacy','gsplat','packed','radius1']
def read(p,default=None):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return default
def data(name):return OUT/'workers'/name/name
def build():
    state=read(OUT/'status.json',{})
    summary={'state':state.get('state'),'variants':{},'comparisons':{},'updated':time.time()}
    lines=['# 25000 步重跑：15000 步开始固定点集','',
        '四组独立从相同初始化训练。停止增密/裁剪/opacity reset：15000；车辆区域 loss 独立在 25000 启动。',
        '最终 PSNR/SSIM 统计全部 707 张训练图；中途评估仅固定 21 张图，因此不能直接把最终均值与中途均值当作同一条质量曲线。',
        '保留生产训练 mask、天空、颜色校正、车辆轨迹优化。此处不是独立测试集指标。','',
        '|配置|完成步数|PSNR|SSIM|训练循环墙钟 h|训练步累积 h|整个任务 h|',
        '|---|---:|---:|---:|---:|---:|---:|']
    metrics={};curves={}
    for name in NAMES:
        completion=read(OUT/'workers'/name/'completion.json',{})
        e=read(data(name)/'eval_25000.json')
        rows=[]
        try:
            for line in (data(name)/'curve.jsonl').read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:pass
        except FileNotFoundError:pass
        curves[name]=rows
        if e:metrics[name]=e
        per=completion.copy();per.update(steps=rows[-1]['step'] if rows else 0)
        if e:per.update(psnr=e['psnr'],ssim=e['ssim'],image_count=len(e['rows']),points=sum(e['points'].values()))
        per['phases']={}
        for label,items in [('densifying',[r for r in rows if 100<r['step']<15000]),('fixed',[r for r in rows if r['step']>=15000])]:
            if items:
                per['phases'][label]=dict(mean_step_ms=statistics.mean(r['total_ms'] for r in items),median_step_ms=statistics.median(r['total_ms'] for r in items),max_cuda_allocated_mb=max(r['peak_allocated_mb'] for r in items),median_transient_mb=statistics.median(r['peak_increment_mb'] for r in items))
        summary['variants'][name]=per
        def hours(k):return '{:.3f}'.format(completion[k]/3600) if k in completion else '运行中'
        lines.append('|{}|{}|{}|{}|{}|{}|{}|'.format(name,per['steps'],'{:.5f}'.format(e['psnr']) if e else '待完成','{:.6f}'.format(e['ssim']) if e else '待完成',hours('training_wall_seconds'),hours('sum_step_seconds'),hours('job_wall_seconds')))
    lines+=['','训练循环墙钟含中途/最终评估和 checkpoint 保存；训练步累积仅包含逐步 forward/loss/backward、拓扑与优化器更新及数值检查。整个任务另含数据加载及初始化。',
        '不同 GPU 或并行时的墙钟可能受共享 CPU/I/O 和外部 GPU 负载影响。最终另用同 checkpoint、同 GPU、轮换顺序的配对测试比较后端成本。','',
        '|配置|阶段|平均步 ms|中位步 ms|最大 PyTorch allocated MiB|瞬时显存中位数 MiB|','|---|---|---:|---:|---:|---:|']
    for name,per in summary['variants'].items():
        for phase,p in per['phases'].items():lines.append('|{}|{}|{:.3f}|{:.3f}|{:.1f}|{:.1f}|'.format(name,phase,p['mean_step_ms'],p['median_step_ms'],p['max_cuda_allocated_mb'],p['median_transient_mb']))
    bench=read(OUT/'workers/gsplat/benchmark_25000.json',{})
    for name,ref in [('gsplat','legacy'),('packed','gsplat'),('radius1','gsplat')]:
        if name not in metrics or ref not in metrics:continue
        a,b=metrics[ref],metrics[name];dp=b['psnr']-a['psnr'];ds=b['ssim']-a['ssim']
        c=dict(reference=ref,delta_psnr=dp,delta_ssim=ds,quality_pass=dp>=-.15 and ds>=-.003)
        if name in bench.get('comparisons',{}):c['controlled_performance']=bench['comparisons'][name]
        if name=='gsplat' and 'legacy' in bench.get('comparisons',{}):
            ratio=bench['comparisons']['legacy']['time_ratio'];c['gsplat_vs_legacy_time_ratio']=1/ratio
        c['pass']=c['quality_pass'] and (name=='gsplat' or c.get('controlled_performance',{}).get('benefit',False))
        summary['comparisons'][name]=c
    lines+=['','## 最终验收','', 'PSNR 降幅 ≤ 0.15 dB、SSIM 降幅 ≤ 0.003；单项优化还需实测性能收益。', '', '|配置|参考|ΔPSNR|ΔSSIM|质量通过|综合通过|','|---|---|---:|---:|---|---|']
    for n,c in summary['comparisons'].items():lines.append('|{}|{}|{:+.5f}|{:+.6f}|{}|{}|'.format(n,c['reference'],c['delta_psnr'],c['delta_ssim'],c['quality_pass'],c['pass']))
    lines+=['','## 各视角最终质量','', '|配置|视角|张数|PSNR|SSIM|','|---|---|---:|---:|---:|']
    for name,e in metrics.items():
        for cam in ['00','01','02','03','04','09','10']:
            rr=[r for r in e['rows'] if r['image'].endswith('_'+cam)]
            lines.append('|{}|{}|{}|{:.5f}|{:.6f}|'.format(name,cam,len(rr),statistics.mean(r['psnr'] for r in rr),statistics.mean(r['ssim'] for r in rr)))
    # Only render comparison images after every training variant finishes evaluation.
    if len(metrics)==4:
        from PIL import Image,ImageDraw
        rows=metrics['gsplat']['rows'];mean=metrics['gsplat']['psnr']
        chosen=[min(rows,key=lambda r:r['psnr']),min(rows,key=lambda r:abs(r['psnr']-mean)),max(rows,key=lambda r:r['psnr'])]
        for label,row in zip(['low','mean','high'],chosen):
            dst=OUT/('comparison_'+label+'.jpg')
            if not dst.exists():
                ims=[Image.open(data(n)/'images_25000'/(row['image']+'.png')).convert('RGB') for n in NAMES]
                w=720;h=round(w*ims[0].height/ims[0].width);sheet=Image.new('RGB',(w*2,(h+36)*2),'white');draw=ImageDraw.Draw(sheet)
                for i,(name,im) in enumerate(zip(NAMES,ims)):
                    x=(i%2)*w;y=(i//2)*(h+36);sheet.paste(im.resize((w,h)),(x,y+36));draw.text((x+8,y+10),name+' / '+row['image'],fill='black')
                sheet.save(dst,quality=92)
            lines+=['','{} 示例：{}。'.format(label,row['image']),'','![{}]({})'.format(label,dst.name)]
    lines+=['','运行状态：'+str(state.get('state'))+'。详细 GPU 调度与启动时间见 scheduler_events.jsonl。','', '没有执行 Git push。']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    body=[]
    for line in lines:
        if line.startswith('!['):body.append('<img width="100%" src="'+line.split('](')[1][:-1]+'">')
        else:body.append(html.escape(line))
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>StreetGS 25k</title><body style="padding:25px;font:15px monospace"><pre style="white-space:pre-wrap">'+'\n'.join(body)+'</pre></body>')
    return summary
if __name__=='__main__':build()
