from pathlib import Path
import json, math, html, difflib, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw

ROOT=Path('/mnt/zhoukaixuan_workspace/code/street_gaussians');out=ROOT/'output/gsplat_train_check'
done=json.loads((out/'complete.json').read_text());assert done['actual_updates_per_backend']==500
one=json.loads((out/'one_step.json').read_text());diag=json.loads((out/'position_gradient_diagnostic.json').read_text())
synthetic=json.loads((out/'synthetic_backward.json').read_text())
ev=json.loads((out/'evaluation.json').read_text());train=[json.loads(s) for s in (out/'train_curve.jsonl').read_text().splitlines()]
assert len(train)==1000 and len(ev)==168
assert {b:sum(r['backend']==b for r in train) for b in ['legacy','gsplat']}=={'legacy':500,'gsplat':500}
assert all(r['background_position_lr']>0 for r in train)
steps=sorted(set(r['step'] for r in ev));backs=['legacy','gsplat']
averages={b:{step:{k:float(np.mean([r[k] for r in ev if r['step']==step and r['backend']==b])) for k in ['psnr','ssim']} for step in steps} for b in backs}
final={b:{r['image']:r for r in ev if r['step']==500 and r['backend']==b} for b in backs}
names=sorted(final['legacy']);per_image=[]
for name in names:
    a=final['legacy'][name];b=final['gsplat'][name]
    per_image.append(dict(image=name,legacy_psnr=a['psnr'],gsplat_psnr=b['psnr'],delta_psnr=b['psnr']-a['psnr'],
        legacy_ssim=a['ssim'],gsplat_ssim=b['ssim'],delta_ssim=b['ssim']-a['ssim']))
by_camera=[]
for cam in ['00','01','02','03','04','09','10','all']:
    rows=per_image if cam=='all' else [r for r in per_image if r['image'].endswith('_'+cam)]
    by_camera.append(dict(camera=cam,count=len(rows),**{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='image'}))
gap=averages['gsplat'][500]['psnr']-averages['legacy'][500]['psnr']
sgap=averages['gsplat'][500]['ssim']-averages['legacy'][500]['ssim']
summary=dict(averages=averages,by_camera=by_camera,per_image=per_image,delta_psnr=gap,delta_ssim=sgap,
    initial_background_position_lr=one['losses']['legacy']['background_position_lr'],
    initial_track_position_lr=one['losses']['legacy']['track_position_lr'],
    numerical_stability=done,position_gradient_diagnostic=diag,
    caveats=['Fixed point count, no densification/pruning/opacity reset','Training-view validation, not held-out evaluation',
             'Track parameters retain original 3000-step warmup: gradients tested, updates inactive during 500 steps',
             'SH degree schedule and all losses preserve original 25000-step horizon',
             'Native rasterizers differ in covariance clipping derivatives, alpha caps and tile coverage'])
(out/'summary.json').write_text(json.dumps(summary,indent=2))

fig,axs=plt.subplots(2,2,figsize=(13,8),dpi=150)
colors={'legacy':'#3467c9','gsplat':'#e17b2e'}
for b in backs:
    rows=[r for r in train if r['backend']==b]
    losses=np.array([r['loss'] for r in rows]);window=25
    axs[0,0].plot(np.arange(window,501),np.convolve(losses,np.ones(window)/window,mode='valid'),label=b,color=colors[b])
    axs[0,1].plot(steps,[averages[b][s]['psnr'] for s in steps],'-o',label=b,color=colors[b])
    axs[1,0].plot(steps,[averages[b][s]['ssim'] for s in steps],'-o',label=b,color=colors[b])
axs[0,0].set(title='Training loss (25-step moving mean)',xlabel='Actual optimizer updates',ylabel='Loss')
axs[0,1].set(title='Fixed 21 training views: masked PSNR',xlabel='Actual optimizer updates',ylabel='PSNR (dB)')
axs[1,0].set(title='Fixed 21 training views: masked SSIM',xlabel='Actual optimizer updates',ylabel='SSIM')
axs[1,1].bar([r['camera'] for r in by_camera[:-1]],[r['delta_psnr'] for r in by_camera[:-1]],color='#397fa0')
axs[1,1].axhline(0,color='black',lw=.8);axs[1,1].set(title='500 updates: gsplat minus legacy',xlabel='Camera ID',ylabel='PSNR difference (dB)')
for ax in axs.flat:
    ax.grid(alpha=.2)
    if ax is not axs[1,1]:ax.legend()
fig.tight_layout();fig.savefig(out/'training_curves.png');plt.close(fig)

examples=out/'examples';examples.mkdir(exist_ok=True)
for r in per_image:
    name=r['image'];folder=out/'images'/'500'
    panels=[('GT',Image.open(folder/(name+'_gt.png'))),('Initial (shared)',Image.open(out/'images'/'0'/(name+'_legacy.png'))),
            (f'Legacy: {r["legacy_psnr"]:.3f} dB',Image.open(folder/(name+'_legacy.png'))),
            (f'gsplat: {r["gsplat_psnr"]:.3f} dB',Image.open(folder/(name+'_gsplat.png')))]
    w=480;h=round(panels[0][1].height*w/panels[0][1].width)
    sheet=Image.new('RGB',(4*w,h+45),'white');d=ImageDraw.Draw(sheet)
    for i,(label,im) in enumerate(panels):
        d.text((i*w+8,8),name+' / '+label,fill='black');sheet.paste(im.resize((w,h)),(i*w,40))
    sheet.save(examples/(name+'.jpg'),quality=94)

columns=['camera','legacy_psnr','gsplat_psnr','delta_psnr','legacy_ssim','gsplat_ssim','delta_ssim']
table='| Camera | Legacy PSNR | gsplat PSNR | Delta PSNR | Legacy SSIM | gsplat SSIM | Delta SSIM |\n|---|---:|---:|---:|---:|---:|---:|\n'
htmlrows=[]
for r in by_camera:
    cells=[r['camera']]+[f'{r[k]:.5f}' for k in columns[1:]]
    table+='| '+' | '.join(cells)+' |\n'
    htmlrows.append('<tr>'+''.join('<td>'+s+'</td>' for s in cells)+'</tr>')
lossdelta=one['losses']['gsplat']['loss']-one['losses']['legacy']['loss']
text=f'''# StreetGS / gsplat：1 步与 500 步固定点集训练验证

## 实验结果

同一份 sample_00380 初始化快照、相机采样序列、随机状态、训练 mask、损失和学习率调度；物理 GPU 1。两个后端分别执行 1 次独立诊断更新，然后恢复同一份初始化，各执行 500 次实际 optimizer 更新。未 push。

固定 21 张训练视角上，500 步后 legacy PSNR {averages['legacy'][500]['psnr']:.5f} dB、gsplat {averages['gsplat'][500]['psnr']:.5f} dB，差 {gap:+.5f} dB；SSIM 差 {sgap:+.6f}。

两组训练完成，loss 未出现 NaN/Inf，定期检查的梯度和最终参数均为有限值，点数保持不变。此结果用于固定点集训练验收，不代表完整增密训练或 25000 步质量已经通过。

## 实验约束

- 关闭增密、剪枝（含静态剪枝）、opacity reset；直接使用 StreetGS 的 Adam 和参数组。
- 保留原 25000 步调度，不把学习率在 500 步内提前衰减到终值。densify_until_iter 仍为 25000，避免意外提前启用 actor-region loss。
- 掩码与损失直接执行当前 train.py 的 AST 代码块，保存为 loss_block.py；没有另写一套近似 loss。
- 两后端每一步恢复同一份 RNG，因此天空随机射线扰动相同。每个模型都从相同初始化重新开始。
- 评估取第 0、40、99 帧 × 7 个视角，沿用训练有效区域 mask，图像为训练加载分辨率。此处不是此前扩大机盖/后视镜并排除动态区域的历史评估协议，也不是独立测试集。
- 初始快照文件及既有 25000 步 checkpoint 均未修改。

## 1 步：梯度与参数更新

首个相机 {one['image']}。loss：legacy {one['losses']['legacy']['loss']:.9f}，gsplat {one['losses']['gsplat']['loss']:.9f}，差 {lossdelta:+.9f}。

背景 XYZ 梯度整体相对 L2 差约 16.55%，但差异能量的 {diag['difference_energy_in_clipped_fraction']*100:.6f}% 集中在 {diag['clipped_active_points']} 个触发投影视锥裁剪的有效点。其余背景点梯度相对差 {diag['inside']['relative_l2']*100:.5f}%，余弦相似度 {diag['inside']['cosine']:.10f}。

小场景验证中，未裁剪区域的位置、尺度、旋转、不透明度及 SH 梯度相对差均小于 2e-6。在构造的裁剪样例上，z 梯度相对有限差分的误差约为旧内核 88.7%、gsplat 0.85%。这是对该样例的数值验证，不是声称所有梯度分支均已覆盖。

原因：投影协方差中 tx = z * clip(x/z)。发生裁剪时 tx 随 z 变化，旧 CUDA backward 的 z 梯度缺少这项链式贡献；gsplat 的对应分支包含它。未为了匹配旧梯度而修改 gsplat 内核。

初始化高斯是各向同性，旋转方向理论上不影响形状。旧内核产生了约 1e-9 量级的背景旋转梯度，gsplat 为零；Adam 的极小 epsilon 会放大这种噪声，使第一步参数更新不逐项一致。因此不能只凭参数逐项差异判定适配错误。

车辆轨迹位姿保留原 3000 步 warmup：前 500 步学习率为 0。本次确认其梯度能传回，但没有验证轨迹参数开始更新后的优化行为；需要后续跨过 3000 步的实验。

## 本次修复

1. gsplat 训练模式不裁剪 RGB，保留原 StreetGS 的训练语义；评估模式继续裁剪。合成高亮颜色测试确认 >1 的训练 RGB 和梯度保留。
2. 增加 optim.fixed_gaussians 开关。gsplat 只有在固定点集模式才能训练；未接入 screen-space absgrad 与增密统计时，不允许进入完整增密训练。
3. 修正 checkpoint 恢复顺序：先恢复 spatial_lr_scale，再创建位置学习率调度。否则直接加载到新模型时，背景位置学习率会被固化为 0。修正前诊断归档于 before_restore_fix，没有计入正式 500 步结果。

这一问题暴露在“直接恢复到尚未 create_from_pcd 的新模型”路径；此前常规训练会先从点云初始化并设置场景尺度，不能据此推断此前 25000 步训练的位置参数也被冻结。

修正后首步背景位置 LR = {one['losses']['legacy']['background_position_lr']:.10f}，两组背景位置均发生实际更新。测试脚本显式检查非零位置学习率和位置更新，防止两边一起冻结而得到虚假的一致结果。

## 500 步逐视角结果

每个视角 3 张图，指标为逐图算术平均。

{table}
## 代码与复现

主要修改：lib/models/gsplat_renderer.py、lib/models/gaussian_model.py、lib/config/config.py、train.py。
验证脚本：scripts/compare_fixed_training.py、scripts/check_fixed_training.py。

```bash
cd /mnt/zhoukaixuan_workspace/code/street_gaussians
CUDA_VISIBLE_DEVICES=1 TRAIN_CHECK_PHASE=all bash scripts/run_gsplat_eval.sh \\
  scripts/compare_fixed_training.py \\
  --config output/sample_00380_pose_ba_25000_20261001_172042/evaluate.yaml
```

普通训练配置需同时设置 render.backend: gsplat 和 optim.fixed_gaussians: true。增密训练尚未实现，不应解除此保护。

复现脚本写 output/gsplat_train_check；其中 legacy/iteration_500.pth 与 gsplat/iteration_500.pth 含 optimizer 状态。脚本做满 500 次更新；原 train.py 的最终迭代不执行 optimizer.step，本实验没有沿用这一计数习惯。

没有测试增密/剪枝、opacity reset、跨 3000 步的轨迹更新、长期收敛或其他场景。单卡交替执行的耗时包含验证检查，不作为 gsplat 加速比例结论。
'''
(out/'REPORT.md').write_text(text);(ROOT/'docs/GSPLAT_FIXED_TRAINING_VALIDATION.md').write_text(text)
ordered=sorted(per_image,key=lambda r:r['delta_psnr']);picks=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
cards=''.join(f'<h3>{r["image"]}：PSNR 差 {r["delta_psnr"]:+.4f} dB</h3><img src="examples/{r["image"]}.jpg">' for r in picks)
gallery=''.join(f'<details><summary>{r["image"]} / PSNR 差 {r["delta_psnr"]:+.4f} dB</summary><img loading="lazy" src="examples/{r["image"]}.jpg"></details>' for r in per_image)
page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>gsplat 固定点集训练验证</title><style>
body{{font:16px/1.65 system-ui;color:#18304a;max-width:1500px;margin:40px auto;padding:0 24px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #cbd5e1;padding:9px;text-align:right}}img{{max-width:100%;height:auto}}.note{{padding:20px;background:#eef4fb;border-radius:12px}}details{{margin:15px 0;padding:12px;border:1px solid #cbd5e1;border-radius:9px}}summary{{cursor:pointer}}</style>
<h1>StreetGS → gsplat：1 步＋500 步训练验证</h1><p>sample_00380 · 相同初始化 / 相机顺序 / RNG / mask / loss · GPU 1 · 未 push</p>
<div class="note"><b>两个后端各完成 500 次实际参数更新，点数固定。</b><br>21 张固定训练视角：PSNR {averages['legacy'][500]['psnr']:.4f} → {averages['gsplat'][500]['psnr']:.4f} dB（{gap:+.4f}）；SSIM 差 {sgap:+.6f}。<br>本次不包含增密或 3000 步以后轨迹参数更新的验收。</div>
<h2>收敛曲线</h2><img src="training_curves.png"><p>Loss 为 25 步滑动平均；PSNR/SSIM 在同一组 21 张训练图像上评估，使用训练有效区域 mask。</p>
<h2>1 步测试发现并核对的问题</h2><p>背景位置梯度的差异集中在 7 个边缘裁剪点；普通区域相对差约 0.0135%。合成有限差分支持 gsplat 的裁剪求导。另修复了直接恢复快照时，位置学习率调度先于场景尺度恢复的问题，确保两组位置参数实际更新。</p>
<p>车辆轨迹原有 3000 步 warmup 保持不变，因此本次验证其梯度，但没有覆盖它开始更新后的行为。</p>
<h2>500 步各视角指标</h2><table><tr><th>Camera</th><th>Legacy PSNR</th><th>gsplat PSNR</th><th>ΔPSNR</th><th>Legacy SSIM</th><th>gsplat SSIM</th><th>ΔSSIM</th></tr>{''.join(htmlrows)}</table>
<h2>样例</h2><p>从左到右：真实图 / 共同初始化 / legacy 500 步 / gsplat 500 步。以下依次为 PSNR 差最小、中位、最大的样例。</p>{cards}
<h2>全部 21 张</h2>{gallery}<p><a href="REPORT.md">完整说明</a> · <a href="summary.json">汇总指标</a> · <a href="one_step.json">梯度与更新明细</a> · <a href="position_gradient_diagnostic.json">边缘梯度诊断</a></p></html>'''
(out/'report.html').write_text(page)
names_code=['lib/config/config.py','lib/models/gsplat_renderer.py','lib/models/gaussian_model.py','train.py',
    'scripts/compare_fixed_training.py','scripts/check_fixed_training.py','scripts/check_gsplat_boundaries.py','docs/GSPLAT_FIXED_TRAINING_VALIDATION.md']
patch=[]
for name in names_code:
    old=out/'prechange'/name
    a=old.read_text().splitlines(True) if old.exists() else []
    patch.extend(difflib.unified_diff(a,(ROOT/name).read_text().splitlines(True),fromfile='before/'+name,tofile='after/'+name))
(out/'training_changes.patch').write_text(''.join(patch))
(out/'code_manifest.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names_code},indent=2))
print(json.dumps(dict(delta_psnr=gap,delta_ssim=sgap,averages=averages,by_camera=by_camera),indent=2))
