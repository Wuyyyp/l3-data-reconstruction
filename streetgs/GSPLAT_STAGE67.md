# gsplat 第六、七阶段：完整训练与单项优化

## 实验安排

- 第六阶段：原 StreetGS CUDA 后端与 gsplat，使用相同初始化、相机序列、训练损失、mask 和 25000 步学习率计划。
- 第七阶段：分别开启 `packed=True`、`radius_clip=1`，各自对比普通 gsplat，暂不叠加。
- 先训练 4000 步；平均 PSNR 降幅 ≤ 0.15 dB、平均 SSIM 降幅 ≤ 0.003 且无数值异常才继续。
- 单项优化还需降低耗时或显存。通过的实验携带优化器状态从 4000 步继续到 25000 步。
- 25000 步基线不达标时停止候选优化的长训练。

## 为什么需要这些修改

1. **屏幕空间梯度桥接**：gsplat 提供像素坐标梯度，StreetGS 使用 NDC 坐标统计。因此 x/y 分别乘以图像宽/高的一半。背景使用逐像素绝对梯度的 L1 累积，放到 StreetGS 约定的第三列；车辆继续使用带符号梯度。这里没有改成 gsplat DefaultStrategy 的 L2 统计，也没有更换 StreetGS 的增密阈值。
2. **packed 映射**：packed 只输出可见项，用 `gaussian_ids` 恢复完整高斯数组的统计及 radii，避免索引错位。
3. **白色背景混合**：gsplat 1.5.3 的 packed 路径对背景张量形状有断言问题。统一先渲染黑色背景，再加 `(1-alpha)*background`，同时保留梯度；深度及其他特征不混入白色。
4. **显式完整训练开关**：新增 `render.gsplat_densification=True`，保留原先固定点集测试及默认旧后端。原 StreetGS 负责 clone/split/prune、opacity reset 和优化器状态调整。
5. **checkpoint 元数据补全**：旧 checkpoint 不保存车辆 `random_initialization` 标志。加载初始化时按原 PLY 点数规则恢复，后续 checkpoint 显式保存，确保车辆增密路径一致。
6. **可审计实验驱动**：直接提取当前 train.py 的损失、mask 和拓扑更新语句；轮换后端执行顺序，记录逐步耗时、点数、梯度有限性和轨迹变化。每次渲染/拓扑更新分别固定随机种子。

## 保留的流程及解释边界

- 600 步首次增密，3000 步 opacity reset，3000 步后车辆轨迹学习率解冻。
- 短实验也保留 `densify_until_iter=25000`，不把 actor-region loss 提前到 4000 步。
- 每组运行恰好执行相应数量的参数更新，包括末步；两后端一致。原 train.py 在最终迭代跳过更新的习惯没有搬到测试驱动中。
- gsplat 可见性/半径判定与原内核并非逐 bit 一致，因此长训练点数不要求相等；要求质量和数值行为达标。
- 21 张评估图来自训练视图，使用当前生产训练 mask；不是历史上扩大机盖/后视镜并排除动态区域的评估口径，也不是独立测试集。
- 性能验收用同一个 checkpoint，在相同相机上轮换执行顺序，3 轮预热后统计 36 组配对 forward/loss/backward。条件为耗时中位数降低超过 1% 且 bootstrap 比值上界小于 1，或瞬时显存降低超过 5%。
- 瞬时显存是该步峰值减去开始前已分配显存，不是包含全部常驻模型和优化器的总显存。

## 结果位置

`output/gsplat_stage67/`：`status.json`、`pipeline.log`、逐配置 `curve.jsonl`、`eval_STEP.json`、`iteration_STEP.pth`、`gate_*.json`、`benchmark_*.json`、`REPORT.md` 及 `report.html`。

报告随实验进度更新。是否通过以实际 gate 文件为准，启动实验不表示验收通过。

没有执行 Git push。每次上传仍需用户针对当次 push 的明确命令。
