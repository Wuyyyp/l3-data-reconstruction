# 保留 MoGe / DVP 特性的迭代优化

新增 `balanced` 实验策略，默认仍为 `legacy`。没有更改 MoGe 推理、初始深度/法线、稀疏深度范围、APD 强/弱纹理传播、RANSAC、自适应窗口、边缘/标签支持或 OpenMVS 融合阈值。

MoGe 目前作为 FIRST_INIT 的深度与世界法线先验；后续尺度继承优化后的深度、法线与弱纹理状态，并非每一轮强制贴回 MoGe。该传递方式保持不变。保留算法机制不等于已经证明减少轮数后质量不变。

以 960×640 画布为例：

| 尺度 / 阶段 | legacy 内部迭代 | balanced 内部迭代 |
|---|---|---|
| 480×320 初始化 | 3 | 3 |
| 480×320 几何更新 | 3 + 3 + 3 | 3 + 3 |
| 960×640 升尺度精化 | 3 | 3 |
| 960×640 几何更新 | 3 + 3 + 3 | 3 + 3 + 2 |

总外轮 8→7，内部更新 24→20。低尺度保留两轮完整几何约束；最终尺度保留三轮几何更新，前两轮各三次内部更新，最后一轮用两次更新收尾。弱纹理判别半径仍从 6→4→2 收紧。高于两个尺度时，中间尺度采用与粗尺度相同的安排；尺度数量仍由现有图像尺寸逻辑决定。

第一版 24→15 的策略在 21 张图上更快，但连续性代理指标与几何一致性比例变差，因此没有采用。当前配置保留更充分的最终尺度精化。

## 配置

- `legacy`：原有轮次与原地深度发布，用作历史基线。
- `legacy-sync`：轮次与 legacy 完全相同，只改为整轮结束后发布深度，用于隔离深度版本语义的影响。
- `balanced`：上述精简轮次，加整轮深度发布。
- `openmvs`：三尺度非几何估计，各四次内部更新；最终尺度两轮几何估计，每轮一次内部更新，加整轮深度发布。对 960×640 画布为 240×160→480×320→960×640。MoGe 在 240 层初始化，后续两层启用 APD。480/960 的 RANSAC 与旋转参数按原分辨率层级保留，不因新增 240 层整体错移。此配置额外检查 `DVP_OPENMVS_SCHEDULE_V1` 二进制标记。

新策略每张图先写 `depths.next.dmb`，所有 worker 结束计算后再发布为 `depths.dmb`，发布结束再做一次 barrier。因此几何轮中每张图使用同一版本的邻居深度。只有深度被跨视图读取；法线、弱纹理等本视图状态保持原有更新方式。新策略 barrier 轮询由 1 秒缩短为 20 毫秒，减少多卡等待；legacy 保持原行为。中断后应从 stage 2 重跑，不能在半轮内恢复。

## 使用

先重新编译项目。然后通过已有流水线选取配置，并使用独立结果目录：

```bash
bash mvs_process/run_batch_mvsnet_to_openmvs_ply.sh \
  --sample-root /path/to/prepared \
  --result-root /path/to/dvp_balanced \
  --gpu-ids 6,7 --dvp-profile balanced
```

直接调用 APD 时可设置环境变量：

```bash
DVP_ITERATION_PROFILE=balanced DVP_SKIP_FUSION=1 build/APD /path/to/scene 0
```

旧程序不会理解此配置，流水线对非 legacy 配置检查 `DVP_ITERATION_POLICY_V1` 二进制标记。请求记录包含新策略名称，防止 `--resume` 误用其他策略的完成标记。legacy 仍兼容旧请求记录。直接多卡调用时须与旧程序一样在新场景或已清理的 APD/APD_parallel_barrier 目录启动，不可复用旧 barrier 标记。

日志新增 `DVP_PASS_START` / `DVP_PASS_DONE`，记录尺度、几何开关、内部迭代、弱纹理判别半径与整轮墙钟时间。

## 验证

```bash
c++ -std=c++11 scripts/test_iteration_schedule.cpp -o /tmp/test_dvp_schedule
/tmp/test_dvp_schedule
bash -n mvs_process/run_batch_mvsnet_to_openmvs_ply.sh
```

真实数据应依次对比 legacy、legacy-sync、balanced，固定相同 MoGe priors、相机、mask、邻居和 GPU 数量。先看墙面/道路的连续性、远处有效深度与孔洞、边缘错误连接，再看融合点云；点数或平均深度差不能独立证明质量。只有通过代表性场景的质量对照后，才考虑将 balanced 设为生产默认。减少的迭代数不能直接换算成加速倍数。

原有 CUDA 随机初始化使用 `clock64()`，因此即使配置相同也不会保证逐像素重现；整轮深度发布只隔离读写版本，并未更改随机数机制。

## OpenMVS 调度与双融合输出

在原批处理命令中使用 `--dvp-profile openmvs`。保持默认 `legacy` 不变。该配置改变调度，APD CUDA 核心与 DVP 几何代价保持不变，不承诺与原生 OpenMVS 相同速度或相同质量。

另一条分支可用 `export_dvp_to_colmap.py` 将同一批原始 DVP 输出写成 COLMAP workspace，再调用原生 `stereo_fusion`：

```bash
python3 mvs_process/export_dvp_to_colmap.py \
  --run /path/to/result/09_dvp \
  --source /path/to/prepared/03_converted \
  --model /path/to/prepared/07_sparse/triangulated_text \
  --depth-ranges-json /path/to/result/sparse_depth_ranges_openmvs.json \
  --output /path/to/result/11_colmap_fusion

colmap stereo_fusion \
  --workspace_path /path/to/result/11_colmap_fusion \
  --workspace_format COLMAP --input_type geometric \
  --output_path /path/to/result/11_colmap_fusion/colmap_filtered_fused.ply \
  --StereoFusion.num_threads 8 --StereoFusion.max_image_size 960 \
  --StereoFusion.min_num_pixels 5 --StereoFusion.max_reproj_error 2 \
  --StereoFusion.max_depth_error 0.01 --StereoFusion.max_normal_error 10 \
  --StereoFusion.check_num_images 50
```

导出依赖 numpy、opencv 和 scipy。输出必须是新目录。保留相同有效 mask、逐视图稀疏深度范围和深度导出的相机法线；两分支均直接读取 APD 深度，COLMAP 不读取经过 OpenMVS 后处理的深度。COLMAP 内参与外参使用 DVP 相机的高精度参数并去除 padding；原稀疏点及 tracks 仅用于邻图选择。COLMAP 根据稀疏重叠选择最多 50 个邻图，OpenMVS 使用已有 20 视图列表，这一差异属于各自融合策略。

COLMAP 路线的过滤指 StereoFusion 内部的深度、重投影、法线和支持像素过滤。没有运行 COLMAP PatchMatch，没有伪造 NCC 分数，也没有复现与 PatchMatch 最后一次 sweep 耦合的光度预过滤。文件后缀 `.geometric.bin` 是融合读取格式，不表示运行过 COLMAP 几何 PatchMatch。两分支结果差异不能归因于某一个阈值。


## 新增legacy19与CPU流水线

legacy19保留原in-place发布：480为3＋3/3/3，960为3＋2/2，共19次。不是balanced策略，也不会冻结几何深度。CPU流水线仅可用于单worker的legacy/legacy19，原默认不变。编译、开关与质量限制见[CPU_PIPELINE.zh-CN.md](CPU_PIPELINE.zh-CN.md)。
