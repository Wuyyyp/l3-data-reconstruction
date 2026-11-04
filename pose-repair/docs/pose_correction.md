# 位姿修正

本文档说明匹配筛选、多相机联合优化和最终位姿字段约定。完整运行命令和交付入口见项目根目录的 `README.md`。

## 匹配与质量筛选

### 全量匹配前的 clip 预筛选

初始 pose 质量诊断不放在全量 LoMA 之后，否则坏 clip 已经消耗了大量匹配时间。完整匹配阶段
由 `match-delivery` 在同一个 Python 进程内执行预评估、语义分割和正式 LoMA 匹配：

```text
canonical pose/image inputs
  -> sampled semantic masks + LoMA preflight
  -> initial pose quality report
  -> full semantic masks
  -> full matching with the same LoMa GPU feature cache
```

预筛选按每个 intra baseline 和 inter relation 做时间分层采样，默认每组 20 对，默认总上限
800 对。它使用原始图像、初始 pkl pose、相机标定和抽样语义掩膜，因此可以在全量语义分割和
全量匹配之前发现明显异常的 clip。报告同时保留 intra-camera 和 inter-camera 的统计，不用一个平均值
掩盖某条相机边失效的问题。

预筛选输出：

```text
<run_output_root>/preflight_quality/
  summary.json
  report.md
  sampled_pairs.csv
  pair_metrics.csv
  raw_match_diagnostics.parquet

<run_output_root>/preflight_match_cache/
  index.json
  pair_*.parquet
```

预筛选的 LoMA 点对会按稳定的 camera/relation/frame pair key 缓存。随后全量语义分割复用抽样
图像的掩膜，正式匹配阶段读取这些点对，再应用动态类别掩膜；只有未出现在预筛选 cache 中的
图像对才重新运行 LoMA。因此预筛选不会导致同一图像对被 LoMA 计算两次。

配置位于车型配置的 `preflight_diagnostic` 段：

```yaml
preflight_diagnostic:
  enabled: true
  sample_pairs_per_relation: 20
  max_total_pairs: 800
  reuse_preflight_matches: true
  min_raw_matches_per_pair: 20
  min_ransac_inlier_ratio: 0.20
  reprojection_median_warn_px: 5.0
  reprojection_median_fail_px: 20.0
  max_bad_pair_fraction: 0.25
  min_bad_pairs_for_failure: 3
  min_failed_relation_fraction_for_failure: 0.20
  fail_action: stop_before_dynamic_compensation
```

预筛选始终使用语义掩膜，不提供无掩膜运行开关。已有有效语义 cache 会直接复用；抽样图像的
掩膜缺失时，预筛选只生成这些图像所需的掩膜，后续全量语义分割继续复用同一 cache。
语义 cache 中的 `class_masks` 是主数据，`masks` 只是由类别集合和 dilation 参数生成的二值
加速缓存。二值缓存被删除后会从 `class_masks` 自动恢复，不会重复执行模型推理。

预筛选报告使用三档状态：`PASS`、`WARN`、`FAIL`。P95 仍会写入报告，但不参与状态判定，
因为少量尾部噪声会显著影响 P95。少量异常帧、单个异常 pair 或单个异常关系只会产生 `WARN`，
不会直接导致整个 clip 失败。判定分两层：先对每个 `group_key`（同一相机关系和筛选关系）统计
超过 median 重投影阈值的 pair 数量和比例，再统计失败关系组占全部有效关系组的比例。
只有以下条件同时满足时才产生 `FAIL`：

```text
bad_pair_count >= min_bad_pairs_for_failure
bad_pair_count / valid_pair_count >= max_bad_pair_fraction
failed_relation_count / valid_relation_count >= min_failed_relation_fraction_for_failure
```

默认重投影 median 失败阈值为 20 px。单个关系至少要有 3 个异常 pair 且异常比例至少 25%，
并且失败关系至少占全部有效关系的 20%，clip 才因关系级重投影问题失败。这使得偶发误匹配或
单条相机边异常不会阻断后续三维场景重建，但大范围关系失效时仍会阻止完整流程。默认 `FAIL` 会在
全量语义分割和正式匹配前停止当前集成阶段，保留已经生成的预筛选 cache 和报告供排查；`WARN`
允许继续，并会写入后续运行记录。

重投影相关的状态判定只使用 retained median；P95、RMSE、ray gap 等仍写入报告供定位问题，
但 P95 不参与 `PASS/WARN/FAIL`，避免少量尾部噪声放大 clip 风险。除上述关系比例外，相机图
断连、有效 pair 数不足、整体 RANSAC 内点率过低或整体 retained 重投影 median 超过失败阈值
仍可独立触发 `FAIL`。

该诊断已接入 `match-delivery`，默认顺序为：

```text
normalize -> dynamic-compensation -> export-dynamic-pkl
          -> persistent-match(preflight -> semantic -> full matching)
          -> optimize -> QC/export
```

`FAIL + stop_before_dynamic_compensation` 在集成匹配阶段会返回非零状态并停止全量语义分割和
正式匹配；`WARN` 和 `PASS` 继续运行。排查时可配置 `fail_action: report_only` 只记录失败而不阻断。
单独运行 `diagnose-clip-quality` 仍可用于诊断，但它与后续正式匹配不共享进程内 GPU cache。

匹配后端由车型配置的 `matching.backend` 决定；当前交付配置使用 LoMa，但筛选和优化不依赖某个特定模型。语义掩膜在匹配前排除车辆、行人、骑行者、动物、植被和天空等不稳定区域。原始匹配只依赖图像、匹配后端和语义掩膜；更换 pkl 的 ego 字段后，可复用原始匹配，但必须重新计算几何筛选。

每个匹配 cache 目录包含：

```text
raw_matches.parquet
retained_matches.parquet
pair_metrics.csv
visualizations/
```

- `raw_matches.parquet`：语义掩膜过滤后的后端匹配结果；
- `retained_matches.parquet`：通过几何质量筛选、供优化使用的匹配；
- `pair_metrics.csv`：每个图像对的数量与筛选统计；
- `visualizations/`：抽样匹配和重投影质检图。

筛选会计算并检查：导航或 pkl 位姿可用性、三角化深度、三角化距离、三角化夹角、两条视线最近距离、双向重投影误差和图像 RANSAC 内点。重投影误差与视线最近距离可以使用 `adaptive_mad` 阈值；具体阈值、候选数量和约束来自车型配置的 `assumptions`，不应写死为某一车型的数值。

当一对图像的现有 pose 明显失效且配置启用 `pose_degraded_detection_enabled` 时，`pose_degraded_quality_mode: ransac_only` 只保留“图像 RANSAC 内点且三角化深度有效”的匹配，避免由失真的当前 pose 再次错误过滤。

关键字段：

- `quality_filter_valid`：当前质量筛选是否通过；
- `quality_reject_reason`：第一个拒绝原因；
- `image_ransac_inlier`：图像 RANSAC 是否判为内点；
- `precision_filter_valid`：兼容字段，当前与 `quality_filter_valid` 同值。优化器优先使用此字段选择 retained 匹配。

## 多相机联合优化

优化使用全部 cache 中的 retained 匹配，或按 `optimization.pair_sample_max_matches` 对每个图像对采样。设第 `i` 帧、相机 `k` 的初始 body-to-world 位姿为 `T_WB(i)`，固定安装外参为 `T_BC(k)`，则优化后的相机世界位姿由以下变量共同构成：

```text
T_WC(k, i) = T_WB(i) · ΔT_body(i) · T_BC(k) · ΔT_camera(k, i)
```

- 固定安装外参修正：每个相机一个受边界和 L2 正则约束的 SE(3) delta；
- 相机逐帧扰动：可选的每个 `camera + frame` 局部 delta；
- body/ego 逐帧扰动：可选的每个 frame 局部 delta。

优化以匹配几何误差为目标，并使用固定外参、逐帧扰动、平滑、长期均值和同帧共同模式约束来减少变量之间的混淆。启用 mode projection 时，周期性把共同模式吸收到更适合解释它的变量块：跨相机同帧共同趋势优先归入 body/ego，单相机长期稳定趋势优先归入固定安装外参，局部快速变化才由相机逐帧扰动解释。

## pkl 与交付约定

优化前主线读取动态补偿 pkl 的：

```text
ego2global_transformation_matrix_camera_reference_offset
```

优化输出的 pkl 保留原始 `ego2global_transformation_matrix`，并写入车型配置指定的 optimized ego 字段。当前交付配置使用：

```text
ego2global_transformation_matrix_camera_reference_offset_optimized
```

同时，`infos[*].sensors.cams.<camera>.extrinsic` 会写入优化后的固定相机安装外参。最终点云投影应使用优化后的 ego 字段、pkl 内相机外参，以及交付目录的 `compensated_perception/*.pcd`。

优化输出目录包含 `metrics/summary.json`、`metrics/edge_summary.csv`、`metrics/loss_history.csv`、逐图像外参和优化后的 pkl。评估应同时查看重投影误差的均值、中位数、P95、RMSE 和离群点诊断；少量严重失配会显著放大全量 RMSE，不能只凭 RMSE 判断结果。

### 优化后评价与精简输出

优化后仍对全部 retained 匹配逐条计算优化前后误差。全局统计、相机关系统计和异常计数均来自全量数据，不使用 QC 样本近似。`evaluation_backend: auto` 在 CUDA 可用且图像已去畸变时使用 Torch 批量评价；不满足条件时自动回退到 legacy CPU 评价。Torch 结果中非有限值或重投影误差大于 10 px 的长尾条目会再由 legacy 路径精确复算，复算数量记录在 `metrics/summary.json` 的 solver 信息中。

为避免优化完成后写出三份数百 MB、内容高度重复的逐匹配表，正式流程不再生成：

```text
observations/before_matches.parquet
observations/after_matches.parquet
observations/after_matches_reprojection_inliers.parquet
```

保留的评价产物为：

```text
metrics/summary.json
metrics/edge_summary.csv
metrics/reprojection_outlier_diagnostics.json
metrics/reprojection_outlier_groups.csv
observations/reprojection_outliers.parquet
observations/qc_matches.parquet
observations/qc_sample_manifest.json
```

- `summary.json` 和 `edge_summary.csv`：基于全部 retained 匹配的精确汇总；
- `reprojection_outliers.parquet`：保留所有超过硬阈值的异常匹配明细；
- `qc_matches.parquet`：按相机关系分层、确定性抽取的图像对及其优化前后投影信息；
- `qc_sample_manifest.json`：记录总图像对数、抽中图像对数和匹配数。

车型配置中的 `projection_qc_sample_ratio` 是唯一的优化后 QC 采样比例，当前默认值为 `0.01`。采样在联合优化输出阶段完成，后续重投影 QC 直接消费 `qc_matches.parquet`，不会再次采样。最终交付物的 `optimization_diagnostics/` 会包含异常明细、QC 样本和采样清单。

控制台会依次显示优化前全量评价、GPU 优化、优化后全量评价、精确统计、异常/QC 导出、逐图像外参导出和优化后 pkl 导出，便于区分 GPU 优化结束后的 CPU 后处理进度。

## 实现入口

```text
src/dongfeng_registration/staged_pipeline.py
src/dongfeng_registration/match_quality.py
src/dongfeng_registration/extrinsic_optimization.py
src/dongfeng_registration/result_interpolation_export.py
```

相关测试包括 `tests/test_semantic_matching.py`、`tests/test_match_quality.py` 和 `tests/test_extrinsic_optimization.py`。
