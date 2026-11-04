# 动态目标点云补偿

本文档说明动态目标点云补偿的实现约定。完整运行入口、环境和车型配置见项目根目录的 `README.md`。

## 输入与参考时刻

在完整 `run-delivery` 中，本阶段在集成匹配阶段的预评估之前执行。`match-delivery` 会在动态
补偿和 pkl 导出完成后，先运行预评估，再继续全量语义分割和正式 LoMA 匹配。预评估失败时
会停止后两步，但不会回滚已经完成的动态补偿输出。

补偿阶段只读取 canonical dataset：

```text
<vehicle>_canonical/
  result_interpolation_standard.pkl
  lidar/perception/*.pcd
  lidar/raw_lidars/<lidar>/*.pcd
```

标准 pkl 提供 ego 位姿、相机和 lidar 时间戳与外参、点云路径和 3D Box。实际参与处理的 raw lidar 与参考相机由车型的 `delivery.raw_lidars` 和 `delivery.reference_cameras` 配置决定，不能假定不同车型使用相同的传感器列表。

每帧的补偿目标时刻为：

```text
reference_time_s = mean(reference camera timestamps) + reference_time_offset_s
```

`reference_time_offset_s` 是车型和数据段相关参数，定义在 `delivery` 配置中。它不改写原始相机时间戳或点的逐点 `timestamp`；只用于动态补偿的目标时刻，以及动态补偿 pkl 中图像参考时刻 ego 位姿的插值。

## 补偿方法

原始 perception 点云是多路 lidar 融合结果。不同 lidar 扫到同一目标的时间不同，因此流程按 raw lidar 独立处理，再融合输出：

1. 将 raw lidar 点变换到当前帧 ego 坐标系。
2. 用当前帧 3D Box 选出各动态目标的框内点。
3. 按逐点 `timestamp` 对同一个 `frame + lidar + object` 分簇。
4. 为每个 `object_id + lidar` 构建按时间排序的 observation 轨迹。
5. 用相邻 observation 的 Box 运动作为初始运动。
6. 可用时以 Box 初值执行 ICP，或使用跨 lidar 的对象级运动候选；不满足质量条件时回退到 Box 运动。
7. 将相对运动插值到 `reference_time_s`，把框内点移到目标时刻。
8. 重新拼接所选 raw lidar 的完整点记录，写出融合 perception 点云。

补偿只改变被成功补偿的动态目标点的 `x/y/z`。每个所选 raw lidar 的点记录均会参与输出，不重采样，也不改写 `timestamp`、`intensity` 等非坐标字段。由于输出由 raw lidar 重新融合，点顺序和原始 perception 文件不应被视为契约。

## 时间戳分簇与回退

环扫 lidar 可能在一帧内的不同扫描时刻两次经过同一目标。实现按相邻点的时间间隔切分时间簇，避免对单个 Box 直接取平均时间。相关参数由脚本参数控制：

```text
--timestamp-cluster-max-gap-s
--timestamp-cluster-max-gap-fraction
--min-object-points
```

ICP 使用 Box 运动作为初值，并受最小点数、最大对应距离、fitness、RMSE、相对 Box 的平移/旋转 delta 和最大迭代次数约束。若没有可用于插值的相邻 observation，或运动超过 `--max-motion-translation-m`，该时间簇不会被补偿。每条 lidar/object 报告会记录 `status`、`motion_source`、ICP 指标和时间簇信息。

## 性能配置

车型配置中的 `dynamic_compensation` 控制预扫描、缓存和逐帧写出：

```yaml
dynamic_compensation:
  workers: 8
  track_workers: 4
  cloud_cache_max_gb: 8.0
  single_lidar_output_mode: hardlink
```

- `workers` 并行处理补偿和输出帧。
- `track_workers` 并行读取帧并构建 object tracks。
- `cloud_cache_max_gb` 是解压后 PCD 的内存上限。缓存采用全有或全无策略；完整数据超过上限时会清空缓存并回到二次读取，不会只缓存部分帧。
- `single_lidar_output_mode: hardlink` 对单 lidar 数据只写一次 PCD，并为逐 lidar 目录建立同文件硬链接；文件系统不支持硬链接时自动回退到复制。

安装可选的 `python-lzf` 后，`binary_compressed` PCD 使用原生 LZF 解码。未安装时自动使用纯 Python 兼容实现，补偿结果不变。

track 预扫描会保存每个 `frame + lidar + object` 的点索引和时间簇，补偿阶段直接复用，不再重复执行框内点筛选。启用 ICP/对象运动投票时也复用同一个帧级 PCD context，不会产生第三次读取。

## 阶段输出

`run-delivery` 的 `dynamic-compensation` 阶段写入 `delivery.dynamic_output_root`：

```text
compensated_perception/*.pcd
compensated_raw_lidars/<lidar>/*.pcd
dynamic_3d_boxes_json/*.json
dynamic_3d_boxes_by_lidar/<lidar>/*.json
summary.csv
timing.json
```

`timing.json` 记录 track 构建、逐帧补偿与写出、总耗时、worker 数和 PCD cache 命中情况。

随后 `export-dynamic-pkl` 会生成动态补偿 pkl。该 pkl 将 perception 的 `data_path` 指向补偿点云，写入补偿后的 Box，并新增图像参考时刻 ego 字段：

```text
ego2global_transformation_matrix_camera_reference_offset
```

最终优化会在此字段基础上生成 `ego2global_transformation_matrix_camera_reference_offset_optimized`。投影 QC 使用优化后的字段和动态补偿点云。

`export-deliverables` 成功后，补偿 perception 点云保存在：

```text
deliverables/<vehicle>_final/compensated_perception/*.pcd
```

全量 `run-delivery` 会在交付导出成功后删除 `dynamic_output_root/compensated_perception`，以避免保留重复的大文件。`compensated_raw_lidars`、Box JSON 和 `summary.csv` 仍留在 `outputs` 中用于诊断。若要从 `projection-qc` 或更早阶段重新运行，需要先重新执行动态补偿来恢复该中间点云目录。

## 诊断与实现入口

优先检查 `summary.csv` 中每帧的补偿数量、ICP/Box 使用数量和时间簇报告数量；需要定位单个目标时，查看对应的 `dynamic_3d_boxes_json/*.json` 的 `lidar_reports`。

核心实现：

```text
scripts/compensate_dynamic_raw_lidars.py
src/dongfeng_registration/dynamic_object_compensation.py
scripts/export_dynamic_compensated_result_interpolation_pkl.py
scripts/visualize_dynamic_compensated_pkl_projection.py
```

相关测试位于 `tests/test_dynamic_object_compensation.py`。
