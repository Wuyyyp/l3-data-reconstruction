# 前向 LiDAR 语义动态过滤融合

`scripts/fuse_lidar_optimized_pose.py` 将多帧前向 LiDAR 补偿点云融合到世界坐标系，并在融合前按七路相机语义类别图移除动态物体点。

## 输入约定

- 前向 `*_compensation` PCD 已处于车体（ego）坐标系；只应用扫描参考时刻的 `ego2global_transformation_matrix`，不再乘 LiDAR 安装外参。
- PCD 文件名的数值时间戳是扫描对齐参考时刻。脚本按该时间戳从 pkl 插值或读取 ego 位姿。
- 语义缓存必须包含 `class_masks/<timestamp>.png` 和同名 `metadata/<timestamp>.json`。metadata 中的 `class_id_to_name` 用于解释类别 ID。

## 筛选规则

每帧点云按相机内外参投影到七路相机。任一点只要落入任一可见相机的动态类别像素即被移除；没有落入图像、或对应类别图缺失的点保留，并记录在 `frame_filter_stats.csv`。

默认移除：`person`、`rider`、`bicycle`、`motorcycle`、`car`、`bus`、`truck`、`train`、`trailer`、`caravan`、`other_vehicle`、`animal`。默认在类别掩码边界额外膨胀 5 像素。天空和植被不属于该融合筛选的动态类别。

## 运行示例

```powershell
python scripts\fuse_lidar_optimized_pose.py `
  --pkl outputs\T3_joint_dynamic_pose\deliverables\result_interpolation_optimized.pkl `
  --lidar-root data\T3_canonical\lidar\front_lidar_compensation `
  --point-coordinate-frame ego `
  --ego-field ego2global_transformation_matrix `
  --semantic-cache-root data_cache\T3\semantic_masks `
  --output-dir outputs\T3_front_lidar_optimized_ego_world_dynamic_filtered `
  --output-ply-name optimized_ego_dynamic_filtered_world_fused.ply `
  --frame-pose-csv-name frame_filter_stats.csv `
  --summary-name dynamic_filtered_summary.json `
  --count 598
```

输出目录包含：

- `optimized_ego_dynamic_filtered_world_fused.ply`：世界坐标系融合点云，保留输入的 `intensity`、`ring`、`timestamp` 等结构化字段；
- `frame_filter_stats.csv`：逐帧输入、范围筛选、语义删除、保留点数及七路 mask 加载状态；
- `dynamic_filtered_summary.json`：位姿字段、坐标系语义、筛选类别、点数汇总和运行时间。
