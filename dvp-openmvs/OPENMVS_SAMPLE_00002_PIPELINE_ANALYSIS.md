# sample_00002_f5_105_left：旧版 OpenMVS 全流程审计

> 本文只分析已经存在的输入、日志、审计文件和归档结果，没有重新执行 COLMAP、OpenMVS 或点云计算。

## 1. 路径与最终结果

完整样本名：

```text
sample_00002_clip_M18-2_07_20251202110510_DF_f5_105_left
```

| 内容 | 路径或结果 |
|---|---|
| 输入目录 | `/mnt/nuplan/l3data-reconstruction-bingxing/samples/sample_00002_clip_M18-2_07_20251202110510_DF_f5_105_left/converted` |
| 工作目录 | `/mnt/nuplan/l3data-reconstruction-bingxing/tem-test/colmap+openmvs/batch8_first8_roadmesh/results/sample_00002_clip_M18-2_07_20251202110510_DF_f5_105_left` |
| 最终归档 | `/mnt/l3-data-reconstruction/samples/sample_00002_clip_M18-2_07_20251202110510_DF_f5_105_left` |
| 最终融合点云 | 归档中的 `scene_dense.ply` |
| 最终点数 | 2,546,829 |
| 原始深度图 | 归档 `raw_depth_dmap/` 中 707 个 DMAP |
| 归档完成时间 | 2026-09-04 21:16:09 |

归档的 `ARCHIVE_COMPLETE` 标记为 `PASS`，并明确记录 `raw_dmaps=707` 和 `ply=scene_dense.ply`。

## 2. 数据流

```text
converted 输入
  ├─ images / intrinsics / extrinsics / ego_pose
  ├─ dynamic_mask / sky_mask
  └─ lidar_depth / pointcloud.npz（输入中存在）
          │
          ▼
生成 707 张图像清单和 ±20 帧候选图对
          │
          ▼
生成 OpenMVS 忽略掩膜
          │
          ▼
COLMAP SIFT 特征提取与指定图对匹配
          │
          ▼
写入已知相机内外参，固定姿态三角化稀疏点
          │
          ▼
InterfaceCOLMAP：COLMAP 模型 → scene.mvs
          │
          ▼
OpenMVS PatchMatch：707 个深度/法线/置信度 DMAP
          │
          ▼
深度图后处理 + 多视图几何一致性融合
          │
          ▼
scene_dense.ply（2,546,829 点，含 RGB 和法线）
          │
          ▼
归档 raw_depth_dmap、scene_dense.ply、sky_mask 等
```

## 3. 输入读取

处理范围是帧 `000000` 到 `000100`，共 101 帧；每帧读取 7 个相机，共 707 张图像。

| 相机编号 | 相机 | 数量 | 输入分辨率 |
|---|---|---:|---:|
| 00 | left_front | 101 | 1920×1280 |
| 01 | right_front | 101 | 1920×1280 |
| 02 | rear | 101 | 1920×1280 |
| 03 | left_rear | 101 | 1920×1280 |
| 04 | right_rear | 101 | 1920×1280 |
| 09 | center_fov30 | 101 | 1920×1080 |
| 10 | center_fov120 | 101 | 1920×1080 |

输入清单说明图像已经去畸变并转换为 pinhole 模型。相机位姿来自 `ego_pose`，采用 `camera_to_world`；进入 COLMAP 时转换成：

```text
R_w2c = R_c2w^T
t_w2c = -R_w2c × C
```

位姿审计通过：相机中心往返误差最大约 `2.716e-13 m`，旋转误差最大约 `3.006e-15`。因此这条流程使用的是输入给定的相机姿态。

输入目录中还存在 `lidar_depth` 和 `pointcloud.npz`，但现有日志表明 OpenMVS 的主要输入是 COLMAP 导出的 `scene.mvs`、原图、掩膜和邻居列表；没有证据表明它直接把 `lidar_depth` 当作 PatchMatch 深度初值。

## 4. 图对和邻居选择

流程先在全部 7 个相机之间生成帧距离不超过 ±20 的候选图对：

- 请求并写入 COLMAP 数据库的图对：90,811
- 有正匹配证据的图对：24,813
- 每个 OpenMVS 参考图最终指定 20 个源视图
- 每个参考图至少包含 6 个跨相机源视图，平均约 6.113 个
- 实际没有使用零匹配证据的兜底邻居

邻居选择优先采用有 COLMAP 稀疏匹配证据的视图，同时保证时间邻近和跨相机覆盖。

## 5. 掩膜处理

天空分割采用 `nvidia/segformer-b0-finetuned-ade-512-512`。安全天空掩膜条件包括：

- 分类结果为天空；
- 天空概率至少 0.6；
- 天空相对建筑物的概率差至少 0.25；
- 只保留与图像顶部连通的区域；
- 在 1920×1280 尺度腐蚀 5 像素，保留不确定的天际线边界。

安全天空像素占全部输入像素约 15.34%。相机 10 还叠加了车体/设备遮挡区域。传给 OpenMVS 的掩膜将忽略区域标记为 255，并通过 `--ignore-mask-label 255` 排除。

## 6. COLMAP 特征、匹配和固定姿态稀疏重建

COLMAP 数据库包含：

- 7 个相机模型；
- 707 张图像；
- 1,898,555 个 SIFT 特征点，平均每张约 2,685 个；
- 90,811 条请求图对；
- 24,813 个具有有效几何匹配的图对。

之后用已知内外参构造 COLMAP 模型，并通过 `point_triangulator` 生成稀疏点。`InterfaceCOLMAP` 读取到：

- 707 张已标定图像；
- 160,529 个稀疏点；
- 641,604 次稀疏点观测；
- 平均每个稀疏点约 4.00 个视图。

高层配置记录 `bundle_adjustment: false`。三角化日志内部仍出现若干 bundle adjustment report；结合固定姿态审计通过，更可能是三角点/轨迹细化步骤，而不是释放相机位姿进行全局 BA。

## 7. COLMAP 转 OpenMVS

实际接口命令为：

```bash
InterfaceCOLMAP \
  -w <work>/02_openmvs/global \
  -i <work>/02_openmvs/colmap_input \
  -o scene.mvs \
  --image-folder <input>/converted/images \
  --binary 1 --archive-type 2 -v 2
```

生成的 `scene.mvs` 大小约 9.01 MB，包含 707 张图像和 160,529 个稀疏点。

## 8. OpenMVS PatchMatch 深度估计

OpenMVS 版本为 2.4.0，使用 NVIDIA A800 80GB 的 CUDA 设备 2，最大线程数 8。主要参数为：

| 参数 | 值 | 含义 |
|---|---:|---|
| `resolution-level` | 1 | 输入图像缩小一半 |
| `max-resolution` | 960 | 工作图像长边上限 |
| `min-resolution` | 640 | 工作图像尺度下限 |
| `sub-resolution-levels` | 2 | 两级子分辨率处理 |
| `number-views` | 20 | 每张参考图使用 20 个邻居 |
| `iters` | 4 | 4 次光度 PatchMatch 迭代 |
| `geometric-iters` | 2 | 2 次几何一致性迭代 |
| `postprocess-dmaps` | 1 | 深度图后处理开启 |
| `remove-dmaps` | 0 | 保留 DMAP |

因此实际深度图分辨率为：

- 相机 00–04：960×640，共 505 张；
- 相机 09、10：960×540，共 202 张。

归档中 707 个 DMAP 都包含深度、法线、置信度和视图信息，每个 DMAP 保存 1 个参考图 ID 加 20 个邻居 ID。

### 稀疏点和深度搜索范围

OpenMVS 确实读取了 `scene.mvs` 中的 160,529 个稀疏点。但是每个参考图的邻居日志均显示 `(0 shared points)`。这说明：稀疏点云被载入场景，但外部邻居列表没有把共享稀疏点关联到每个 PatchMatch 深度问题中，不能据此认定每张图的 min/max 是由它与邻居间的共享稀疏点直接估计出来的。

运行日志中出现过类似 `[0.125899, 108.352]`、`[0.0251231, 102.172]` 的内部初始搜索范围；而归档 DMAP 头部最终全部写成 `[0.1, 100.0] m`。现有证据支持的结论是：内部初始化范围可以略宽于 100 m，但有效输出范围最终被固定或裁剪到 0.1–100 m。仅凭现存日志不能进一步确定这个固定范围来自补丁代码、运行配置还是编译时默认值。

## 9. 深度图后处理与融合

深度图配置包括：

```text
NCC Threshold Keep = 0.80
Descriptor Min Magnitude Threshold = 0.025
Normal Diff Threshold = 25°
Speckle Size = 7
Interpolate Gap Size = 7
Random Smooth Depth = 0.02
Random Smooth Normal = 13
Random Smooth Bonus = 0.98
```

融合参数为：

| 参数 | 值 | 作用 |
|---|---:|---|
| `fusion-mode` | 0 | 执行普通稠密融合 |
| `fusion-filter` | 2 | 使用较严格的融合过滤模式 |
| `number-views-fuse` | 2 | 一个融合点至少需要 2 个视图支持 |
| `fusion-depth-diff-threshold` | 0.010 | 相对深度差阈值约 1% |
| `fusion-reprojection-threshold` | 2.5 | 重投影误差阈值 2.5 像素 |
| `Normal Diff Threshold` | 25° | 法线方向一致性限制 |
| `filter-point-cloud` | 0 | 融合后没有再启用额外点云过滤 |
| `estimate-colors` | 2 | 为融合点估计颜色 |
| `estimate-normals` | 2 | 为融合点估计法线 |
| `estimate-roi` / `crop-to-roi` | 0 / 0 | 不估计和裁剪 ROI |

这意味着远处点需要至少两个视图在相对深度、重投影位置和法线方向上同时一致。远处像素的小视差、较弱纹理以及深度误差放大，会更容易在 PatchMatch 或融合阶段被过滤掉。

## 10. 最终输出和归档

最终 `scene_dense.ply` 的头部记录：

```text
element vertex 2546829
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property float nx
property float ny
property float nz
```

所以这个 PLY 是 OpenMVS 将 707 张深度图做几何一致性融合后得到的有色、带法线稠密点云，不是 DVP-MVS 自带的融合结果。

归档还包含：

- 707 个原始 DMAP；
- 707 个天空掩膜；
- 后续 roadmesh 结果；
- 后续 StreetGS 结果。

其中 roadmesh 和 StreetGS 是 OpenMVS 稠密点云之后的下游处理，不属于 OpenMVS 深度融合本身。

## 11. 日志完整性说明

现有工作目录中的 `densify.log` 只记录到 `Estimated depth-maps 160/707 (22.63%)`，当时工作目录也只留下了 160 个阶段性 DMAP；该日志没有保留随后成功完成的控制台输出。因此不能从这份日志准确还原最终融合耗时、每级过滤掉的点数或融合支持视图直方图。

但最终归档提供了独立的完成证据：

- `ARCHIVE_COMPLETE=PASS`；
- 707/707 个 DMAP 已归档；
- `scene_dense.ply` 存在且包含 2,546,829 点；
- 后续 roadmesh 报告读取的输入点数也正好是 2,546,829。

工作目录内残留的 `RUN_STATUS=RUNNING` 和批任务的 `RUNNING_RESUME` 属于未更新的状态文件，不能代表最终归档未完成。

第一次启动曾因 `COLMAP_GPUS must contain exactly seven GPU IDs` 失败；修正 GPU 配置并恢复运行后，才生成了最终完整归档。
