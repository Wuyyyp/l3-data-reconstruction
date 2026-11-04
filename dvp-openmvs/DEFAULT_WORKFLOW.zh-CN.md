# DVP-MVS 新交付数据固定工作流

确认日期：2026-10-19。

适用数据：上游已经完成位姿修复和 COLMAP BA，并按本文样例结构交付的 T3 clip。

**固定约定：默认使用方案 D。直接使用交付的 BA 后相机位姿和同一套 BA 后稀疏模型；DVP 阶段不再执行 BA。动态 mask 只选择本次重建范围内地图 XY 最大两点距离严格大于 1 m 的轨迹。天空、固定车体和边界 mask 继续保留。**

本文是这类新交付数据的操作规范。使用第 5 节的一键 D 入口；通用 prepared sample-root 脚本仍保留旧任务的兼容默认值。

## 1. 样例与数据来源

已检查的交付目录：

```text
/data/l3data-reconstruction-bingxing/samples-restored/corner-case-t3/road-construction/2026_09_11_16_02_31_dlp_pilotGtParser
```

下文以 `CLIP_ROOT` 表示单个交付 clip 根目录，文件相对路径均相对于它。

| 数据 | 固定来源与用途 |
|---|---|
| BA 后相机位姿 | `colmap/poses/image_camera_to_world_after_ba.csv`；所有 DVP、LiDAR 投影、3D 框投影和 OpenMVS 相机使用这一套位姿 |
| BA 后稀疏模型 | `colmap/sparse_ba/{cameras,images,points3D,rigs,frames}.bin`；复用内参、位姿、点坐标与观测关系 |
| BA 报告 | `colmap/colmap_reconstruction_report.json`；确认 BA 已开启、模型和位姿来源 |
| 帧、标定和目标标注 | `result_interpolation_optimized.pkl` 的 `infos`、`sensors.cams`、`gt_boxes`、`track_id` |
| LiDAR 点云 | `compensated_perception/*.pcd`；保持交付的坐标约定，不对已补偿目标重复补偿 |
| ego 到世界变换 | PKL 字段 `ego2global_transformation_matrix_camera_reference_offset_optimized`；用于交付 ego 坐标点云和标注框中心转到地图坐标 |
| 原始 RGB | BA CSV 的 `image_path`，或 PKL 的相机 `data_path` 配合已核验的图像根目录；按相机、时间戳、原始帧号匹配 |
| `bbox/` | 附加的检测、跟踪、SAM 和 3D 框结果；不自动替代原有 PKL track 分类规则，也不整体并入动态 mask |

### 必须区分的位姿版本

- `image_extrinsics/image_camera_to_world_after.csv` 是位姿修复后的 **BA 前输入**，不是最终 BA 位姿。
- PKL 中已有的 `camera_to_world_optimized` 不能代替 `image_camera_to_world_after_ba.csv`。
- `colmap/sparse_input`、`colmap/sparse_triangulated` 不是本流程采用的最终模型。
- 默认使用 `colmap/sparse_ba`。`sparse_filtered` 只有核验确实来自同一次 BA、位姿和内参相同后才可作为点过滤版本使用，不能仅凭目录名替换。
- CSV 的 `T_cw` 是 camera-to-world；DVP 的 world-to-camera 必须取其逆。COLMAP 的旋转和平移表示 world-to-camera，相机中心为 `-R.T @ t`。
- 缺少 BA 后位姿或模型时停止并报告；不自动退回 BA 前位姿，不自动重新 BA。

### 当前样例核验结果

- PKL：439 帧，原始帧号 `0–438`；compensated PCD：439 个，按 PKL 时间戳逐帧存在。
- BA CSV 和 `sparse_ba`：均包含 3,072 张图像、7 台相机；模型有 267,163 个稀疏点。
- 逐图像核对 BA CSV 与模型：无名称匹配失败；相机中心最大差约 `2.71e-10 m`，旋转矩阵元素最大差约 `1.59e-14`。
- 首帧 `center_camera_fov30` / `1789113742508102.jpg` 没有 BA 位姿。这是已知缺项：处理本样例时明确排除该图并写入审计，不补 BA 前位姿；其余相机帧保留。
- 完整图像路径在交付机使用 `/mnt/duanxuhao_bingxing/...`，本机对应挂载为 `/data/mnt/duanxuhao_bingixng/duanxuhao_bingxing/...`。逐项替换这个前缀后，3,072 张有 BA 位姿的 RGB 全部存在。`bingixng` 是当前实际目录名。
- 本样例 canonical 图像可在以下根目录找到：

```text
/data/mnt/duanxuhao_bingixng/duanxuhao_bingxing/dataconvert_t3/pose_dynamic_compensation/outputs/corner_case/road_construction/T3/2026_09_11_16_02_31_dlp_pilotGtParser/canonical/images
```

路径重映射只改变文件位置，不能改变时间戳或用相邻帧图像代替。

## 2. 分段和坐标对齐

沿用 101 帧窗口：`f0_100`、`f101_201`、`f202_302`、`f303_403`……，区间两端均包含。窗口内的动态轨迹独立分类，不用整段 clip 的位移决定某个静止窗口是否 mask。

若任务没有另行指定，全 clip 处理时保留不足 101 帧的尾段。本样例对应四个完整窗口和 `f404_438` 尾段；它不是上一批数据“固定五个完整窗口”的任务约定。

准备数据时可以将窗口帧号归零、重排图像名，但必须保存以下映射：

```text
原始 frame + camera + timestamp_ns + BA image name / ID
    <-> 窗口 frame_index + prepared label + DVP image ID
```

按这个映射从交付模型选择窗口图像、裁剪点轨迹并移除不再有观测的点；相机位姿、稀疏点坐标和内参不能被重新优化。重命名必须同步更新模型图像名和观测引用，不能把七相机子目录中的微秒时间戳文件名直接当成旧的 `000000_00.jpg`。

由选定模型生成 OpenMVS `scene.mvs` 和窗口邻居关系。不要直接使用交付的全 clip 图对作为每个窗口的 DVP 邻居；DVP 只引用该窗口实际存在的图像。

LiDAR 米制标定的投影链为：

```text
P_world  = optimized_ego_to_world @ P_ego
P_camera = inverse(BA_camera_to_world) @ P_world
pixel    = K @ P_camera
```

对 ego 坐标 GT 框采用同一坐标链。不要同时把 BA 相机增量再乘到 ego 位姿上；不要使用原始 `ego2global_transformation_matrix` 替换交付优化参考位姿。

## 3. 动态 mask：严格大于 1 m

继续采用当前 DVP 的 PKL 标注轨迹规则，默认 track 来源是 PKL 的 `track_id` 和 `gt_boxes`。先将同一 track 在窗口内各帧的框中心由 ego 坐标转到地图坐标，再计算：

```text
d(track) = max(norm(center_world_xy[i] - center_world_xy[j]))
selected_dynamic = d(track) > 1.0  # 米，严格大于
```

- 使用最大任意两点距离；不用累计路程、速度阈值或仅首尾距离。
- 恰好 1 m 不选择。缺少有效 ID 或不足两个有效位置的轨迹不凭空判为动态，并单独记录为信息不足。
- 一条轨迹在窗口内满足条件后，对其在该窗口内所有出现帧生成 mask，不只遮掉发生大位移的帧。
- 选中的 3D 框使用对应图像的 BA 后位姿和匹配内参投影；复用近裁剪、投影覆盖和边缘扩张逻辑。
- 保留天空、窗口内同相机重复出现的固定车体/后视镜语义区域、camera 10 固定车体下沿、与图像边界连通的黑边及边缘余量。静止车辆、施工设施等不因类别名称而被整体遮掉。
- 相同的有效区域用于 MoGe/LiDAR 标定、DVP 深度估计、DMAP 转换和融合。源数据和上游 BA 模型保持原样。

**不能直接使用 `colmap/matching/pose_valid_masks` 或 `dense_valid_masks` 作为本流程的最终动态 mask。** 本样例报告显示这些 mask 按 car/person/truck 等类别排除像素，部分还包含 vegetation，不符合“位移大于 1 m 才选择动态目标”的规则。应重新生成符合本节规则的组合 mask。

`bbox/optimized_3d` 中带 `world` 字样的文件也不能直接把 `cx/cy/cz` 当世界坐标；本样例该表的 `center_frame` 为 `camera`。如果以后明确改用 bbox 多视角轨迹，需要单独约定 `global_track_id`、`source_tracks`、明确的 `world_x/world_y` 字段及其位姿版本，并关联到对应 SAM mask；不能将局部 track ID 与 PKL track ID 混用，也不能按现成 `motion_state` 直接替代本节 1 m 判断。

审计至少记录窗口原始帧范围、阈值、所用 ego 字段、逐 track 最大位移和选择结果。现有批处理要求：

```text
dynamic_policy = map-xy-window-displacement-gt1m-v1
dynamic_threshold_m = 1.0
```

现有分类函数 `select_tracks(..., threshold_m=1.0)` 支持该阈值；旧函数默认值仍是 3 m，不能省略参数。已用本样例 PKL 验证窗口分类：`f0_100` 选中 5/11 条、`f101_201` 为 14/26、`f202_302` 为 19/30、`f303_403` 为 7/12、`f404_438` 为 3/10。这是轨迹规则检查，不代表已生成或运行重建。

## 4. 稠密重建主流程

```text
交付 PKL + RGB + compensated PCD
             + BA 后 CSV / 稀疏模型
    -> 按窗口选择图像，建立 ID / 时间戳映射
    -> 复用 BA 后模型，构建 scene.mvs 与邻居
    -> 生成 >1 m 动态 mask，合并原有天空 / 车体 / 边界 mask
    -> 从 BA 后稀疏点计算逐视角深度范围
    -> MoGe3 深度 + LiDAR 米制尺度标定
    -> 方案 D：BA 稀疏点尺度校准 + 跨视图 MoGe 支持检查
    -> 方案 D：10 源视图、legacy19 的 DVP 深度和法线
    -> 保留原始深度，再做跨视图先验过滤
    -> 转成 OpenMVS DMAP
    -> OpenMVS 深度过滤 / dense-fuse
    -> 最终 PLY
```

后续此类交付数据默认采用完整方案 D，固定参数如下：

| 项目 | 固定约定 |
|---|---|
| 内部 COLMAP BA | 关闭，显式 `--no-colmap-ba`；也不重新特征匹配和三角化 |
| 方法 | D；不能仅设置 `legacy19` 就声称完成方案 D |
| DVP 迭代配置 | `legacy19`，两尺度 7 次 pass，内部更新数 `[3,3,3,3,3,2,2]`，合计 19 次 |
| DVP 源视图 | 保留窗口内原始 20 邻居候选及排序，DVP 使用其中前 10 张；不是重新按最近帧任取 10 张 |
| 方案 D 先验校准 | MoGe/LiDAR 初始标定后，使用同一套交付 BA 稀疏点做稳健尺度校准，并检查跨视图 MoGe 支持 |
| 方案 D 后过滤 | 保留后处理前的原始 DVP 深度；对可靠先验不一致像素检查原始 20 邻居的几何支持，再执行先验过滤 |
| 执行方式 | `DVP_CPU_PIPELINE=1`；单窗口单 GPU，不同窗口分配不同 GPU 并行；关闭额外 fast/cache/exec/input-overlap 实验开关 |
| 计算尺寸 | `960×640` 画布；保持宽高比，16:9 内容为 `960×540`，上下各补 50 像素 |
| 图像与内参 | 使用已经去畸变的 RGB，不再二次去畸变；按实际 RGB 缩放及 padding 同步更新 K |
| 深度范围 | BA 后模型的可见正相机 Z，`min×0.9`、`max×1.1`；窗口子模型无直接观测时投影完整交付 BA 模型的同视锥稀疏点，再用同相机最近有效视角回退；不使用固定 100 m 上限 |
| MoGe3 米制尺度 | 有效 LiDAR 深度大于 2 m，按中位数比值标定；不设 80 m 上限，复用现有稳健裁剪和缺少投影点时的相机/整段回退 |
| DVP 原生融合 | 关闭，`DVP_SKIP_FUSION=1`，只输出逐图深度和法线 |
| OpenMVS | 使用原始 20 邻居融合 DVP 过滤后深度；不追加 OpenMVS 几何 PatchMatch，至少 2 视图支持、深度差 1.5%、重投影 2.5 px、法线差 25°；speckle 和插值间隙参数均为 7 |

本样例五路 RGB 为 `1920×1280`、两路前视为 `3840×2160`。BA 模型内参对应这些原始尺寸。五路到 DVP 的缩放比例为 0.5，两路前视直接到 DVP 有效内容的比例为 0.25；如果先准备成 `1920×1080`，必须先把前视 K 同步缩小一半。不能把 4K 内参直接配给 1920 或 960 宽图像。

方案 D 沿用上一批 T3 实验的算法步骤和参数，但 BA 的来源改为本次交付结果，不再由本地 DVP 流程重做。普通 20 源视图 / legacy 24 次更新流程只作为显式对照方案，不是新交付数据的默认。

方案 D 的校准与过滤沿用已有实现：稀疏尺度拟合使用重投影误差不大于 2 px、track 长度至少 3 的点，保留 20% point ID 做尺度校准验证；同相机 MoGe 跨视图支持使用帧偏移 `±3/±6`。交付模型中某相机没有足够的合格稀疏对应时，该相机保留 LiDAR 标定尺度并在报告中标明回退，其余相机继续稀疏校准。可靠支持至少为 2 的先验像素，若深度偏差超过校准阈值，再检查原始 20 邻居中 1% 深度差范围内的几何支持；不足 2 个支持或落在搜索远端 98% 附近的相应不一致像素被删除。尾段与窗口边界只使用实际存在的相邻图像。

## 5. 新交付数据一键入口

新交付目录使用 `mvs_process/run_delivery_d.py`。默认把整段 clip 分成 101 帧窗口，保留尾段；`--gpu-ids` 给不同窗口分配单独 GPU，每个 GPU 同时只运行一个窗口。`--check` 只校验交付 BA CSV、模型、RGB、位姿映射和运行依赖，不做稠密重建。

```bash
python /data/l3data-reconstruction-bingxing/DVP-MVS/mvs_process/run_delivery_d.py \
  --clip-root /data/l3data-reconstruction-bingxing/samples-restored/corner-case-t3/road-construction/2026_09_11_16_02_31_dlp_pilotGtParser \
  --output-root /data/l3data-reconstruction-bingxing/tem-test/delivery-d \
  --gpu-ids 0,1,2,3,4,5,6,7 --check
# 校验通过后移除 --check 启动完整方案 D；--prepare-only 只生成窗口输入。
```

该入口直接读取 `colmap/poses/image_camera_to_world_after_ba.csv` 和 `colmap/sparse_ba`，按窗口删除窗口外图像及观测并重命名窗口图像，验证 CSV/模型位姿一致后生成 scene.mvs。它按窗口内地图 XY 位移严格大于 1 m 选择动态轨迹，将投影的动态框与语义像素、天空、车体和黑边组合；使用已补偿 PCD、BA 相机位姿投影 LiDAR。之后运行通用入口的 MoGe/LiDAR 阶段，再运行 D 稀疏点尺度校准与跨视图 MoGe 支持检查。DVP 使用前 10 邻居、legacy19 和 CPU pipeline；先保留原始深度再用原 20 邻居做先验过滤，最终以原 20 邻居、1.5% 深度差进入 OpenMVS 融合。内部 BA、特征匹配和三角化均不会重做。

每个窗口输出 `<output-root>/<clip-name>/f<start>_<end>/10_openmvs_fusion/openmvs_filtered_fused.ply`，并保留交付映射、排除图像、轨迹、mask、校准、原始与过滤后深度及执行日志。缺少 BA 图像时显式记录并跳过，不回退到 BA 前位姿。已有完整 PLY 会被复用；若交付 BA 文件指纹变化，入口会拒绝复用并要求新结果目录。

旧 `mvs_process/run_batch_mvsnet_to_openmvs_ply.sh` 仍接收 prepared sample-root，其 legacy/20 邻居/1% 融合默认值只是兼容旧任务，不能直接传入新交付 clip 根目录，也不能把单独的 `--dvp-profile legacy19` 当作完整 D。

## 6. 输出与质量检查

结果按 `<result-root>/<clip-name>/f<start>_<end>/` 组织，目录名保持简洁。每个窗口至少保留：

- 输入来源、BA CSV/模型指纹、图像 ID/时间戳映射和已知缺项记录。
- 动态轨迹选择报告、mask 审计与抽样预览。
- MoGe 标定先验，以及后处理前的 DVP 原始 `depths.dmb` 和法线；保留原始浮点深度，另存原始深度的 PNG 伪彩预览。
- OpenMVS DMAP、执行参数和日志。
- 最终 `10_openmvs_fusion/openmvs_filtered_fused.ply`，记录 PLY 点数。

原始深度 PNG 沿用 `/data/l3data-reconstruction-bingxing/colmap+openmvs/pipeline_tools/dmap_utils.py` 的 `depth_color` 逻辑：相机 Z 深度按 `1 - clip(depth, 0, 100) / 100` 映射，使用 OpenCV TURBO，无效像素为黑色。100 m 是统一显示色标上限，不截掉浮点原始深度或最终点云的远处数据。PNG 按一级降采样后的有效 RGB 尺寸输出，五路 `960×640`、两路前视去掉 padding 后 `960×540`；在元数据中注明 PNG 是伪彩预览，不是可直接读取米值的数值深度。

入口不移动或覆盖已有结果，原始交付数据只读。

流程完成和重建质量分开判断。深度文件数量齐全、程序正常退出或 PLY 非空，只能说明流程完成。检查窗口运动基线、有效 mask 比例、深度分布和跨视图一致性；点数异常稀少要标记需复查。特别是近乎静止窗口，BA 不能创造视差，不可将仅有几百/几千点的结果直接作为质量合格的稠密重建。

## 7. 可选远景开关（默认关闭）

在第 5 节的一键 D 命令上追加 `--far-only` 即可开启；不传该参数仍执行原来的完整重建，继续调用原有 `build-cpu-pipeline/APD`。不通过环境变量自动开启，也不改变 BA、动态 mask 或 DVP/OpenMVS 的原有参数。

```bash
/data/l3data-reconstruction-bingxing/DVP-MVS/runtime/venv/bin/python \
  /data/l3data-reconstruction-bingxing/DVP-MVS/mvs_process/run_delivery_d.py \
  --clip-root "$CLIP_ROOT" \
  --output-root /data/l3data-reconstruction-bingxing/tem-test/delivery-d \
  --gpu-ids 0 --far-only
```

每个窗口独立定义远景：点到该窗口全部自车位置的最小地图 XY 距离严格大于 50 m。恰好 50 m 删除；Z 高度不参与距离计算。使用 PKL 的 `ego2global_transformation_matrix_camera_reference_offset_optimized`，包含窗口内没有注册 BA 图像的帧。不会用整个长 clip 的其它窗口轨迹裁剪当前窗口；例如 `f101_201` 使用原始帧 101～201 的自车位置。

处理分两步：

1. D 先验校准后，利用跨视图支持至少 2 的米制先验判断可靠近处区域。距轨迹加先验误差余量小于 40 m 的区域才跳过，并留出 16 像素边界支持。未知、无效或不确定深度保留；与已有有效区域 mask 取交集，源 RGB 保持原状。CUDA 跳过这些参考像素的主要匹配计算。
2. OpenMVS 融合后，对最终 PLY 按全部窗口自车位置严格执行 `min XY distance > 50 m`，保留颜色、法线等顶点属性。最终远景可以为空，这只表示没有符合距离条件的输出点。

远景模式使用独立的 `<output-root>/<clip>/f<start>_<end>_far50/`，普通模式仍用 `f<start>_<end>/`。开关在多 GPU 窗口分配、`--check`、`--prepare-only` 和续跑时均会传递。不同模式/规则的中间结果拒绝混用；不能通过复制旧完整结果或重命名目录冒充远景结果。输入 BA/PKL 指纹变化时使用新结果目录。

主要输出：

- `10_openmvs_fusion/openmvs_filtered_fused.ply`：严格 50 m 外的最终点云。
- `10_openmvs_fusion/openmvs_filtered_fused_uncropped.ply`：裁剪前的融合诊断结果，仍可能包含计算边界保留的近处点；同目录 `.mvs` 也属于裁剪前结果。
- `far_only/masks/`、`trajectory.npy`、`mask_report.json`、`crop_report.json`：计算 mask、窗口轨迹和审计。
- `audit/run_config.json`、`D.complete`：记录 `far_only` 与规则版本，支持相同配置续跑。

远景程序从同一源码独立编译到 `build-far-only/APD`，不会替换默认二进制。当前部署依赖下的编译命令：

```bash
cd /data/l3data-reconstruction-bingxing/DVP-MVS
cmake -S . -B build-far-only -DCMAKE_BUILD_TYPE=Release \
  -DOpenCV_DIR="$PWD/build-cpu-pipeline/local-opencv-config" \
  -DBOOST_ROOT="$PWD/runtime/native/usr"
cmake --build build-far-only --target APD -j4
runtime/venv/bin/python scripts/test_far_only.py
```

计算用 mask 是带余量的近处估计，不能保证几何结果不变。2026-10-20 的单 clip 试验计入新增 mask 开销后，相比完整计算再裁剪节省约 23% 的公共准备后耗时；这是共享 GPU 的单次结果，不能保证每个场景相同收益或无损。最终距离裁剪的范围是确定的，远处点本身的深度精度仍受 DVP 重建质量影响。
