# DongFeng 多传感器位姿优化代码交付说明

本仓库交付的是可独立运行的项目代码。准备原始数据和模型权重后，可以在本地完成：数据标准化、曝光 offset ego 补偿、动态目标点云补偿、语义分割、特征匹配与筛选、多相机联合优化、最终 pkl/点云/QC 输出。

仓库不交付历史实验数据，也不依赖开发环境中的 `data_cache/` 或 `outputs/`。

专项实现说明：

- [动态目标点云补偿](docs/dynamic_object_compensation.md)
- [位姿修正](docs/pose_correction.md)
- [前向 LiDAR 语义动态过滤融合](docs/t3_front_lidar_fusion.md)
- [COLMAP 静态特征 BA 与稠密重建](docs/colmap_reconstruction.md)
- [运行加速与 PPU 兼容性](docs/runtime_acceleration_and_ppu.md)

## 1. 交付边界

代码交付包含：

```text
src/
scripts/
configs/
docs/
tests/
pyproject.toml
environment.yml
.gitignore
README.md
```

以下目录不随代码交付，也不应提交到 Git：

```text
data/
data_cache/
outputs/
deliverables/
.models/
```

这些目录由运行流程在本地生成。

## 2. 环境安装

当前交付镜像已经提供与 PPU 对应的 Python、PyTorch 和 CUDA 运行时。不要直接使用
`environment.yml` 创建 Conda 环境，否则会新建 Python 3.11 环境，并可能替换镜像中
已验证的 PyTorch/PPU 组合。

在 Linux PPU 镜像中，使用继承系统包的 venv：

```bash
cd /mnt/workspace/Pose_Dynamic_Compensation/code
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e .
python -m pip install "matplotlib>=3.8" "typer>=0.12"
python -m pip install python-lzf
```

`python-lzf` 为动态点云补偿提供原生 PCD LZF 解压。缺失时流程会自动回退到兼容的
纯 Python 解码器，结果不变，但 `binary_compressed` PCD 的预扫描会明显变慢。

主线匹配后端使用官方 LoMa，安装在本地 `.tools/loma`：

```bash
git clone https://github.com/davnords/LoMa.git .tools/loma
python -m pip install --no-deps -e .tools/loma
```

LoMa-B、检测器和描述器权重统一放在共享模型目录
`/mnt/workspace/Perception_Models/torch`。运行前设置：

```bash
export TORCH_HOME=/mnt/workspace/Perception_Models/torch
```

权重文件位于 `$TORCH_HOME/hub/checkpoints/`：

```text
loma_B.pt
dinov2_vitl14_pretrain.pth
dad.pth
```

当前共享目录中的完整文件路径为：

```text
/mnt/workspace/Perception_Models/torch/hub/checkpoints/loma_B.pt
/mnt/workspace/Perception_Models/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth
/mnt/workspace/Perception_Models/torch/hub/checkpoints/dad.pth
```

这三个文件已准备完整，后续 LoMa 的 `match-delivery` 阶段可以离线运行，不应再使用项目内的 `.models/torch` 目录。

随后可验证本地权重可被加载：

```bash
python -c "from loma import LoMa, LoMaB; LoMa(LoMaB()); print('LoMa weights ready')"
```

如需补齐缺失权重，验证命令会从 LoMa 官方地址下载；已有权重可直接离线复用。

语义分割使用共享目录中的 Mask2Former Hugging Face 缓存。运行前设置：

```bash
export HF_HOME=/mnt/workspace/Perception_Models/pose/huggingface
```

此时 `facebook/mask2former-swin-large-mapillary-vistas-semantic` 会从
`$HF_HOME/hub/models--facebook--mask2former-swin-large-mapillary-vistas-semantic`
离线加载。

对应的完整语义分割模型目录为：

```text
/mnt/workspace/Perception_Models/pose/huggingface/hub/models--facebook--mask2former-swin-large-mapillary-vistas-semantic
```

不要升级或重装 `torch`、`torchvision`、`opencv-python`。它们由镜像提供，直接使用
通用 PyPI 版本可能失去 PPU 兼容性。LoMa 的最新依赖元数据可能要求更新的
`torchvision`；在 PPU 镜像中保持镜像自带版本，并以实际导入和匹配测试为准。

可选的 `open3d` 仅用于非 translation-only ICP。当前 PPU/Python 3.12/CUDA 12.8/
PyTorch 2.6 组合没有兼容的 Open3D wheel，默认的 translation-only ICP 使用 SciPy，
不受影响。

验证环境和 PPU：

```bash
export MPLCONFIGDIR=/mnt/workspace/Pose_Dynamic_Compensation/code/.cache/matplotlib
export HF_HOME=/mnt/workspace/Perception_Models/pose/huggingface
ppu-smi -L
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('devices=', [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
python -c "from loma import LoMa, LoMaB; print('LoMa ready')"
python -c "from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation; model_id='facebook/mask2former-swin-large-mapillary-vistas-semantic'; AutoImageProcessor.from_pretrained(model_id, local_files_only=True); Mask2FormerForUniversalSegmentation.from_pretrained(model_id, local_files_only=True); print('Mask2Former ready')"
python -m dongfeng_registration.cli --help
```

在已分配 PPU 的运行终端中，`ppu-smi -L` 应列出至少一张 PPU，且 PyTorch 命令应输出
`cuda_available= True`、大于 0 的 `device_count=`，并列出设备名称；满足这些条件后才可
执行语义分割、LoMa 匹配和联合优化等 GPU 阶段。若出现 `init HGML error: driver is not loaded`、
`torch.cuda.is_available()` 为 `False` 或设备数量为 `0`，说明当前终端/容器未获得 PPU
设备或驱动访问权限；请切换到已分配 PPU 的实例终端后再运行，不要通过重装 PyTorch、CUDA
或重新下载模型权重解决。

正式交付流程通过一个持久的 `match-delivery` 进程依次执行全部 intra/inter 关系。
LoMa 模型只加载一次，特征缓存跨相机和跨关系复用。每路相机的 CPU 特征缓存默认不设上限；
GPU 特征缓存由 `matching.loma_feature_cache_gpu_max_items` 控制，显存压力下按 LRU 淘汰，
被淘汰的特征仍保留在 CPU 缓存中。`matching.loma_process_count` 仍保留用于单独调试命令，
正式配置默认均为 `1`。

```yaml
matching:
  backend: loma
  loma_process_count: 1
```

图像缓存分辨率也由车型配置统一控制，不在运行命令中临时指定：

```yaml
semantic_segmentation:
  processor_image_size: 1024
  persist_dynamic_mask_cache: true
image_cache:
  enabled: true
  mode: preload
  max_long_side: 1600
matching:
  loma_image_size: 1600
```

加载配置时会检查 `image_cache.max_long_side` 是否不小于语义分割的
`processor_image_size` 和 LoMa 的 `loma_image_size`。当前示例配置满足
`1600 >= max(1024, 1600)`；如果把缓存分辨率调低，配置会直接拒绝加载。
缓存图像用于推理时可以是缩放后的图像，但匹配点会在输出前恢复到原始图像坐标，
因此后续相机内参、语义掩码和几何筛选仍使用同一坐标系。

语义 cache 以 `class_masks/<timestamp>.png` 和同名 metadata 作为唯一类别真值。
`masks/<timestamp>.png` 是按当前 `semantic_mask_classes` 和 `dilation_pixels`
生成的二值加速缓存，不是独立推理结果；文件缺失或损坏时，匹配流程会从类别图自动重建，
不会重新运行 Mask2Former。`persist_dynamic_mask_cache: true` 是正式流程默认值，可减少
intra/inter 匹配重复派生 mask 的 CPU 时间。COLMAP 和 LiDAR 动态过滤直接读取类别图，
可以分别使用自己的动态类别集合。

主入口：

```text
dongfeng-registration
python -m dongfeng_registration.cli
```

## 3. 标准输入：canonical dataset

正式流程统一从 canonical dataset 开始。原始车型目录只由 `normalize-dataset` 读取，后续动态补偿、语义分割、匹配筛选、联合优化和交付导出都读取 canonical 目录。

标准结构：

```text
<vehicle>_canonical/
  result_interpolation_standard.pkl
  images/<camera>/*.jpg
  lidar/perception/*.pcd
  lidar/raw_lidars/<lidar>/*.pcd
  metadata/normalization_report.json
```

`result_interpolation_standard.pkl` 会保留原始 `ego2global_transformation_matrix`，并把图像、点云、相机内外参和 3D Box 统一成项目约定格式。

示例：

```powershell
python -m dongfeng_registration.cli normalize-dataset `
  --pose-backend result_interpolation_pkl `
  --dataset-root D:\project_data\M18_raw `
  --output-root D:\project_data\M18_canonical `
  --vehicle-model M18 `
  --pkl result_interpolation.pkl
```

T3 这类旧格式数据可使用 `--pose-backend ved_localization`，并按实际数据补充 `--navigation-csv` 和相机映射参数。

## 4. 完整运行主线：一条命令

默认在位姿优化和交付导出后执行 COLMAP 位姿先验 BA：`colmap.enabled: true`、
`run_bundle_adjustment: true`、`run_dense: false`。稠密重建仅在明确设置
`run_dense: true` 时执行；专用 OpenMVS 测试配置保持其显式稠密模式。

推荐优先使用 `run-delivery`。完整流程参数都写在 delivery 配置文件里，运行时只需要指定一个配置：

```powershell
python -m dongfeng_registration.cli run-delivery `
  --config configs\m18.yaml
```

正式运行前可以先 dry-run，检查即将执行的阶段和命令：

```powershell
python -m dongfeng_registration.cli run-delivery `
  --config configs\m18.yaml `
  --dry-run
```

断点续跑或只跑部分阶段：

```powershell
python -m dongfeng_registration.cli run-delivery `
  --config configs\m18.yaml `
  --from-stage persistent-match `
  --to-stage optimize
```

当前固定阶段顺序为：

```text
normalize
dynamic-compensation
export-dynamic-pkl
persistent-match
optimize
projection-qc
export-deliverables
colmap-reconstruction
```

其中：

- `normalize`：原始数据转 canonical；
- `dynamic-compensation`：动态目标点云补偿，同时使用曝光 offset 计算图像参考时刻 ego；
- `export-dynamic-pkl`：导出包含补偿后点云路径、补偿后 3D Box 和 offset ego 字段的新 pkl；
- `persistent-match`：由 `match-delivery` 在一个进程中依次执行预评估、全量语义分割、同相机和跨相机 LoMA 匹配及 retained 几何筛选；预评估抽样 pair 和 LoMA GPU 特征缓存会被正式匹配复用；
- `optimize`：多相机联合优化；
- `projection-qc`：用最终 pkl 做点云投影 QC；
- `export-deliverables`：整理可核验的交付目录。
- `colmap-reconstruction`：以交付目录中的逐图像外参重建静态特征数据库，执行可选的完整 SE(3) 位姿先验 BA，并可生成稠密点云。构建本地 COLMAP 扩展和先验参数见 `docs/colmap_reconstruction.md`。

`configs/l29.yaml`、`configs/m18.yaml` 和 `configs/t3.yaml` 均默认启用 FOV120：
L29/M18 使用 `front_camera_fov120`，T3 使用 `center_camera_fov120`。T3 的
`center_camera_fov120__center_camera_fov30` 安装基线过小，不运行退化的 `same_index`，
仅使用 `temporal_overlap_12m`；其他受支持的 FOV120 匹配关系仍包含在正式配置中。

T3 提供两个匹配规模配置，正式运行时只需切换 `--config`，不需要手工复制、删除或裁剪
match cache：

```powershell
# 完整版 36 组：7 路相机的 2m/4m/6m intra（21 组）+
# 6 组 same-index inter + 9 组 temporal-overlap inter（15 组）
python -m dongfeng_registration.cli run-delivery `
  --config configs\t3_windows_full37.yaml

# 21 组：7 路相机的 2m/6m intra（14 组）+
# 4 组 same-index inter + 3 组 temporal-overlap inter（7 组）
python -m dongfeng_registration.cli run-delivery `
  --config configs\t3_windows_subset21.yaml
```

两个配置都包含完整的基础模型、筛选和优化参数，并通过
`delivery.inter_relations_by_pair` 指定每个 inter pair 使用的 relation。运行前建议先加
`--dry-run` 检查将执行的匹配组和优化输入。

## 5. delivery 配置文件

示例配置见：

```text
configs/m18.yaml
```

通常只需要改这些字段：

```yaml
delivery:
  raw_dataset_root: D:/project_data/M18_raw
  canonical_root: D:/project_data/M18_canonical
  base_config: configs/m18.yaml
  run_output_root: outputs/M18_run
  semantic_cache_root: data_cache/M18/semantic_masks
  match_cache_root: data_cache/M18/matches
  reference_time_offset_s: 0.007
```

`run_output_root` is the only non-cache output root for one dataset run:

```text
<run_output_root>/
  dynamic_compensation/  # corrected point clouds, boxes, dynamic pkl, summary
  joint_optimization/    # optimization metrics, poses, and intermediate exports
  quality_control/       # dynamic-point-cloud and reprojection visualizations
  deliverables/          # final pkl, compensated point clouds, external poses, reports
  colmap/                # optional masked COLMAP BA and dense reconstruction
```

Semantic masks and LoMA raw/retained matches are reusable caches and remain under
`data_cache/<vehicle>/semantic_masks/` and `data_cache/<vehicle>/matches/`.

LoMa 正式配置使用 GPU 特征缓存。`matching.loma_feature_cache_mode: model` 将 detector/descriptor
特征直接保存在 GPU，并由 `loma_feature_cache_gpu_max_items` 控制 LRU 容量；当前正式配置为
5000。若需要低显存诊断，可切换为 `two_tier` 或 `cpu`，但预评估和正式匹配必须在同一个
`match-delivery` 进程中运行，才能共享这份 GPU cache。

预筛选使用独立的 pair 结果 cache，并与正式匹配共享同一进程内的 LoMa GPU 特征 cache。默认每个 intra baseline 或 inter relation 抽样 20 对，
总上限 800 对，结果写入 `<run_output_root>/preflight_match_cache/`，报告写入
`<run_output_root>/preflight_quality/`。预筛选始终使用语义掩膜；已有有效语义 cache 会被
复用。预筛选产生的 LoMA 点对和 detector/descriptor GPU 特征也会被后续正式匹配复用，
从而不会重复计算同一图像对或同一图像特征。

单独运行预筛选：

```powershell
python -m dongfeng_registration.cli diagnose-clip-quality `
  --config configs\m18.yaml
```

质量状态为 `FAIL` 时，`match-delivery` 默认在全量语义分割和正式匹配前停止；报告中的 `WARN` 允许继续运行。
P95 只作为诊断指标，不参与状态判定。单个异常关系不会直接导致 `FAIL`：先要求某个关系组中
异常 pair 数量至少达到 `min_bad_pairs_for_failure` 且比例达到 `max_bad_pair_fraction`，再要求
失败关系组占全部有效关系组的比例达到 `min_failed_relation_fraction_for_failure`。
`preflight_diagnostic` 位于车型基础配置中，可调整抽样数量、总上限、判定阈值和异常容忍度。

默认判定参数如下：

```yaml
preflight_diagnostic:
  enabled: true
  sample_pairs_per_relation: 20
  max_total_pairs: 800
  min_raw_matches_per_pair: 20
  min_ransac_inlier_ratio: 0.20
  min_valid_depth_ratio: 0.30
  reprojection_median_warn_px: 5.0
  reprojection_median_fail_px: 20.0
  max_bad_pair_fraction: 0.25
  min_bad_pairs_for_failure: 3
  min_failed_relation_fraction_for_failure: 0.20
  fail_action: stop_before_dynamic_compensation
```

状态判定分两层：同一关系中至少 3 个有效 pair 的重投影 median 超过 20 px，且异常比例达到
25%，该关系才记为失败；失败关系达到全部有效关系的 20%，clip 才因关系级重投影问题记为
`FAIL`。低于任一比例门槛时记为 `WARN`。相机图断连、有效 pair 不足、整体 RANSAC 内点率过低
或整体 retained 重投影 median 严重超限仍可独立触发 `FAIL`。

`match-delivery` 默认先执行该诊断。若诊断返回 `FAIL`，命令以非零状态结束，后续全量语义
分割和正式匹配不会启动。动态补偿仍由前一阶段负责。仅在排查时可将 `fail_action` 设置为
`report_only`。显式运行 `diagnose-clip-quality` 仍可单独生成报告，但不会与后续正式匹配共享
进程内 GPU cache。

默认不配置 `frame_count` 时，`run-delivery` 会按全量数据运行。做 smoke test 时可以临时加入：

```yaml
delivery:
  start_frame: 0
  frame_count: 10
```

`reference_time_offset_s` 是图像曝光参考时刻 offset。确定方法建议包括：

1. 用不同 offset 插值得到图像 ego，把静态点云投影到图像上，观察哪个 offset 对齐更好；
2. 对动态目标点云补偿后，只投影框内点云，观察动态物体边界和图像目标的对齐效果。

## 6. 关键字段约定

```text
ego2global_transformation_matrix
```

原始 ego 位姿，保留用于回溯。

```text
ego2global_transformation_matrix_camera_reference_offset
```

根据图像参考时刻插值得到的 ego 位姿，用于动态补偿后的视觉筛选和优化输入。

```text
ego2global_transformation_matrix_camera_reference_offset_optimized
```

多相机联合优化后的图像参考时刻 ego 位姿。做最终点云投影到图像时，应使用这个 optimized ego 字段。

## 7. 输出和交付物

最终交付物位于 `<run_output_root>/deliverables/`，通常包含：

```text
<run_output_root>/deliverables/
  result_interpolation_optimized.pkl
  compensated_perception/*.pcd
  image_extrinsics/
  metrics/
  projection_qc/
  manifest.json
  README_delivery.md
```

其中：

- `result_interpolation_optimized.pkl`：最终 pkl；
- `compensated_perception/*.pcd`：补偿后的融合点云；
- `image_extrinsics/`：逐图像相机外参；
- `metrics/`：优化前后指标；
- `projection_qc/`：点云投影可视化，用于结果核验。

### PKL 中的优化后相机世界位姿

PKL 中每帧相机的 `cams[相机].extrinsic` 是 `T_ego_camera`，即车体坐标系到相机坐标系的变换，不是世界坐标系位姿。导出时该字段已经包含优化后的固定安装外参与该相机该帧的时变扰动。

优化后的车体世界位姿保存在每帧的
`ego2global_transformation_matrix_camera_reference_offset_optimized`。当该车体位姿已经对齐到该相机图像时间戳时，可按以下方式计算相机世界位姿：

```python
import numpy as np

T_world_ego = np.asarray(
    info["ego2global_transformation_matrix_camera_reference_offset_optimized"],
    dtype=np.float64,
)
T_ego_camera = np.asarray(
    info["sensors"]["cams"][camera_name]["extrinsic"],
    dtype=np.float64,
)
T_world_camera = T_world_ego @ np.linalg.inv(T_ego_camera)
camera_position_world = T_world_camera[:3, 3]
```

当车体参考时刻与相机图像时间不同，上述公式不能替代时间插值；原始车体位姿仍保存在
`ego2global_transformation_matrix`，用于与优化结果对比。

为避免相机时间戳与车体参考时间戳不同导致的误差，最终 PKL 还直接保存逐相机、逐图像帧的最终世界位姿：
`info["sensors"]["cams"][camera_name]["camera_to_world_optimized"]`。该 4x4 矩阵已经使用相机图像时刻的插值车体位姿，是联合优化和重投影 QC 使用的最终 `T_camera_to_world`，下游应优先使用它；它与 `image_extrinsics/after/*_image_camera_to_world_after.csv` 中同相机、同帧的 `T_cw_*` 一致。

## 8. 分阶段命令：高级调试/排错

通常不需要手动执行这些命令；它们主要用于定位问题或断点续跑。

语义分割：

```powershell
python -m dongfeng_registration.cli segment-dataset `
  --config configs\m18.yaml `
  --camera front_camera_fov30 `
  --cache-root data_cache\M18\semantic_masks `
  --backend mask2former `
  --batch-size 32 `
  --device cuda:0
```

同相机匹配：

```powershell
python -m dongfeng_registration.cli match-dataset `
  --config configs\m18.yaml `
  --camera front_camera_fov30 `
  --baseline-m 2 --baseline-m 4 --baseline-m 6 `
  --cache-root data_cache\M18\semantic_masks `
  --match-cache-root data_cache\M18\matches
```

跨相机匹配：

```powershell
python -m dongfeng_registration.cli match-multicamera `
  --config configs\m18.yaml `
  --cameras front_camera_fov30 --cameras left_front_camera `
  --semantic-cache-root data_cache\M18\semantic_masks `
  --match-cache-root data_cache\M18\matches `
  --relation same_index `
  --relation temporal_overlap
```

多相机联合优化：

```powershell
python -m dongfeng_registration.cli optimize-multicamera-extrinsics `
  --config configs\m18.yaml `
  --match-cache-root data_cache\M18\matches\loma__mask2former `
  --output outputs\M18_joint_mainline
```

动态点云 QC（仅 3D 框内点，补偿前后对比）：

```powershell
python scripts\visualize_dynamic_compensated_pkl_projection.py `
  --pkl outputs\M18_joint_mainline\deliverables\result_interpolation_optimized.pkl `
  --image-root D:\project_data\M18_canonical `
  --output-dir outputs\M18_projection_qc `
  --offset-ego-field ego2global_transformation_matrix_camera_reference_offset_optimized `
  --frame-stride 5 `
  --point-radius 2
```

匹配重投影 QC（优化前后对比）：

```powershell
python scripts\generate_reprojection_qc.py `
  --optimization-output outputs\M18_joint_mainline `
  --dataset-root data\M18_canonical `
  --output-dir outputs\M18_projection_qc\reprojection
```

交付目录整理：

```powershell
python -m dongfeng_registration.cli export-deliverables `
  --output-root deliverables\M18_final `
  --vehicle-model M18 `
  --optimized-pkl outputs\M18_joint_mainline\deliverables\result_interpolation_optimized.pkl `
  --compensated-perception-dir outputs\M18_dynamic\compensated_perception `
  --remove-compensated-perception-source `
  --optimization-output-dir outputs\M18_joint_mainline `
  --projection-qc-dir outputs\M18_projection_qc
```

## 9. 交付前检查

建议运行：

```powershell
python -m pytest
python -m dongfeng_registration.cli --help
python -m dongfeng_registration.cli run-delivery --help
python -m dongfeng_registration.cli normalize-dataset --help
python -m dongfeng_registration.cli export-deliverables --help
```

确认：

- `.gitignore` 已排除 `data/`、`data_cache/`、`outputs/`、`deliverables/` 和模型权重；
- README 和示例配置没有依赖开发机专用输出目录；
- 可通过 `run-delivery --config <delivery.yaml>` 一条命令启动完整主线；
- 最终投影说明统一指向 `ego2global_transformation_matrix_camera_reference_offset_optimized`；
- 最终交付物包含优化后的 pkl 和补偿后的点云。
