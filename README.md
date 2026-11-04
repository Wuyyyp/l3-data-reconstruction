# L3 七视角 StreetGS 重建

本目录按运行边界拆成四个代码包。示例针对一个连续 101 帧的 L3 窗口，7 个相机共 707 张图。每个组件的 README 包含独立入口、输入、输出和依赖。运行目录请放在自己的本地盘，保留输入文件直到本次训练、渲染及归档结束。一些中间目录用符号链接引用源图和点云。

| 组件 | 输入 | 可交给下游的结果 |
|---|---|---|
| [pose-repair](pose-repair/README.md) | L3 原始 PKL、相机图像、LiDAR、标定、ego 和 clip 配置 | `deliverables/`，包括补偿后的 PCD、基础修复 PKL 和 BA 相机模型 |
| [dvp-openmvs](dvp-openmvs/README.md) | 与 JPEG 配套的 BA 相机、稀疏点及补偿 LiDAR | 世界坐标彩色远景 PLY |
| [vehicle-mesh](vehicle-mesh/README.md) | 101 帧七视图、原生轨迹、LiDAR、车辆 glTF/GLB 资产 | 每辆车每帧一个共享世界姿态，七个相机使用同一个矩阵 |
| [streetgs](streetgs/README.md) | 上述结果，或已准备好的 StreetGS 场景 | 原位高斯、checkpoint、707 图评估和七视角视频 |

## 安装

新机器先看 [INSTALL.md](INSTALL.md)。整套重建使用四个独立的 Python 环境，可在仓库根目录一次安装：

```bash
bash install.sh all
```

安装指南列出了各组件的 Python、PyTorch、CUDA、编译依赖、外部权重和对应环境变量。只使用一个组件时，运行相应的安装命令即可。

## 从新的 Git 检出开始

在仓库根目录执行以下命令。默认入口通过脚本自身的位置定位四个代码包；可以在 local.env 里用相对仓库根目录的路径覆盖模板。Linux 上需要 NVIDIA GPU、CUDA 和 ffmpeg。四个组件使用各自的 Python 环境；关键版本见各目录的 runtime-versions.txt。Pose 的 COLMAP 需支持项目的绝对位姿先验，StreetGS 的 gsplat 扩展需与当前 PyTorch、CUDA 和 GPU 架构匹配。

```bash
cp local.env.example local.env
source ./server-env.sh
python3 check-env.py                    # 也可加 --component pose|dvp|mesh|streetgs
```

按各组件 README 安装 Python 依赖及原生扩展。约定目录为 `.envs/{pose,dvp,mesh,streetgs}/`（解释器）、`tools/`（COLMAP、OpenMVS、LoMa）、`weights/`（MoGe、SAM3、语义模型）和 `assets/3d-assets/`（车辆 GLB）。已有安装可在 local.env 指向外部目录。独立拷贝子包时按该组件 README 设置相同变量。当前两台 L3 服务器如需复用旧安装，可在仓库根目录显式执行 `source ./examples/l3-server-env.sh`。该配置只用于这套部署。

CLI 的 `--delivery`、`--work` 允许传相对当前目录的路径。运行时会解析为绝对路径写入日志、checkpoint 和工作目录中的链接，用来在子进程中定位同一份数据。因此源码目录可以搬迁，已生成的运行目录及外部数据需保留原路径或重新绑定链接。历史示例、上游文档和非默认实验脚本中可能仍有原部署路径。

## 从原始数据开始

1. 在 Pose 包里从 `configs/m18.yaml` 派生本窗口的配置，设置数据、模型、本次输出和帧范围，运行 `run-delivery`。已经做过修复和 BA 的数据可直接从下一步开始。BA 的相机以 `colmap-ba/sparse_after/` 与 `image_camera_to_world_after_ba.csv` 为准。基础 PKL 和基础外参有各自的用途。
2. 在本服务器加载运行环境示例。四个组件可以放在同一父目录，也可分别设置 `POSE_ROOT`、`DVP_ROOT`、`MESH_ROOT`。换服务器时，按组件 README 设置 Python、编译工具、权重及资产库。运行环境示例里指定的目录是外部依赖，不随源码包分发。
3. 对新的本地工作目录运行如下命令。默认从头训练 30000 步。路径中的 `DELIVERY` 是含 PKL 和 `colmap-ba/` 的那一层，`RUN` 是单条窗口的输出目录。

```bash
source ./server-env.sh
DELIVERY=inputs/window/deliverables
RUN=work/clip_M18_example_f0_100
"$GS_PYTHON" "$STREETGS_ROOT/run_pipeline.py" \
  --delivery "$DELIVERY" --work "$RUN" --gpu 0
```

读取器要求该窗口的 BA 完整覆盖 101 帧、707 张图，`sparse_after` 为去畸变 PINHOLE 模型。WYS 用完整画幅缩放到 1600×900，两个轴分别缩放内参及稀疏观测像素。如果源交付已将内部编号归零且没有 `window_verification.json`，用 `--source-start 175` 等参数记录真实起始帧。对有校验记录的 VLA 窗口可直接使用原始窗口目录。

单窗口内，生成 WYS 图像和几何后，DVP、车辆配准及原生 LiDAR 初始化可并行。LiDAR 和 DVP 先合成背景，再用三维车体与深度关系去除由 mesh 替换的车点，并保留其余目标轨迹和前景深度。地面拟合、车辆锚点、阴影模板和路面 mask 完成后启动整场训练。

## 组合方式

| 需求 | 入口或参数 |
|---|---|
| 只生成 WYS 图像、点云投影和 mesh 所需的 `geometry/` | 在总入口加 `--geometry-only` |
| 只准备训练输入并验证，到正式训练前停下 | `--prepare-only` |
| 自己提供 DVP 或其他方法产生的远景点云 | `--far-ply /path/to/far.ply`，点坐标必须在同一世界系，所有点到 ego 轨迹的最短 XY 距离应大于 50 m |
| 只用 LiDAR 背景 | `--no-far` |
| 使用已配准的车辆 | `--registered-mesh /path/to/mesh_registration`，需同时提供原资产路径、mask 及同窗口导出文件 |
| 所有目标使用原生轨迹和 LiDAR | `--no-mesh`，可和 `--no-far` 组合 |
| 训练已有的 prepared 场景 | 单独调用 StreetGS 的 `engine/affine_shadow/launch.py` |
| 单独生成远景或仅做车辆配准 | 分别使用 DVP 或 mesh README 中的入口 |

同一工作目录可从已完成的阶段继续。切换窗口、远景 PLY 或 mesh 策略时使用新的工作目录。若某阶段异常退出，保留日志；遇到残缺阶段输出时先检查该阶段目录，已完成的阶段容易通过 `stages/*.json` 识别。

## 输出与验收

`original/full/checkpoint_030000.pth` 是最终训练 checkpoint。`original/full/eval_checkpoint_030000/results.json` 记录全量评估。当前配置使用 train-all，707 图均参与训练，评估值属于原位重建指标。原位视频使用 `streetgs/workflow/render_l3_wys.py`，上四下三、3840×1120、10 fps；视频里的机盖与后视镜用精细 mask 贴回原图。训练监督也使用精细固定遮挡 mask。

```bash
CUDA_VISIBLE_DEVICES=0 "$GS_PYTHON" "$STREETGS_ROOT/workflow/render_l3_wys.py" "$RUN" "$RUN/video"
```

各包带有独立的源码目录；`dist/` 保存相同目录的压缩版本和 SHA-256。Git 可从四个子目录分别建库，也可将整个目录作为单一仓库。第三方实现的许可文件保存在对应子目录。权重、车辆资产、场景数据、环境、编译产物与训练输出由运行方另行提供。

## 当前服务器验收范围

在实际 101 帧窗口 `20260101074025_DF_f175_275` 上复查了 WYS 全画幅图像转换、BA 像素变换和 LiDAR 投影，并从源码包加载训练引擎与兼容 CUDA 扩展。StreetGS validate 读取 707 个视图和 159,652 个地面点。Mesh 检查覆盖 54 辆车、3,955 个车帧、27,685 个相机变换，原标注字段通过逐帧对照。Pose 运行 32 项交付/BA 测试，DVP 运行 9 项远景测试。交付时做了真实输入与独立入口检查；没有从原始数据重新运行整条 BA、全量 SAM 配准、稠密融合和 30000 步训练。每包的 runtime-versions.txt 记录关键运行库版本，外部模型与 CUDA ABI 需与本机匹配。
