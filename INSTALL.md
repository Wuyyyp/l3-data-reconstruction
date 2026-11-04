# 安装 L3 重建环境

以下步骤面向 Linux + NVIDIA GPU。先准备 Git、ffmpeg、C/C++ 编译器和 CMake。编译 PyTorch 扩展还需 NVCC、CUDA 开发头文件和与对应 PyTorch 版本兼容的 CUDA toolkit。DVP/OpenMVS 另需 OpenCV 开发文件、Boost 等 C++ 依赖，具体见 DVP 包的编译说明。

## 1. 分别安装需要的组件

| 组件 | Python | PyTorch / torchvision | PyTorch wheel |
|---|---|---|---|
| pose | 3.11 | 2.8.0 / 0.23.0 | cu128 |
| dvp | 3.11 | 2.8.0 / 0.23.0 | cu128 |
| mesh | 3.12 | 2.7.1 / 0.22.1 | cu128 |
| streetgs | 3.10 | 2.1.1 / 0.16.1 | cu118 |

从仓库根目录执行。如果已有 python3.10、python3.11、python3.12，可用 `bash install.sh all` 顺序安装四个环境。只需要一个组件时，只运行对应一行。可用第二个参数指定该版本的基础 Python，例如 `bash install.sh streetgs /opt/python3.10/bin/python`。

```bash
bash install.sh pose
bash install.sh dvp
bash install.sh mesh
bash install.sh streetgs
cp local.env.example local.env
source ./server-env.sh
```

脚本在 `.envs/<组件名>/` 创建环境，先装构建工具和对应的 PyTorch wheel，再从组件目录执行 `pip install -r requirements.txt`。StreetGS 还会编译仓库自带的 simple-knn、Gaussian rasterizer 和 fused-ssim。DVP 的 MoGe 和 mesh 的 SAM3 需要访问上游 Git 仓库。切换组件时请为各自的原生扩展选用相应的 CUDA toolkit。

根目录 `requirements.txt` 是四个环境通用的构建工具清单。每个组件的 `requirements.txt` 才是该组件的 Python 依赖；`runtime-versions.txt` 是已有服务器的运行记录。清单没有锁定每个传递依赖。Pose 和 mesh 的直接 Git 依赖分别固定了 LoMa 和 SAM3 的提交。MoGe 的 Git 依赖由 `vendor/MoGe/pyproject.toml` 指定。

若使用单独下载的组件，在其目录中创建表中对应的 Python 环境，并运行以下步骤。将 TORCH、VISION 和 CUDA_INDEX 替换为表中对应值（例如 2.1.1、0.16.1、cu118）。本地包路径以当前组件目录为基准。

```bash
python -m pip install --upgrade pip
python -m pip install 'setuptools>=77,<82' wheel packaging ninja
python -m pip install --index-url "https://download.pytorch.org/whl/$CUDA_INDEX" \
  "torch==$TORCH" "torchvision==$VISION"
python -m pip install -r requirements.txt
```

DVP 先装 `numpy==2.2.6`，最后一步使用 `python -m pip install --no-build-isolation -r requirements.txt`。StreetGS 安装 Python 依赖后再运行 `python -m pip install --no-build-isolation -r requirements-native.txt`。安装完成可用 `python -m pip check` 核对 Python 包元数据。Pose 的新安装选择 opencv-python 4.12，以满足固定提交的 LoMa 声明；已有服务器使用的 headless 4.11 记录在 runtime-versions.txt。两个包都提供 cv2，请在同一 venv 中只选一个。

## 2. 准备原生工具和外部模型

| 组件 | 还需要准备的资源 | 位置或变量 |
|---|---|---|
| pose | 带 absolute pose prior 的 COLMAP；LoMa-B、检测器和描述器权重；语义分割模型 | `DVP_DELIVERY_COLMAP`；Pose YAML 的两个 COLMAP executable；`POSE_TORCH_HOME`、`POSE_HF_HOME` |
| dvp | 编译普通及 far-only APD，安装 OpenMVS 的 InterfaceCOLMAP 和 DensifyPointCloud；准备 MoGe 权重。独立方法 D 还用到语义模型 | `DVP_FULL_BIN`、`DVP_FAR_BIN`、`OPENMVS_BIN_DIR`、`DVP_MOGE_WEIGHTS`、`DVP_MASK2FORMER_WEIGHTS` |
| mesh | SAM3 权重、按车型分目录的 GLB 资产，以及可用的 NVIDIA EGL | `SAM3_CHECKPOINT`、`MESH_ASSET_ROOT` |
| streetgs | 随包 gsplat 源码编译出的 CUDA 扩展，地面预处理 pypatchworkpp，SegFormer-Cityscapes 权重；使用 cubemap 分支时还需要 nvdiffrast | `GSPLAT_CUDA_SO`、`GROUND_PREPROCESS_DEPS`（或装入该 venv）、`ROAD_SEGFORMER_WEIGHTS` |

COLMAP 编译补丁在 `pose-repair/third_party/`。APD 的两种构建命令见 `dvp-openmvs/DEFAULT_WORKFLOW.zh-CN.md`；OpenMVS 源码及构建说明在 `dvp-openmvs/vendor/openMVS/`。gsplat 扩展的编译、定位方式见 `streetgs/README.md`。StreetGS 的 requirements.txt 已包含 pypatchworkpp 1.4.1，需与 Python 3.10 ABI 匹配；如果平台需要自行编译，请参考 [Patchwork++ 上游安装说明](https://github.com/url-kaist/patchwork-plusplus#python)。已有兼容安装可通过 `GROUND_PREPROCESS_DEPS` 指定。直接运行 Pose CLI 时，设置 `TORCH_HOME="$POSE_TORCH_HOME"`、`HF_HOME="$POSE_HF_HOME"` 以使用准备好的模型缓存。

## 3. 检查与启动

```bash
source ./server-env.sh
python3 check-env.py                  # 检查四个组件
python3 check-env.py --component mesh # 或者只检查一个
```

该命令检查文件、工具路径和各组件解释器中的关键导入。通过后，按总 README 给出的 `streetgs/run_pipeline.py` 示例传入自己的 BA 交付目录和工作目录。当前两台现成服务器可在仓库根目录显式加载 `examples/l3-server-env.sh` 复用现有环境。新机器上的正式验收仍应包含实际 GPU 输入和运行测试。
