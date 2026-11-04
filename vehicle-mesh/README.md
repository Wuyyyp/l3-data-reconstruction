# 车辆模型选取与七视角配准

## 安装 Python 依赖

整库检出时，在根目录运行 `bash install.sh mesh`。独立使用本包时，从本目录运行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'setuptools>=77,<82' wheel packaging ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
python -m pip install -r requirements.txt
python -m pip check
```


`register_sample.py` 是 L3 车辆入口。`shared_mesh.py` 从 `MESH_ASSET_ROOT` 搜索模型，使用标注的车辆尺寸和车框内 LiDAR 对候选资产打分。`registration/run_registration.py` 结合 SAM3 掩膜及多帧、多视角观测优化同一车辆的姿态。每辆车每帧只有一个世界变换；有观测的七个相机投影同一个变换。静止和移动的汽车、SUV、卡车、巴士等四轮机动车都参与，行人、骑行者等保留原轨迹与原标注。

独立环境使用 Python 3.12、适配 CUDA 的 PyTorch、SAM3、PyOpenGL/EGL、pyrender 和 trimesh。参见 `registration/requirements.txt`、`registration/install_dependencies.sh` 和 `registration/README.md`。为 `SAM3_CHECKPOINT` 提供权重，为 `MESH_ASSET_ROOT` 提供按车型目录组织的 GLB 资产。服务器示例：

```bash
source ./server-env.sh
# RUN 已经含有下列 geometry 输入；同样格式也可由其他项目生成
( export PYOPENGL_PLATFORM=egl
  if [[ -n "${EGL_LIB_DIR:-}" ]]; then export LD_LIBRARY_PATH="$EGL_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
  if [[ -n "${MESH_ENV_ACTIVATE:-}" ]]; then source "$MESH_ENV_ACTIVATE"; unset PYTHONPATH; fi
  "$MESH_PYTHON" "$MESH_ROOT/register_sample.py" "$RUN" 0 )
```

输入为 `RUN/geometry/input_manifest.json`（707 个视图）、`converted/pointcloud.npz` 与逐帧深度，以及 `mesh_input/` 下的 101 帧 PKL、七个同步视频及视频 manifest。可用总入口的 `--geometry-only` 生成，也可用 `registration/run_registration.py --help` 从已经选好车型的配置启动底层配准。

配准导出到 `RUN/mesh_inputs/mesh_registration/`，包括编辑后的 PKL、`placements.json`、`view_transforms.json`、`asset_matches.json` 和注册掩膜。真实车的颜色仍来自 GT，资产只提供几何。验收文件 `RUN/shared_mesh_acceptance.json` 核对每辆车的七相机变换、帧覆盖，以及原标注字段是否逐帧保留。复用结果须同时保留资产和被符号链接引用的注册 mask；完整 RUN 建议整体保存。
