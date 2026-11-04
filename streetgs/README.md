# StreetGS 整场训练与渲染

## 安装 Python 依赖

整库检出时，在根目录运行 `bash install.sh streetgs`。独立使用本包时，从本目录运行：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'setuptools>=77,<82' wheel packaging ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.1.1 torchvision==0.16.1
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -r requirements-native.txt
python -m pip check
```


默认训练入口为 `engine/affine_shadow/launch.py`，训练主体在同目录的 `code/train.py`；`run_pipeline.py` 负责把 L3 输入准备到该入口。当前配方包括 ImprovedGS、动态优化、图像 Affine、半球天空高斯、可学习地面和车辆阴影。地面 SH=0，其余按 `workflow/train_template.json` 和准备阶段生成的配置运行。训练监督使用精细固定遮挡 mask，WYS 画幅为 1600×900。

运行时需要 Python 3.10、PyTorch/CUDA、与 PyTorch ABI 匹配的 gsplat CUDA 扩展、fused-ssim、Open3D、pypatchworkpp 等。服务器使用 `server-env.sh` 指定已经验证过的解释器、CUDA 扩展和辅助依赖；独立安装时参考上游说明、`requirements.txt`、`submodules/` 与 `deps/gsplat/`。变量 `GSPLAT_CUDA_SO` 应指向编译出的兼容 .so，`STREETGS_EXTRA_PYTHONPATH`、`PREPARATION_DEPS`、`GROUND_PREPROCESS_DEPS` 可指定额外依赖。路面语义使用 `ROAD_SEGFORMER_WEIGHTS`（SegFormer B0 Cityscapes 权重目录）。源码包内含七相机精细固定遮挡模板。

已准备好的场景可以独立训练。准备目录应包含原图及有效 mask、深度、actors、背景/地面/车辆点云、地面 mask、anchors、shadow_templates。配置中的路径须指向本次工作目录。下例的 `CASE` 为本次运行的 `original` 目录。

```bash
source ./server-env.sh
ENGINE="$STREETGS_ROOT/engine/affine_shadow/launch.py"
"$GS_PYTHON" "$ENGINE" --config "$CASE/config.json" --stage validate
"$GS_PYTHON" "$ENGINE" --config "$CASE/config.json" --output-root "$CASE" --gpu 0 --stage train --stop-after 30000
"$GS_PYTHON" "$ENGINE" --config "$CASE/config.json" --output-root "$CASE" --gpu 0 --stage evaluate \
  --checkpoint "$CASE/full/checkpoint_030000.pth" --no-images
```

从本场景已有的中间 checkpoint 续训，在训练命令后加 `--resume "$CASE/full/checkpoint_015000.pth"`。整套流水线默认保存 15000、30000 步。训练曲线、log images 和 checkpoint 在 `original/full/`，评估在 `eval_checkpoint_030000/results.json`。普通 `train.py` 是上游入口，当前 L3 流程使用上述专用引擎。相机、LiDAR 及 mesh 的数据准备、可选开关与七视角导出见父目录 README。

新机器可用打包的 gsplat 源码构建与 PyTorch、CUDA 和目标 GPU 匹配的扩展。设置 CUDA_HOME、TORCH_CUDA_ARCH_LIST 和可写的 TORCH_EXTENSIONS_DIR 后，通过 `PYTHONPATH=deps/gsplat python -c 'from gsplat.cuda._backend import _C; print(_C.__file__)'` 编译或定位扩展，导出该路径为 GSPLAT_CUDA_SO。兼容的 prebuilt 扩展可直接使用。本机包版本记录在 runtime-versions.txt。StreetGS 扩展缓存目录需要可写。
