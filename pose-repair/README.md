# L3 位姿修复与 BA

## 安装 Python 依赖

整库检出时，在根目录运行 `bash install.sh pose`。独立使用本包时，从本目录运行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'setuptools>=77,<82' wheel packaging ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
python -m pip install -r requirements.txt
python -m pip check
```


命令入口在 `src/dongfeng_registration/cli/`。代码先修复基础位姿并补偿感知点云，再以相机观测和位姿先验运行 COLMAP BA。完整命令和其他车型用法见 `UPSTREAM_README.md`。

在独立的 Python 3.11 环境执行 `python -m pip install -e .`，或使用服务器示例中的 `$POSE_PYTHON` 并把本目录的 `src/` 加入 PYTHONPATH。根据配置准备 LoMa 等匹配模型、语义模型及带 absolute pose prior 的 COLMAP。第三方补丁在 `third_party/`，编译脚本在 `scripts/build_*.sh`。配置中的可执行文件和权重路径须指向实际安装位置。

```bash
cd "$POSE_ROOT"
export PYTHONPATH="$POSE_ROOT/src:${POSE_LOMA_SRC:-}"
# 复制配置后修改数据和输出目录，检查 base_config 及各模型路径
cp configs/m18.yaml configs/my-clip.yaml
"$POSE_PYTHON" -m dongfeng_registration.cli run-delivery --config configs/my-clip.yaml --dry-run
"$POSE_PYTHON" -m dongfeng_registration.cli run-delivery --config configs/my-clip.yaml
```

配置至少核对 `dataset_root`、`delivery.raw_pkl`、`delivery.canonical_root`、`delivery.base_config`、`delivery.run_output_root`、`delivery.start_frame`、`delivery.frame_count` 和 `colmap.*executable`。保留原始数据只读。

交付根包含 `manifest.json`、`result_interpolation_optimized.pkl`、`compensated_perception/`、`image_extrinsics/` 和 `colmap-ba/`。BA 的 `report.json` 给出所选帧及 complete/partial 范围。进入 101 帧流程前确认 BA 完整，并且 `sparse_after`、BA CSV 和实际送入的 JPEG 属于同一组相机。

M18 示例从 Pose 包目录运行。delivery 段的相对路径按这个工作目录解释；基础配置的 dataset_root / output_root 按配置文件所在目录解释。把 base_config 指到本次实际修改过的基础配置，核对 colmap_executable 和 absolute_pose_prior_colmap_executable。
