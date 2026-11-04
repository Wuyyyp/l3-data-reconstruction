# DVP + OpenMVS 图生点云

## 安装 Python 依赖

整库检出时，在根目录运行 `bash install.sh dvp`。独立使用本包时，从本目录运行：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'setuptools>=77,<82' wheel packaging ninja
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
python -m pip install numpy==2.2.6
python -m pip install --no-build-isolation -r requirements.txt
python -m pip check
```


本包包含 APD C++/CUDA 源码、深度过滤及格式转换脚本、MoGe 深度先验接口、L3 适配代码和 OpenMVS 源码。APD 产生多视图深度，OpenMVS 对经过滤的深度进行融合。原始说明及编译细节见 `UPSTREAM_README.md`、`DEFAULT_WORKFLOW.zh-CN.md` 和 `vendor/openMVS/BUILD.md`。

依赖包括 CUDA/NVCC、Boost、OpenCV、COLMAP、OpenMVS 的 `InterfaceCOLMAP` 和 `DensifyPointCloud`、Python 视觉环境及 MoGe 权重。使用服务器环境可直接指定现成工具，重新编译时按 `DEFAULT_WORKFLOW.zh-CN.md` 分别生成普通和 far-only APD；OpenMVS 的修改和许可随 `vendor/openMVS/` 提供。

**单独从 Pose 标准交付生成远景。** 下列命令以单条已完成 BA 的窗口为输入。`pose_delivery_input.py` 根据图像身份配对基础 CSV 与 BA CSV，生成方法 D 要求的目录结构，点云仍引用补偿后的 LiDAR。当 PKL 未记录 JPEG 去畸变标记时，传入该窗口实际使用的 Pose 配置。适配器核对配置里的图像约定、canonical 图像路径、时间戳、JPEG 尺寸及 BA 内参。方法 D 的固定/天空/动态过滤使用它自己的默认策略。

```bash
source ./server-env.sh
POSE_CONFIG=inputs/window/pose-config.yaml
"$DVP_PYTHON" "$DVP_ROOT/pose_delivery_input.py" "$DELIVERY" work/dvp-input --pose-config "$POSE_CONFIG"
"$DVP_PYTHON" "$DVP_ROOT/mvs_process/run_delivery_d.py" \
  --clip-root work/dvp-input --output-root work/dvp-out --gpu-ids 0 --far-only --check
# 移除 --check 运行完整融合
```

**复现当前 WYS 流程。** 使用 `l3/dvp_driver.py`，它复用当前窗口已生成的 sky/fixed mask，保留动态目标。总入口会自动准备数据；单独接入时需准备 `RUN/geometry/converted/{fixed_mask,sky_mask}`、WYS 版 BA 稀疏模型与 JPEG、匹配的 PKL，然后用以下命令。原位和 WYS 输入不能混用两套内参。

```bash
PYTHONPATH="$STREETGS_ROOT/workflow" "$GS_PYTHON" "$STREETGS_ROOT/workflow/dvp_input.py" "$RUN"
CLIP=$(find "$RUN/delivery" -mindepth 1 -maxdepth 1 -type d | head -1)
WYS_OLD_CASE="$RUN/geometry" WYS_CASE_ROOT="$RUN" \
 "$DVP_PYTHON" "$DVP_ROOT/l3/dvp_driver.py" --clip-root "$CLIP" \
 --output-root "$RUN/dvp" --gpu-ids 0 --far-only
```

远景开关保留距窗口所有 ego 位置的 XY 距离均大于 50 m 的点。最终文件位于结果窗口的 `10_openmvs_fusion/openmvs_filtered_fused.ply`；配合 `audit/D.complete` 判断是否完成。远景没有有效点时允许零点 PLY。单独验证远景策略可运行 `"$DVP_PYTHON" scripts/test_far_only.py`。
