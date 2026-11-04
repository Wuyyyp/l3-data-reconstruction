# 运行加速与 PPU 兼容性

本文说明正式主线采用的加速策略、资源占用和阿里云 PAI 真武 PPU 上的兼容边界。
PPU 运行必须使用平台提供的 PPU 镜像和软件源。不要把 NVIDIA 环境中的 CUDA 二进制、
PyTorch wheel 或扩展包直接复制到 PPU 实例。

## 1. 正式主线的加速策略

### 图像与 LoMa 特征缓存

- 图像统一按 `image_cache.max_long_side` 缩放后进入 CPU 缓存，默认长边 1600。
- LoMa detector/descriptor 对每张图像只计算一次。
- 特征默认直接保存在 GPU，`loma_feature_cache_gpu_max_items` 默认 5000。
- 预评估和正式匹配位于同一个 `match-delivery` 进程，共用模型、图像缓存和 LoMa 特征缓存。
- GPU 缓存不足时可改为 `two_tier` 或 `cpu`，匹配结果不应因缓存位置改变。

CPU 图像缓存和 GPU 特征缓存通过减少重复解码、缩放和特征提取换取速度。正式运行前应根据
clip 图像数量检查内存和显存。当前 T3 规模建议至少准备 64 GB 系统内存和 24 GB 显存。

### 语义分割

- Mask2Former 默认使用 FP16 autocast。
- 默认 `batch_size: 4`，processor 使用 fast 实现。
- 模型在持久进程内加载一次，按相机批量推理。
- `class_masks` 保存类别真值，动态二值 mask 可由类别图重新生成。
- preview、匹配可视化和重投影 QC 默认均按 1% 采样。

### 匹配筛选与联合优化

- retained 筛选以 relation/cache 组为单位使用多进程，子进程使用 `spawn`，不继承 GPU 上下文。
- DataFrame 中间列改为 NumPy 批量计算后一次性写入，减少逐列插入开销。
- 联合优化和全量误差评价使用 PyTorch 向量化计算；评价异常长尾再走 CPU 精确复算。
- 优化后只保存精确汇总、异常明细和 1% QC 样本，不再写出多份全量逐匹配表。

### 动态点云补偿与导出

- PCD 预扫描、目标轨迹和逐帧写出使用受控线程池。
- 点云使用容量受限的内存缓存，单 LiDAR 不变结果优先使用 hardlink。
- pkl、QC 和交付物导出避免重复读取同一批点云和图像。

## 2. PPU 兼容性结论

阿里云官方说明真武 PPU 兼容 CUDA 和 Triton 编程接口，并提供 PPU 版本的 PyTorch、
torchvision 等软件包。但 NVIDIA CUDA SDK 编译出来的二进制不能直接在 PPU 上运行。

| 功能 | PPU 判断 | 处理方式 |
| --- | --- | --- |
| PyTorch `cuda:0`、张量计算和优化器 | 可用 | 使用 PAI-PPU 官方镜像中的 PyTorch |
| Mask2Former FP16 autocast | 预计可用，需 smoke test | 失败时改为 `inference_dtype: fp32`，并减小 batch |
| LoMa detector/descriptor/matcher | 预计可用，需模型级 smoke test | 保留 PPU 镜像的 torch/torchvision，不安装通用 CUDA wheel |
| LoMa GPU 特征缓存 | 可用 | 显存不足时降低 5000 上限或切换 `two_tier` |
| PyTorch 图优化和 GPU 评价 | 预计可用 | `torch.cuda.is_available()` 为假时代码可回退 CPU，但速度显著降低 |
| `torch.compile(..., backend="cudagraphs")` | 条件可用 | 正式配置默认 `off`；启用失败会自动回退 eager |
| CPU 多进程筛选、线程化 PCD/QC | 可用 | 与 PPU 无关，按 CPU 和内存调整 workers |
| translation-only ICP | 可用 | 使用 SciPy，不依赖 Open3D CUDA |
| Open3D CUDA wheel | 不可直接复用 NVIDIA wheel | 使用 PPU 专用包；没有时保持 translation-only 路线 |
| COLMAP/OpenMVS CPU 构建 | 可用 | 在目标 Linux 环境重新构建 |
| COLMAP/OpenMVS NVIDIA CUDA 构建 | 不可直接运行 | 必须在 PPU SDK 环境重新编译；未验证前使用 CPU 构建 |

`torch.compile(cudagraphs)` 只出现在 benchmark 配置中，正式车型配置保持关闭，因此它不会阻断
主线。PPU 虽兼容 Triton 接口，但具体模型图是否能稳定编译仍取决于 PPU SDK、PyTorch 和
模型算子组合，不能只根据 `torch.cuda.is_available()` 判断。

## 3. PPU 环境安装要求

使用 PAI-PPU 官方训练镜像，并保持镜像自带的 torch、torchvision、triton 和 CUDA 兼容层。
项目安装使用：

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --no-build-isolation --no-deps -e .
```

不要直接执行会从通用 PyPI 替换 torch/torchvision 的安装命令。以下类型的软件包需要使用
PPU 专用版本或在 PPU 实例内重新编译：

- torch、torchvision、triton；
- Open3D、MMCV 等包含 CUDA 扩展的 wheel；
- COLMAP、OpenMVS 等包含 CUDA 代码的本地可执行文件。

官方参考：

- [PPU 编程指南](https://help.aliyun.com/zh/document_detail/2871803.html)
- [PAI-PPU 最佳实践](https://help.aliyun.com/zh/pai/use-cases/pai-pg1-getting-started-best-practices)
- [PyTorch 兼容性](https://help.aliyun.com/zh/document_detail/2871813.html)

## 4. 首次部署验证

先确认实例实际获得 PPU：

```bash
ppu-smi -L
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
```

然后用 5 至 10 帧依次验证：

1. Mask2Former FP16、batch 4；
2. LoMa eager 模式和 GPU 特征缓存；
3. 一组 retained 筛选；
4. 小规模联合优化和 GPU 评价；
5. 完整 `run-delivery --dry-run`；
6. 如需重建，再单独验证 COLMAP/OpenMVS。

PPU 首次验证建议显式保持：

```yaml
semantic_segmentation:
  inference_dtype: fp16
  batch_size: 4
matching:
  model_device: cuda:0
  loma_matcher_compile_backend: off
  loma_feature_cache_gpu_max_items: 5000
```

出现显存不足时，依次降低语义 batch、GPU 特征缓存上限和评价 batch。若 FP16 算子失败，
先改为 FP32 验证正确性。不要先更换 PyTorch 或安装 NVIDIA wheel。

## 5. 重建阶段的特殊说明

项目内 `tools/colmap_pose_prior/bin/colmap` 默认按 CPU 构建，带初始位姿先验的 BA 本身不依赖
CUDA。COLMAP SIFT GPU 和 PatchMatch、OpenMVS CUDA 则依赖目标平台的 CUDA 兼容编译环境。

在 PPU 上没有完成重编译和 smoke test 前，可以采用以下保守路线：

- COLMAP 特征提取和匹配设置 `use_gpu: false`；
- 位姿先验 BA 使用项目内 CPU 构建；
- 稠密重建使用 OpenMVS CPU 构建，或暂时关闭 `run_dense`；
- 不复制 Windows/WSL NVIDIA 环境生成的 `tools/openmvs-cuda`。

PPU 上重新编译 CUDA 版本时，必须使用实例中的 PPU SDK、`nvcc` 兼容编译器和对应官方镜像，
并重新执行小规模重建验证。通过 NVIDIA GPU 编译出的现有二进制不属于可交付的跨平台产物。
