# DVP-MVS 与 OpenMVS 参数对齐及深度过滤脚本

本文记录当前仓库中相关文件的位置、参数和运行关系。相关实验文件主要位于 `moge3_dvp_sample_00001_clean7/`。

## 1. DVP-MVS 侧的参数和输入配置

| 文件 | 用途与关键配置 |
| --- | --- |
| [run_dvp.sh](moge3_dvp_sample_00001_clean7/run_dvp.sh) | 单卡运行入口；设置 `DVP_FUSION_MIN_CONSISTENT=2`、`DVP_DEPTH_SPECKLE_SIZE=7`，运行 `./build/APD`。 |
| [run_dvp_parallel.sh](moge3_dvp_sample_00001_clean7/run_dvp_parallel.sh) | 7 卡并行入口，使用相同融合和散斑参数。运行前会删除当前实验的 `scene/APD` 和 `scene/APD_parallel_barrier`。 |
| [prepare_dvp_scene.py](mvs_process/prepare_dvp_scene.py) | 复用 OpenMVS masks；图像保持宽高比并补边到 `960×640`；相机文件中的深度端点经 APD 扩展后对应配置的深度范围。 |
| [finalize_dvp_scene.py](mvs_process/finalize_dvp_scene.py) | 使用已验证的 OpenMVS 邻接关系，生成每张参考图包含 20 个邻居的 `scene/pair.txt`。邻接来源为 `neighbors_diverse_pm20_top20.tsv`，包含跨相机及前后 20 帧范围的多样性。 |

标准 MVSNet 格式没有统一的动态 mask 字段。本项目的 PKL 转换结果额外包含
`03_converted/dynamic_mask/`；预处理步骤 6 把非零动态区域与天空、车体 mask 合并。组合
mask 会用于 DVP 输入和逐轮深度清零，也会通过 OpenMVS `--mask-path` 用于最终过滤融合。

参数读取及 DVP 自身过滤、融合实现在 [APD.cpp](APD.cpp)：

- `DVP_FUSION_MIN_CONSISTENT`：最少一致源视图数，默认 2。
- `DVP_DEPTH_SPECKLE_SIZE`：深度散斑连通分量大小阈值，默认 7；调用 `FilterDepthSpeckles` 时相对深度阈值为 `0.05`。
- DVP 自身融合的一致性条件包括重投影误差 `<2 px`、相对深度差 `<0.01`、法线夹角约 `<10°`，另外还有动态一致性分数条件。

当前实现是部分参数和输入对齐，并不代表 DVP 与 OpenMVS 两套过滤、融合算法完全一致。尤其是视图计数口径、散斑实现和一致性判定，不能仅凭参数数值相同就认为等价。

上述 DVP 原生融合仍可供历史脚本使用；当前
[MVS 批处理](mvs_process/run_batch_mvsnet_to_openmvs_ply.sh) 设置 `DVP_SKIP_FUSION=1`，
只保留 DVP 逐视角深度估计，最终点云统一由 OpenMVS 过滤和融合生成。

## 2. OpenMVS 深度过滤与融合入口

运行脚本：

[moge3_dvp_sample_00001_clean7/openmvs_fusion/run_openmvs_fusion.sh](moge3_dvp_sample_00001_clean7/openmvs_fusion/run_openmvs_fusion.sh)

该脚本调用的可执行文件为：

```text
/mnt/nuplan/open-source-projects/openMVS/install/bin/OpenMVS/DensifyPointCloud
```

它使用已有 OpenMVS `scene.mvs`、mask 和邻接文件，以实验目录中的 `openmvs_fusion/dmaps/` 为工作目录，输出 `dvp_openmvs_fused.mvs` 及点云。

主要命令行参数：

```bash
--cuda-device -2 --max-threads 8
--resolution-level 1 --max-resolution 960 --min-resolution 640
--sub-resolution-levels 2
--number-views 20 --number-views-fuse 2
--iters 4 --geometric-iters 0
--fusion-mode 0 --fusion-filter 2
--fusion-depth-diff-threshold 0.010
--fusion-reprojection-threshold 2.5
--postprocess-dmaps 1
--estimate-colors 2 --estimate-normals 2
--ignore-mask-label 255
--tower-mode 0 --estimate-roi 0 --crop-to-roi 0
--filter-point-cloud 0
--remove-dmaps 0 --archive-type 2 -v 3
```

其中深度过滤／融合相关配置重点看 `--fusion-filter`、`--fusion-depth-diff-threshold`、`--fusion-reprojection-threshold` 和 `--postprocess-dmaps`。这是一个调用 OpenMVS 完成过滤与融合的脚本，不是仅输出过滤后深度图的独立入口。

## 3. OpenMVS 配套参数文件

配置文件：[openmvs_fusion/dense_config.cfg](moge3_dvp_sample_00001_clean7/openmvs_fusion/dense_config.cfg)

脚本通过 `--dense-config-file` 加载，当前内容为：

```ini
NCC Threshold Keep = 0.80
Descriptor Min Magnitude Threshold = 0.025
Normal Diff Threshold = 25
Speckle Size = 7
Interpolate Gap Size = 7
Random Smooth Depth = 0.02
Random Smooth Normal = 13
Random Smooth Bonus = 0.98
```

以上是文件中设置的值；具体参数是否参与某次运行，取决于 OpenMVS 实际执行的阶段。例如不能仅凭存在 `Interpolate Gap Size` 就断言本次运行执行了补洞。

## 4. DVP 深度图转换为 OpenMVS DMAP

转换脚本：[openmvs_fusion/convert_dvp_to_dmap.py](moge3_dvp_sample_00001_clean7/openmvs_fusion/convert_dvp_to_dmap.py)

处理流程：

1. 读取实验的 `manifest.json`，建立源图像标签与 DVP 图像 ID 的映射。
2. 读取 `scene/APD/<图像ID>/depths.dmb` 和 `scene/blocks/mask_<ID>.jpg`。
3. 根据 `padding_xy`、`content_size` 去除图像补边。
4. 仅保留有限、处于 `0.5–80 m` 且 mask 值不小于 128 的深度，其余置零。
5. 从已有 OpenMVS DMAP 模板读取图像标识、邻居 ID、相机内外参等元数据。
6. 根据深度重新估计相机坐标系法线，输出包含深度和法线的 `.dmap`。

脚本要求显式提供三个参数：

```text
--run        DVP 实验目录，包含 manifest.json 和 scene/
--templates  已有 OpenMVS depth*.dmap 模板所在目录
--output     转换后的 DMAP 输出目录
```

`run_openmvs_fusion.sh` 不会自动调用转换脚本，必须先准备好 `openmvs_fusion/dmaps/`。模板数量需与实验图像数量一致，去除补边后的深度图尺寸需与对应模板一致。

## 5. 运行关系与已有结果

整体流程为：

```text
prepare_dvp_scene.py + finalize_dvp_scene.py
    → DVP 输入、mask、先验及 pair.txt
    → run_dvp.sh 或 run_dvp_parallel.sh
    → scene/APD/<图像ID>/depths.dmb
    → convert_dvp_to_dmap.py（需要已有 OpenMVS DMAP 模板）
    → openmvs_fusion/dmaps/depth*.dmap
    → run_openmvs_fusion.sh
    → dvp_openmvs_fused.mvs / dvp_openmvs_fused.ply
```

准备好 DMAP 后，可在仓库根目录执行：

```bash
bash moge3_dvp_sample_00001_clean7/openmvs_fusion/run_openmvs_fusion.sh
```

脚本中的可执行文件、输入场景、mask、邻接文件和输出位置使用当前机器的绝对路径，迁移环境时需要调整。

已有日志 [openmvs_fusion_retry.log](moge3_dvp_sample_00001_clean7/openmvs_fusion/openmvs_fusion_retry.log) 第 4273 行记录：707 张深度图完成过滤融合，生成 2,892,630 个点。对应结果为 [dvp_openmvs_fused.ply](moge3_dvp_sample_00001_clean7/openmvs_fusion/dvp_openmvs_fused.ply)。这是已有运行记录，本次整理未重新运行重建流程。
