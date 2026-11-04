# DVP-MVS

已完成上游 BA 的新交付数据使用一键[方案 D 入口](mvs_process/run_delivery_d.py)，详细约定见[新交付数据固定工作流](DEFAULT_WORKFLOW.zh-CN.md)。它直接读取交付 BA 位姿及稀疏模型，不重新 BA；普通 prepared sample-root 通用入口仍保留兼容默认值。

```bash
python mvs_process/run_delivery_d.py \
  --clip-root /data/l3data-reconstruction-bingxing/samples-restored/corner-case-t3/road-construction/2026_09_11_16_02_31_dlp_pilotGtParser \
  --output-root /data/l3data-reconstruction-bingxing/tem-test/delivery-d \
  --gpu-ids 0,1,2,3,4,5,6,7 --check
# 去掉 --check 后运行；每个 101 帧窗口只占一张 GPU，不足 101 帧的尾段也处理。
```

远景模式默认关闭。在一键 D 命令上追加 `--far-only`，只输出当前窗口所有自车位置 50 m 以外（地图 XY）的点；结果独立放在 `f0_100_far50` 等目录。计算阶段提前跳过可靠近处区域，最终 PLY 严格裁剪。规则、编译与恢复说明见[远景开关](DEFAULT_WORKFLOW.zh-CN.md#7-可选远景开关默认关闭)。

Zhenlong Yuan, Jingguo luo, Fei Shen, Zhaoxin Li, Cong Liu, Tianlu Mao and Zhaoqi Wang, [**DVP-MVS: Synergize Depth-Edge and Visibility Prior for Multi-View Stereo**](https://arxiv.org/pdf/2412.11578), AAAI 2025.
![](images/DVP-MVS-pipeline.png)

## About
DVP-MVS is the first method who combines the monocular depth estimation algorithm **Depth-Anything-V2** with multi-view stereo(**MVS**) algorithm to address patch deformation instability of PatchMatch-based MVS.

Our paper was accepted by **AAAI2025**!

If you find this project useful for your research, please cite:  

```
@inproceedings{yuan2025dvp,
  title={DVP-MVS: Synergize depth-edge and visibility prior for multi-view stereo},
  author={Yuan, Zhenlong and Luo, Jinguo and Shen, Fei and Li, Zhaoxin and Liu, Cong and Mao, Tianlu and Wang, Zhaoqi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={9},
  pages={9743--9752},
  year={2025}
}
```
## Dependencies

The code has been tested on Ubuntu 18.04 with Nvidia Titan RTX, and you can modify the CMakeList.txt to compile on Windows.
* [Cuda](https://developer.nvidia.cn/zh-cn/cuda-toolkit) >= 10.2
* [OpenCV](https://opencv.org/) >= 3.3.0
* [Boost](https://www.boost.org/) >= 1.62.0
* [cmake](https://cmake.org/) >= 2.8

**Besides make sure that your [GPU Compute Capability](https://en.wikipedia.org/wiki/CUDA) matches the CMakeList.txt!!!** Otherwise you won't get the depth results! For example, according to [GPU Compute Capability](https://en.wikipedia.org/wiki/CUDA), RTX3080's Compute Capability is 8.6. So you should set the 
cuda compilation parameter 'arch=compute_86,code=sm_86' or add a '-gencode arch=compute_86,code=sm_86'.

## Usage
## Usage

### L3 PKL data preprocessing

The repository-local eight-step preprocessing pipeline converts an M18/L3 PKL segment into
an OpenMVS `scene.mvs`, masks, and view-neighbor file. See
[data_preprocess/README.zh-CN.md](data_preprocess/README.zh-CN.md) and run:

```bash
bash data_preprocess/run_preprocess_1_to_8.sh --check
```

- Compile
>
    mkdir build & cd build
    cmake ..
    make

#### ETH Dataset

You may download [train](https://www.eth3d.net/data/multi_view_training_dslr_undistorted.7z) and [test](https://www.eth3d.net/data/multi_view_test_dslr_undistorted.7z) dataset from ETH3D, and use the script [*colmap2mvsnet.py*](./colmap2mvsnet.py) to convert the dataset format(you may refer to [MVSNet](https://github.com/YoYo000/MVSNet#file-formats)). You can use the "scale" option in the script to generate any resolution you need.

```python
python colmap2mvsnet.py --dense_folder <ETH3D data path, such as ./ETH3D/office> --save_folder <The path to save> --scale_factor 2 # half resolution
```

#### Tanks & Temples Dataset

We use the version provided by MVSNet. The dataset can be downloaded from [here](https://drive.google.com/file/d/1YArOJaX9WVLJh4757uE8AEREYkgszrCo/view), and the format is exactly what we need.

#### Other Dataset

Such as DTU and BlenderMVS, you may explore them yourself. !!! But remember to modify the [ReadCamera](https://github.com/whoiszzj/APD-MVS/blob/d9f9731235f4db05712024213e32346b6a01f5d6/APD.cpp#L84) function when you test on DTU !!!

### Run

After you prepare the dataset, and you want to run the test for ETH3D/office, you can follow this command line.

```bash
./APD <ETH3D root path>/office
```

The result will be saved in the folder office/APD, and the point cloud is saved as  "APD.ply"

It is very easy to use, and you can modify our code as you need.

## Acknowledgements

This code largely benefits from the following repositories: [APD-MVS](https://github.com/whoiszzj/APD-MVS), [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) and [ACMMP](https://github.com/GhiXu/ACMMP.git). Thanks to their authors for opening the source of their excellent works.


## CPU流水线与19次更新

见 [CPU流水线使用说明](mvs_process/CPU_PIPELINE.zh-CN.md)。原legacy24默认保留，legacy19与CPU流水线通过显式开关选择。

## COLMAP BA 接入

`mvs_process/run_batch_mvsnet_to_openmvs_ply.sh` 默认关闭 COLMAP BA，直接使用准备好的输入位姿运行 DVP-MVS 深度估计及 OpenMVS 点云融合。用 `--colmap-ba` 或 `DVP_COLMAP_BA=1` 显式开启 Dongfeng_Pose 的完整位置/旋转位姿先验 BA。位置与旋转先验标准差默认分别为 0.1 m、0.5°，为软约束。

`--no-colmap-ba` 可以覆盖环境变量并显式跳过 BA；用 `--ba-position-sigma`、`--ba-rotation-sigma` 调整约束强度。普通 BA 需要显式指定 `--colmap-ba --ba-mode standard`。缺少先验 BA 程序时不会自动降级。

BA 的优化位姿、稀疏模型和融合场景统一保存在结果目录的 `colmap_ba/`。使用同一组优化位姿准备 DVP 相机文件和 OpenMVS DMAP。详细用法见 [MVS 流程说明](mvs_process/README.zh-CN.md#colmap-位姿先验-ba默认关闭)。
