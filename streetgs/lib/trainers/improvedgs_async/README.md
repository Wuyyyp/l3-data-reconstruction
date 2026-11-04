# 修正车辆 mask 的纯静态 ImprovedGS 入口

入口：scripts/run_improvedgs_async_static.py。训练核心：本目录 train.py；评测 evaluate.py；桥接 bridge.py；异步加载 async_pipeline.py。

车辆框修正位于 lib/utils/m18_utils.py：文件外参为 camera-to-ego，投影车辆坐标时使用 np.linalg.inv(ext)。自车遮挡仍由 bridge.valid_mask 在图像坐标上设置。天空仍使用 sky mask 筛选，不是直接半球。

默认数据 runs/dynamic_efficiency_20261013/converted，固定 BA，7相机101帧，静态排除车辆，无actor。默认25000步，随后评测707训练图（非独立测试集）。

- improvedgs：逐图affine颜色校正，无bilateral。
- improvedgs_bilateral：关闭affine，启用bilateral。
- 两者：gsplat渲染、ImprovedGS LAS/EAS/RAP/MU/GaussianAdam，4M背景预算，天空上限100k；点云4线程，mask单独预读，RGB/depth后台加载和CUDA预取。

在Pod内执行：

```bash
cd /mnt/zhoukaixuan_workspace/code/street_gaussians
PY=/mnt/zhoukaixuan_workspace/code/Improved-GS/.venv/bin/python
$PY scripts/run_improvedgs_async_static.py --variant improvedgs --gpu 0 --output output/static_mask_fixed_improvedgs_20261019
$PY scripts/run_improvedgs_async_static.py --variant improvedgs_bilateral --gpu 1 --output output/static_mask_fixed_bilateral_20261019
```

GPU编号仅为示例。启动器检查两次空闲及250GiB主机可用内存；忙则报错，不抢占。若同时运行，在两个终端分别执行。

输出目录必须不存在，每次从源重建本次input_ply，不读取历史训练的初始化缓存；同一次评测复用本次input_ply。不会清除共享OS缓存或CUDA编译缓存。此启动器只做从头训练，不能用它继续已有训练或旧错误mask checkpoint。

加 --smoke 可做3帧/21图元数据上的6步训练与2图评测，测试输出必须单独命名，不用于正式训练缓存。默认正式训练输出 train.log、eval.log、full/curve.jsonl、full/timing.json、train_exit.json、eval_exit.json、full/eval/results.json，滚动checkpoint，不输出中间图/PLY。

配置模板位于 configs/experiments_m18/static_async_improvedgs.json 和 static_async_improvedgs_bilateral.json。实际运行配置另存输出目录config.yaml；评测另存eval_config.yaml。底层依赖使用现有 Improved-GS venv、deps/improvedgs_dynamic、output/static_four_stages_20261017/deps 和相邻gsplat_shape_recovery_20261013/deps。旧实验代码快照未覆盖；不要再运行旧suite。
