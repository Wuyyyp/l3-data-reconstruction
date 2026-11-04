# ImprovedGS 地面＋锚定完整训练入口

仓库入口 train.py / render.py 通过 model.ground.enabled=true 启用本模块。
支持 --training-mode improvedgs_static、improvedgs_dynamic 或 improvedgs_ground；
车辆是否参与训练由 model.nsg.include_obj 决定。未开启地面的旧命令保持原路径。

## 数据

当前接受已有 T3Prepared 数据，不在训练期间重新提取道路或生成分割。
source_path 指向 converted；父目录需要 input_manifest.json、actors.json。
converted 需要 images、valid_mask、sky_mask、target_mask、lidar_depth。
若父目录有 static_excluded，使用它作为动态排除区域，否则使用 target_mask。
图像尺寸可来自每条 view 的 width/height 或 manifest 全局 resolution。
静态模式排除自车与动态像素；完整场景模式保留车辆 RGB 监督，地面监督始终排除车辆。

ground.npz：xyz(N×3)、normals(N×3)、rotation(N×4，wxyz)、rgb(N×3，[0,1])。
地面和去地面背景使用减去 manifest.world_origin 后的同一坐标系。
ground.masks：cam_XX/000000.npy 二维地面 mask。
动态模式还需要 actors.json 配准仿射变换及 actor_ply_dir/points3D_obj_XXX.ply。
当前验证范围：SH=1、cubemap 天空、固定相机/车辆轨迹；此入口不优化位姿。

## 配置

示例：configs/experiments_m18/sample00024_improvedgs_ground.json。
选择新的 model_path，不覆盖历史输出。
model.ground：
- enabled=true，mode=anchored，split=true。
- init、background_ply、masks：上述输入路径。
- warmup=3000：只训练地面的步数，之后联合训练。
- point_cap=1200000：必须大于地面初始点数。
- split_threshold=0.0002。
- actor_ply_dir：完整场景模式必填。

improvedgs.background_budget 单独限制背景，不含地面和车辆。
训练总计划由 train.iterations 控制，--stop-after 用于提前保存并停止。
恢复训练要求配置与输出中的 pipeline_config.json 一致。

## 使用

在 street_gaussians 仓库中，使用具有依赖的 Python：
python train.py --training-mode improvedgs_static --config configs/experiments_m18/sample00024_improvedgs_ground.json --stage validate
python train.py --training-mode improvedgs_static --config configs/experiments_m18/sample00024_improvedgs_ground.json --gpu 1
python train.py --training-mode improvedgs_static --config CONFIG --gpu 1 --resume OUTPUT/checkpoint_003000.pth
python render.py --training-mode improvedgs_static --config CONFIG --gpu 1 --checkpoint OUTPUT/checkpoint_025000.pth

GPU 忙时显式 --allow-share 才共享，不终止其他进程。
--stage smoke 在第3/8/18步触发测试分裂；正式训练不要用此模式。
--eval-limit 控制验证图数量，--no-images 跳过渲染 PNG。
默认保留检查点；train.keep_latest_only=true 才清理前一个检查点。

## 约束和训练

地面固定初始法线/旋转，法线厚度1毫米。
中心 x=x0+n*0.1*tanh(offset)，法向位移不超过10厘米，附加位移平方惩罚。
切平面尺度、透明度、SH、offset 可训练。预热后继续优化地面，不冻结。
独立地面 RGB 监督使用 ground_mask、有效 mask、非车辆区域的交集。
背景/地面/车辆通过同一次 gsplat 排序和 alpha 合成，不修改 CUDA rasterizer。
动态车辆使用 AΣA转置 变换协方差，避免特征分解造成梯度异常。
米制期望深度用于 LiDAR 监督。
动态模式存在 mesh_depth/NAME.npz 时启用车辆 mesh 深度监督，
默认权重0.05（optim.lambda_mesh_depth）；对应像素不再重复使用 LiDAR 损失。
静态模式不读取车辆 mesh 深度。

地面在500～15000步每100步检查分裂，沿最大切平面轴分裂；
每次最多8192个父点，受梯度、尺寸和总量限制。
保留未变点的 Adam 动量，子点动量清零；保存分裂来源索引和累计统计。
checkpoint 保存背景、地面、车辆、天空、颜色校正、优化器与采样/RNG 状态。

## 渲染和指标

render.py 恢复检查点后输出验证集 PNG、results.json 和 ground_gaussians.ply。
地面 PLY 是标准3DGS字段（SH、opacity、log-scale、quaternion），在中心化坐标系中。
frame%5==4 为验证帧，数据可能使用全 clip LiDAR 初始化，属于已知几何下的留出图像评测。
PSNR 从原尺寸浮点 RGB [0,1] 计算，预测截断[0,1]，有效像素RGB通道 MSE：
PSNR=-10log10(MSE)，汇总为逐图 dB 平均。PNG不参与算分。
all：所有有效区域，动态模式包含车辆。
static：排除自车/动态区域，包含天空。
ground：投影地面 mask 中有效且非车辆的区域，不代表整幅道路。
vehicle：有效车辆区域；静态模式为空，结果为null。
训练包含每图仿射颜色校正，但此处评测 raw 未校正渲染，不拟合验证曝光。
分数包含几何、纹理、天空和曝光色彩偏差，不能只解释为几何质量。


性能路径及验证见 [PERFORMANCE.md](PERFORMANCE.md)。默认启用已验证加速，实际运行功能记录在 performance_features.json。
