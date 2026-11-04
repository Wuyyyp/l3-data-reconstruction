# 道路提取 → 地面＋锚定：统一静态重建流水线

正式入口：scripts/run_ground_pipeline.py。
内部步骤位于 lib/preprocessing/ground_pipeline/，训练复用 lib/trainers/improvedgs_ground/。
不需要手工执行旧的 extract/refine/prepare 实验脚本，也不依赖旧实验的提取输出。

## 一条命令跑完整流程

在 world-model / zhoukaixuan-mvs-worker-0 的仓库目录中执行：

cd /mnt/zhoukaixuan_workspace/code/street_gaussians
/mnt/zhoukaixuan_workspace/code/Improved-GS/.venv/bin/python scripts/run_ground_pipeline.py --config configs/pipelines/sample00024_ground_pipeline.json --gpu 1

默认依次执行：
1. 原始逐帧 LiDAR：Patchwork++、自适应孤立点过滤、与近处道路连接的局部平面筛选。
2. 分离地面、动态点和静态背景：任何帧速度绝对值超过0.5的 track，在全段中按3D框排除；
   利用邻近地面补充地面分类；保存一个选定中间帧的动态点云用于检查。
3. 全局局部平面细化：从剩余静态背景中去掉有道路支持的残留地面。
4. 生成训练数据：道路局部 PCA、法线和旋转、平面投影、去地面背景体素降采样。
5. 道路点投影为图像地面 mask，并结合背景/LiDAR遮挡、自车、动态、天空mask排除误标。
6. 检查数据并自动生成 train_config.json，训练25,000步。
7. 恢复最终checkpoint，渲染验证集、计算PSNR并导出独立地面高斯PLY。

默认地面预热3,000步，之后联合训练；地面切向分裂继续到15,000步，默认上限120万。
地面没有新增独立合并或裁剪机制。背景保持 ImprovedGS 原有增密、裁剪。
当前是静态流程，actors.json为空，动态目标不参与重建。
保持当前模型：SH=1、cubemap天空、LiDAR深度；未加入天空高斯球或DVP深度。
已集成的 packed、融合masked SSIM、异步加载等加速继续生效。

## 换数据

/mnt/zhoukaixuan_workspace/code/Improved-GS/.venv/bin/python scripts/run_ground_pipeline.py --source /path/to/converted --output output/my_ground_run --ego-mask /path/to/fixed_mask --gpu 1

输出必须是仓库 output 下的新目录。不会修改源数据、已有提取结果或已有训练。
已有目录必须显式 --resume。GPU忙时可显式 --allow-share，不会停止其他任务。
示例配置使用此前指定的 sample01020 自车mask；新clip应指定对应的自车mask。
不提供 --ego-mask 时尝试 source/fixed_mask，不会悄悄取消自车排除。

## 输入约定

原始 converted 目录需要：
- pointcloud.npz：pointcloud 和 camera_projection 两个按整数帧号索引的字典。
  点为ego坐标系米制N×3，投影每3列为相机编号/u/v。
- input_manifest.json：records、calibrations，包含每视图 camera_to_world、K、原图尺寸。
- ego_pose/000000.txt：对应 LiDAR 的ego到world变换。
- images/000000_00.jpg。
- dynamic_mask、sky_mask：同名PNG/JPG，非零动态像素需排除。
- lidar_depth/000000_00.npy：mask/value稀疏深度字典。
- track/track_info.txt：与既有 converted 格式一致的带速度3D框。
- 显式给定的自车mask目录：优先每帧同名文件，否则回退到000000_相机号模板。
  非零表示自车排除区；支持0/1和0/255。

位姿直接沿用输入，不进行BA。世界坐标需Z向上，车辆框与点云需在同一ego坐标系。
当前复用的方法包含近地面、连通性与局部平面假设，不等于语义分割真值。
稀疏/不连通路面可能漏掉，低矮路沿可能被划入地面，预览图保留供检查。

图像按校准比例缩到宽1600：
1920×1280 → 1600×1066；3840×2160 → 1600×900。
生成对应K、mask、深度。位姿等无需改变的资料软连接到源数据。

## 分阶段与续跑

仅预处理：
python scripts/run_ground_pipeline.py --config configs/pipelines/sample00024_ground_pipeline.json --stage prepare

检查道路点、data/previews 后再训练：
python scripts/run_ground_pipeline.py --output output/sample00024_ground_end_to_end --resume --stage train --gpu 1

完整继续（训练完成后自动评测）：
python scripts/run_ground_pipeline.py --output output/sample00024_ground_end_to_end --resume --gpu 1

仅评测：
python scripts/run_ground_pipeline.py --output output/sample00024_ground_end_to_end --resume --stage evaluate --gpu 1

还支持 --stage extract / separate / refine，依次执行到指定阶段；
--stage validate 只检查输入，不创建输出。
训练/评测要求至少有一张 frame%5==4 的验证图和一张其他帧的训练图。

流水线保存参数、输入文件元信息指纹、预处理代码指纹和阶段完成标记。
--resume 复用已完成且未改动的阶段；输入或预处理代码/参数改变时要求新输出。
失败阶段的临时目录保留为 unfinished，不覆盖已完成结果。
训练从最新 checkpoint 恢复，较新的未保存日志行先归档再回退；
中间日志损坏会拒绝续跑，不静默忽略。
进程锁传给直接子进程，避免仍在处理时另一个入口修改同一目录。

## 主要参数

--width 1600
--workers 4
--frames 0:100 或 --frames 0,1,4
--cameras 0,1,2,3,4,9,10
--ground-voxel 0.10
--background-voxel 0.20
--normal-z-min 0.94
--plane-rms-max 0.06
--support-radius 1.0
--ground-tolerance 0.15
--occlusion-tolerance 0.15
--dynamic-dilation 11
--ground-point-cap 1200000
--background-budget 3403964
--warmup 3000
--iterations 25000

--stop-after N 只控制本次训练停在哪一步，不改变总训练计划；
--eval-limit N 限制评测数量。少量帧测试应显式选择 --warmup 2 --stop-after 10，
不能把这种冒烟测试结果作为完整重建质量。

## 输出结构

pipeline_config.json / input_fingerprint.json / pipeline_status.json
logs/                              各预处理阶段日志
stages/                            成功阶段标记
preprocess/extraction/             逐帧道路索引、world道路PLY、投影预览
preprocess/separation/             静态背景和单帧动态点云
preprocess/refined/                细化背景、额外剔除的道路点
data/input_manifest.json           校准、世界原点和视图列表
data/prepared/ground.npz            训练用道路点、法线、旋转、RGB
data/prepared/nonground.ply         中心化的静态背景点
data/ground_masks/                 图像地面mask
data/converted/                    图像、有效/动态/天空mask、稀疏深度
data/previews/                     绿=地面、紫=自车、红=动态
data/input_validation.json          mask、深度和几何检查结果
train_config.json                  自动生成的正式训练配置
train/checkpoint_025000.pth
train/eval_checkpoint_025000/       渲染图、指标、独立地面高斯

原始道路PLY是world坐标；训练NPZ/背景及最终高斯使用减去manifest.world_origin后的中心化坐标。
不要把两种坐标系直接混合。道路mask来自稀疏投影，有遮挡和未覆盖区域，不会自动补满整条道路。
源点云、初始化和地面mask可以使用全clip；验证属于留出图像评测，不是完全隔离验证帧传感器输入。

## 已验证

sample00024 的0、1、4帧，00和10相机：
完整经历提取、分离、细化、数据生成、短训练、checkpoint恢复和渲染。
三个帧的 Patchwork++ 最终道路索引与原已验证脚本逐项一致。
CPU测试：tests/test_ground_pipeline_cpu.py，覆盖断点日志恢复和mask约定。
Patchwork++扩展已放到 deps/ground_preprocess，当前环境使用Python3.10；不再从HUGSIM实验目录导入。
