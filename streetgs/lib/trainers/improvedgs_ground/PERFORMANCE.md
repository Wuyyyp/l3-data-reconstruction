# 地面＋锚定 ImprovedGS 性能接入

适用入口：train.py / render.py，model.ground.enabled=true。
实现目录：lib/trainers/improvedgs_ground。
静态和动态示例配置默认启用以下路径，保持原地面/锚定/深度监督、SH阶数、分裂策略。

## 已接入的加速

- 异步 CPU 数据加载、mask 预加载、独立 CUDA stream 预取，队列最多两个视图。
- 原 ImprovedGS LAS/EAS/RAP/MU 和 ours_adam 背景优化器保持启用。
- mesh 车辆直接读取固定配准仿射矩阵，跳过旧的逐车 rigid pose 转换与重复四元数展开。
- 车辆位置与 AΣA转置 协方差批量变换，仿射数据缓存最多两组。
  缓存键包含帧、车辆集合、各车辆点数，拓扑改变后自动重建。
- 相机投影矩阵缓存：按尺寸、设备、dtype、矩阵地址和版本失效。
- packed gsplat 投影；将可见 Gaussian 的梯度、absgrad、半径回填到完整点序。
  背景、车辆和地面分裂仍看到各自正确的索引和统计。
- 合并 foreach Adam：复用参数组，学习率变化动态更新，组内学习率分叉或拓扑变化时重建。
  各车辆原 optimizer 保留状态所有权，兼容 checkpoint、裁剪、克隆、分裂及晚出现车辆。
- Fourier 时间基缓存、一次屏幕梯度计算的复用、避免统计时的 CUDA nonzero 索引。
- 融合 CUDA masked SSIM，维持原来11×11全有效窗口规则；
  L1/深度等改为 masked reduction，避免读取 GPU 布尔量造成 CPU 等待。
- 地面预热仅渲染地面，不再额外渲染无监督梯度的全场景。
- 标量日志分批回传，去掉每步全设备 synchronize 和逐参数扫描。
  每步保留异步 loss 有限性断言；首步、定期间隔及 checkpoint 仍完整检查。
- 训练结束不再默认全数据重新哈希；需要时 data.audit_payloads=true。

原 render.batch_actor_pose / batch_actor_xyz / batch_actor_world 等旧 rigid-backend 开关
不控制新的固定 mesh 仿射路径；实际启用项写入每次运行的 performance_features.json。
这里不启用轨迹优化，也不通过降分辨率、降低 SH 或减少训练步数提速。

## 默认开关

render.gsplat_packed=true
render.cache_fourier=true
optim.merge_actor_adam=true
data.streaming_load=true
data.cuda_prefetch=true
data.audit_gpu_payload=false
data.audit_payloads=false
train.log_interval=100
train.audit_interval=1000
train.profile_interval=100

profile_interval=0 不记录 CUDA 时间事件；正值按间隔采样，首步也采样。
curve.jsonl 仍含每一步记录，但分批写出；train_ms 与 cuda_timeline_ms 只在采样步有测量值。
host_submit_ms 是主机提交耗时，不可当成训练吞吐。
--stage smoke 保留每步审计和小规模强制分裂，仅用于测试。

## 验证与边界

测试与对照放在 output/ground_performance_20261027。
unit_validation.json：packed 图像/深度/alpha、模型及屏幕梯度、EAS 权重、融合 masked SSIM、
仿射变换梯度、缓存失效、晚出现车辆的 Adam 状态、参数替换等。
smoke_validation.json：静态和34车动态短训练、实际地面分裂、恢复、渲染。
trajectory_parity.json：与旧实现使用相同种子和视图，10步训练及验证指标接近。
benchmark_comparison.json：完整初始点数，GPU1顺序对照，100步训练中的21～100步计时。
初始化/编译/最终检查点写盘不计入该吞吐；只采样两张训练视图，不代表全序列训练时长。

不改变 CUDA rasterizer 源码、不引入混合精度或不受验证的图捕获。
位姿/几何约束与视觉模型不变，浮点运算顺序可能产生微小数值差异。
