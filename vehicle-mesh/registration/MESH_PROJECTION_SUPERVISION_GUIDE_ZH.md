# 远端 Mesh 投影监督：实现说明与排错指南

日期：2026-10-22  
依据：本目录现有代码的只读检查。本文不代表已验证远端实现，也不代表本地完整流水线已跑通。

## 1. 目标与推荐实现顺序

用一个车辆 mesh，在给定相机标定下渲染出轮廓，以图像中的目标观测约束车辆的位置、尺寸和朝向。

建议拆成两个独立问题：

1. **投影正确性**：给定确定的几何与位姿，是否投影到正确像素。
2. **监督正确性**：即使投影无误，观测 mask 是否真的能约束完整车辆。

不要在坐标、标定尚未确认时，靠增加优化轮次来解释误差。正确的投影也可能因为遮挡监督不当而优化到错误解。

远端已有投影版本时，优先补齐第 11 节的诊断输出，再按第 12 节逐级验收，无需先重写渲染器。

推荐最小版本：固定 mesh 拓扑和相机标定，每条 track 共享一组尺寸，每帧优化世界位置和 yaw，暂时固定 roll/pitch。先在完整观测上测试，再加入遮挡感知监督。

## 2. 本地代码实际采用的技术

| 环节 | 当前实现 |
|---|---|
| 读取模型 | trimesh，读取场景节点及其变换 |
| 渲染 | pyrender + OpenGL/EGL 离屏光栅化 |
| 投影模型 | IntrinsicsCamera，使用 fx、fy、cx、cy |
| 优化监督 | 渲染深度大于 0 得到二值轮廓，与 SAM3 mask 计算 IoU |
| 搜索 | SciPy Nelder–Mead、双向坐标扰动、前帧结果热启动 |
| 多视角 | 同一物体、同一帧共享一个位姿矩阵，各相机分别计算投影 |
| 加速 | 简化 proxy mesh、目标周围 ROI、缩小渲染分辨率 |
| 验收 | 在完整模型、原始分辨率上重新计算 IoU |

这条实现是**不可微渲染 + 无梯度搜索**。`depth > 0`、NumPy 和 SciPy 搜索不会把梯度传回 PyTorch 参数。若远端使用 Adam 训练 mesh 位姿，必须检查渲染器是否提供所需梯度，而不是仅让顶点张量设置 `requires_grad=True`。

核心文件：

- `utils/register_baseline.py`：模型规范化、相机变换、展示渲染。
- `utils/optimize_9dof.py`：轮廓渲染、ROI 内参、IoU、单帧搜索。
- `utils/optimize_grounded_temporal.py`：接地、世界坐标轨迹平滑。
- `utils/render_optimized.py`：用导出位姿渲染视频。

## 3. 首先固定坐标契约

统一使用齐次列向量，约定 `T_A_B` 表示“将 B 坐标变换到 A 坐标”：

```text
x_A = T_A_B @ x_B
T_A_C = T_A_B @ T_B_C
```

不要只写一个名为 `extrinsic` 的矩阵而不说明方向。建议远端明确保存：

| 字段 | 含义 |
|---|---|
| `T_world_ego[t]` | 当前帧 ego 到 world |
| `T_camcv_world[c,t]` | world 到 OpenCV camera |
| `T_world_object[t]` | 物体刚体坐标到 world，仅旋转和平移 |
| `size_lwh[track]` | 整条轨迹共享的长宽高，米 |
| `N_object_asset` | 模型资产到规范化物体坐标 |
| `K[c,t]` | 与实际图像分辨率匹配的内参 |

必须另外注明：各轴正方向、左右手系、yaw 正方向、角度单位、box 中心还是底面中心，以及时间戳单位。

### 3.1 本地的轴转换不能直接照搬到远端

本地写死了：

```python
C = np.array([[0, -1, 0, 0],
              [1,  0, 0, 0],
              [0,  0, 1, 0],
              [0,  0, 0, 1]])
```

它实现 `x_new=-y_old, y_new=x_old, z_new=z_old`，并对 yaw 加 π/2。它是绕 Z 的旋转，不能凭代码注释推断所有数据集的轴语义，也不能修复左右手系差异。

远端应从源数据定义推导转换矩阵。对单位坐标轴各变换一次，检查方向。纯旋转应满足 `R.T @ R ≈ I`、`det(R)≈+1`。镜像变换的行列式为负，若确实需要，必须同步处理三角形绕序和法线。

不要同时把轴变换烘焙进顶点，又对物体姿态重复应用同一个变换。

## 4. Mesh 资产如何规范化

### 4.1 场景节点变换不能丢

GLB/GLTF 可能包含车身、车轮、玻璃等多个节点，同一 geometry 也可能实例化多次。必须逐个节点取得其到场景基准的变换，再变换顶点。只遍历 `scene.geometry.values()` 可能丢失节点位置或重复实例。

本地做法：

```python
for node in scene.graph.nodes_geometry:
    G_asset_node, geometry_name = scene.graph[node]
    mesh = scene.geometry[geometry_name].copy()
    mesh.apply_transform(B_asset @ G_asset_node)
```

其中 `B_asset` 用来把资产轴转换为约定的物体轴。当前代码固定绕 X 加 90°，**不保证适用于任意 OBJ/GLB 或不同来源车型**。

### 4.2 长宽高与单位盒

建议约定物体局部 X 为车长方向、Y 为车宽方向、Z 为向上方向。先确认车头朝局部 +X 还是 -X，再计算轴对齐包围盒。

设所有已变换节点的总体边界为 `lo, hi`，则：

```python
asset_size = hi - lo
asset_center = (hi + lo) / 2
v_unit = (v_aligned - asset_center) / asset_size
v_object = diag([L, W, H]) @ v_unit
```

归一化后每个轴约在 `[-0.5, 0.5]`。当前方案对三个轴分别缩放，模型可以产生非等比形变。这能匹配不同尺寸，但也可能把错误车型拉成目标，需限制形变范围。

注意事项：

- 长宽高必须对应规范化后的轴，不能把资产原始 XYZ 默认当作 LWH。
- 车长方向正确不代表车头方向正确，轮廓对前后 180° 通常约束较弱。
- 全部节点只能共同归一化一次。逐个部件分别居中会把车轮和车身叠到一起。
- 模型含孤立点、天线或异常部件时，总 AABB 可能受污染，先检查实际几何。
- 规范化会丢掉资产原始米制尺寸。若依赖车型真实尺度，需要单独保留尺寸元数据或先验。
- 缓存应使用完整资产路径及处理参数，而非只有文件 stem。本地 `model_id` 缓存可能让不同目录下同名模型串用。

## 5. 投影与 OpenCV/OpenGL 转换

### 5.1 可直接核对的投影链

对物体规范化顶点 `v_unit`：

```text
v_world = T_world_object[t] @ S_track @ v_unit
v_camcv = T_camcv_world[c,t] @ v_world
u = fx * X/Z + cx
v = fy * Y/Z + cy
```

其中 `S_track=diag(L,W,H,1)`。`T_world_object` 是 SE(3)，不含尺寸，便于检查旋转正交性。

如果本来有相机到世界的位姿，应求逆得到 `T_camcv_world`。不能把 camera-to-world 当作 world-to-camera 直接乘顶点。

### 5.2 pyrender 需要相机在场景中的 pose

OpenCV 相机通常 X 向右、Y 向下、Z 向前。OpenGL 相机 X 向右、Y 向上、观察方向为 -Z。对于这里的约定：

```python
F = np.diag([1., -1., -1., 1.])
camera_pose_in_scene = np.linalg.inv(T_camcv_scene) @ F
```

这与本地 `inv(extrinsic) @ CV_FROM_GL` 一致。这里转换的是相机坐标约定，不是对模型额外翻转一次。pyrender 相机节点采用 OpenGL 坐标约定，见[官方文档](https://pyrender.readthedocs.io/_/downloads/en/latest/pdf/)。

最小渲染伪代码如下，`mesh_unit`、`scene` 和 `renderer` 由调用方创建：

```python
object_node = scene.add(mesh_unit, pose=T_scene_object @ S_track)
camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
                                  znear=0.1, zfar=300.0)
camera_node = scene.add(camera, pose=np.linalg.inv(T_camcv_scene) @ F)
depth = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
silhouette = depth > 0
# 每次更新物体 pose；用完移除旧相机节点，避免多个相机/节点累积。
```

`znear/zfar` 是本地设置示例，应按场景调整。靠近相机的三角形必须由渲染器裁剪，不能简单丢掉“某个顶点在相机后面”的整个目标。

### 5.3 内参、图像处理和畸变

本地仅传 fx、fy、cx、cy，没有显式应用畸变，也没有使用 K 的 skew 项。远端若使用原始畸变图像，不能只套针孔投影。

可选方案：图像和 mask 一起去畸变，使用对应的新 K；或者在一致的畸变模型下变换渲染结果。前者更便于排错。禁止只对 RGB 去畸变而保留原 mask。

缩放、裁剪、padding、旋转都必须同步到 K 和监督图像。对于先裁剪 `(x0,y0)` 再缩放到 `(rw,rh)`：

```python
sx, sy = rw / crop_width, rh / crop_height
A = np.array([[sx, 0, -sx*x0],
              [0, sy, -sy*y0],
              [0, 0, 1]])
K_roi = A @ K
```

这是本地使用的连续像素坐标变换。不同 resize/grid_sample 实现可能存在半像素中心差异，要统一像素中心和 `align_corners` 约定，再用合成点测试，不能到处补一个经验性的 0.5。

二值 mask 缩放用最近邻，概率/软轮廓采用与设计匹配的插值。图像 H/W、渲染 viewport 的 width/height 不要反写。

## 6. 监督应该比较什么

### 6.1 三种不同的 mask

建议分别命名并保存：

- `S_object`：目标 mesh 单独渲染的完整投影轮廓，仍受图像边界裁剪。
- `V_object`：考虑其他目标和场景遮挡后的可见投影。
- `M_obs`：SAM 或人工标注得到的观测 mask，通常是可见区域，也可能被污染。

单独渲染一个目标只能得到自身的可见表面轮廓，**不包含建筑或其他真实物体对它的遮挡**。当前优化路径主要比较 `S_object` 和 `M_obs`。

### 6.2 完整观测

目标完整可见、mask 可靠时，可用对称轮廓监督：

```text
L_iou = 1 - sum(S*M) / (sum(S) + sum(M) - sum(S*M) + eps)
```

可按需加轮廓距离、可信 2D 框边缘和尺寸先验。2D 框如果来自 visible box，不可默认当作 amodal/full box。

IoU 在目标完全不重叠时会出现平坦区。无梯度搜索可使用更宽的粗初始化和多候选，或者加入中心/距离变换项帮助靠近目标，再切换到轮廓细化。

### 6.3 遮挡观测

被建筑或其他车辆遮住时，完整投影通常大于可见 mask。直接对称 IoU 会惩罚被遮挡但本来合理的车身，诱导缩小尺寸或移动深度。

若没有可信遮挡深度，先使用单向包含约束：

```text
L_cover = sum(M_obs * (1-S_object)) / (sum(M_obs) + eps)
```

它要求可靠的可见像素被模型覆盖，但不强迫模型面积等于可见面积。**单独使用包含项会允许无限放大**，必须联合共享尺寸先验、完整帧监督、几何锚点及合理搜索范围。

更细的形式是定义可靠像素权重 Q，以及确认属于可见背景的区域 B：

```text
L_cover = sum(Q*M_obs*(1-S)) / (sum(Q*M_obs)+eps)
L_out   = sum(B*S) / (sum(B)+eps)
```

`B` 不是简单的 `1-M_obs`。遮挡区域和未知区域不能直接视为背景。只有确认没有目标且不受遮挡的区域才参与外溢惩罚。

有可信遮挡几何时，渲染场景/遮挡者的深度，按相同相机 Z 深度比较：

```text
V_object = S_object AND (D_object <= D_occluder + tolerance)
```

无深度背景应视为未知或无已知遮挡者，不能当作零距离。深度必须同单位、同相机、同时间、同“轴向 Z 或射线距离”定义。场景深度若已经含目标自身，要处理自遮挡/目标泄漏，否则会错误屏蔽或强化当前估计。

联合渲染多个车辆时可以处理这些模型之间的遮挡，但错误遮挡者位置也会传播错误。低置信遮挡关系使用软权重，不宜直接硬屏蔽所有像素。

### 6.4 截断与污染要单独处理

- 图像边缘截断：只在实际图像域评价，接触边缘的那条 2D 框边不再要求与完整框边相等。避免用可见高度强推完整高度和深度。
- 人工 crop 边界：不是物理遮挡。ROI 应容纳合理候选投影，否则模型伸出 crop 的部分可能逃避惩罚。
- mask 污染：包含前景设施/邻车时，包含项同样会拉偏模型，应清理、降低权重或停用该 mask。
- 窗户/玻璃等定义差异：先决定监督的是车身外轮廓、可见实体像素还是实例区域，渲染材质透明度和 SAM 标注定义必须一致。

VLM 可提供完整性/遮挡类型候选，但不能直接把其自报置信度当成标定概率。本项目此前 8B 试验存在 mask 污染漏报，需要人工核查与几何规则联合使用。

## 7. 参数化与优化：先限制不合理自由度

推荐参数：

```text
每条 track：d = d_prior * exp(delta_size)，所有帧/相机共享
每帧：world_center[t]，yaw[t]
可选：道路坡度提供的 roll/pitch、小范围接地修正
```

尺寸不能每帧独立自由变化。先选完整、分辨率高、遮挡少的关键帧确定车型/尺寸，再让部分可见帧辅助位置与时序约束。

Yaw 建议先做轨迹级粗候选选择，再小范围逐帧修正。初始轴差接近 90° 时，±6° 或 ±20° 的局部搜索无法恢复。前后 180° 不一定能靠轮廓区分，应保留前后歧义或引入车头外观、运动方向等证据。

本地 `decode_pose` 使用 `R0 @ R_delta @ diag(size)`，平移修正位于 canonical ego 坐标中。`R_delta` 作为局部旋转组合，而非对世界 yaw 随意加数。SciPy 小写 `xyz` 表示 extrinsic Euler 序列，不能与大写 `XYZ` 混用，见[官方说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)。优先用矩阵组合检查，而不是靠 Euler 名字猜方向。

### 7.1 如果继续采用 pyrender 无梯度搜索

1. 固定资产轴和相机，先搜索位置与 yaw。
2. 尺寸按轨迹共享，限制在合理车型范围内。
3. 多尺度搜索，在每级使用对应 K。
4. proxy/ROI 提议候选，原始分辨率完整 mesh 验收。
5. 最终验收使用与优化一致的遮挡感知目标，不能又切回偏置的完整轮廓 IoU。
6. 保存每项 loss 和每相机表现，不能只保存均值。

当前均值不下降验收仅用于 `optimize_9dof.py`；后续接地/平滑可能牺牲轮廓 IoU，因此不能宣称整个流水线保证 IoU 单调增加。

### 7.2 如果远端采用可微渲染

无需为了名称一致强行改成 pyrender。关键是对统一的投影、轮廓与参数定义做一致性测试。

- 使用该渲染器提供的轮廓/覆盖率梯度，不要在训练路径中立即 `>0` 阈值化。
- 不把优化参数转成 NumPy、不 `.detach()` 或 `.item()` 后重建矩阵。
- 检查一次 backward 后中心、yaw、共享尺寸的梯度是否非零且有限。
- 对少量参数做有限差分核对。硬轮廓是分段常数，有限差分需合理步长，不能要求每个亚像素扰动都改变像素。
- soft silhouette 的边缘宽度应随分辨率解释，过软可能用模糊边界而非真实几何降低 loss。
- 保存硬轮廓/原图分辨率指标，避免只比较软训练 loss。

## 8. 多视角、时间平滑与接地

同一时刻只维护一个 `T_world_object[t]`，所有相机通过标定计算投影。禁止每相机各拟合一套位置，最后再平均，这会掩盖不一致。

聚合时使用观测可靠度权重，并按实际有效视角归一化。小目标、污染 mask、严重遮挡不应和完整高分辨率目标同权。建议先逐帧聚合多视角，再按轨迹聚合，防止长轨迹或观测多的相机主导。

动态车辆需核对相机时间戳。不同步时应把世界轨迹插值到各相机曝光时间，不能假定同 frame ID 严格同时。若有 rolling shutter，快速运动误差应单独考虑，不能全部交给尺寸/yaw 拟合。

平滑在世界坐标中做，使用实际时间差：

```text
velocity[t] = (center[t+1]-center[t]) / dt[t]
accel[t] = 2*(velocity[t]-velocity[t-1]) / (dt[t]+dt[t-1])
L_motion = robust_penalty(accel)
L_yaw = robust_penalty(wrapped_yaw_rate 或其变化)
```

角度差先按周期处理。长缺帧处分段，不把两个片段强行连起来；静止与动态目标使用不同先验。不要直接惩罚动态车辆的全部速度为零。

道路平面写为 `n·X + b = 0`。应约束可信轮胎/底部支撑点接近局部道路，且车体不穿地。道路有坡度时不能强行用全局水平 Z。模型 AABB 中心不等于接地点。

本地接地高度来自输入 box 底面，只有有限修正，它不是独立道路估计。若 box 本身错误，接地参考也会错；若参考来自 GT3D，必须标为 oracle。最低顶点可能是排气管等部件，不能无条件当作轮胎接触点。

## 9. Mesh 监督不能独立解决尺度与深度

即使车型正确，单视角轮廓也可能允许“较小且较近 / 较大且较远”等多解。更多迭代不保证找到真实距离。

建议加入独立证据：可靠车型尺寸、同时刻跨相机几何、局部道路接地、可信度量深度。动态目标不能直接按静态物体用自车运动做跨帧三角化。

深度图通常描述可见表面，而框中心在车内。应渲染 mesh 表面深度与观测表面深度比较，不能直接用观测深度监督框中心。必须区分相机轴向 Z、射线距离和逆深度。

若只使用 mask，报告应写“轮廓拟合改善”，不能仅凭 IoU 上升推断 3D 几何改善。

## 10. 远端运行与工程注意事项

- EGL 环境变量在导入 OpenGL/pyrender、创建 context 之前设置。多进程每个 worker 独立创建 context。
- `CUDA_VISIBLE_DEVICES` 和 EGL device 的编号不一定一一对应，尤其容器或多 GPU 环境。用单 worker 实测设备与显存占用，再开启并发。
- 纯投影监督可以绕过 SAM3，直接消费已有 mask。分割环境与几何渲染环境可分开，避免模型依赖互相冲突。
- 原始资产只读，生成 proxy 另存或缓存在内存，不覆盖原 mesh。
- 缓存包含资产哈希、标定、预处理、参数边界与 loss 配置哈希。配置改变后不能只凭文件存在继续复用。
- JSON/PKL 保存时注明变换方向、单位、轴顺序、时间戳和参数版本。保留初始化位姿及来源。
- PKL 只加载可信来源，它并非安全的任意数据交换格式。
- 停止/恢复采用原子写入与完成状态，不能把中途写了一半的文件当成成功。

### 10.1 本地包迁移前已知问题

| 问题 | 影响与建议 |
|---|---|
| 接地阶段备份尚未生成的 `validation_report.json` | 首次运行会缺文件，备份前检查存在性或调整阶段顺序 |
| 多处固定 101 帧、10 FPS | 从 manifest/时间戳派生，不直接套到其他序列 |
| 校验截图写死 `videos/<camera>.mp4` | 按 manifest 的实际视频路径读取 |
| 模型缓存仅用 `model_id` | 用绝对路径/资产内容与变换配置构造 key |
| 完整 mesh 与可见 mask 对称 IoU | 先区分完整/遮挡/污染，再决定监督 |
| 动态目标逐帧尺寸仅作平滑 | 改成优化变量层面共享尺寸，不能只依赖后处理 |
| 接地与时序强约束可能改变最终 IoU | 分别报告约束前后结果，验收完整复合目标 |

这些问题尚未在本次文档工作中修复。远端无需原样移植整个包，可只对照投影与监督模块。

## 11. 必须保存的诊断输出

每个核查样本保存原图、观测 box、观测 mask、渲染完整轮廓、可见轮廓（若有）、渲染深度、前后投影、BEV。另输出：

```text
track_id, source_track_id, frame_id, camera_id, timestamp
asset_path/hash, asset axis transform, asset bounds, shared size_lwh
T_world_object_init/final, T_camcv_world, K, image_wh
crop_xywh, resize_wh, K_roi, distortion/rectification info
visibility label, mask quality, valid edges, visibility weights
loss_cover, loss_iou, loss_out, loss_size, loss_temporal, loss_ground
rendered area, observed area, near/far clipping, ROI clipping flags
optimizer status, iterations/evaluations, boundary-hit flags
```

展示图建议把三种区域分别着色：交集、仅渲染区域、仅观测区域。保留未做视觉修饰的原始渲染 mask。不能先把渲染轮廓裁成观测 mask 再宣称投影吻合。

本地展示恢复细长竖直前景缺口只是启发式合成效果，不是完整遮挡建模，不参与当前优化 IoU，不能作为优化改善证据。

## 12. 逐级验收与常见现象

### 一级：不优化，验证投影

1. 用标准立方体和已知相机，手算中心及角点像素，验证左右上下、近大远小。
2. 将同一组顶点按公式投影，并叠加渲染轮廓，检查内参、矩阵方向和半像素约定。
3. 对真实 mesh 显示局部 XYZ 与车头标记，确认轴和旋转方向。
4. 相同三角形、相同双面/背面剔除设置下，对比远端渲染器与基准版本的轮廓及深度。先允许栅格边缘的小差异，禁止用优化补偿明显系统偏移。
5. 全图结果与 ROI 结果还原到同一像素域后比较。候选伸出 ROI 时改用全图或扩大 ROI。

### 二级：合成可恢复实验

从已知位姿渲染一个目标 mask，故意扰动一个参数，检查是否能恢复。先测 XY/Z/yaw，最后测试共享尺寸。一次不要开放全部自由度。若使用相同渲染器自生成目标，这是实现自检，不是现实准确率验证。

### 三级：真实完整帧

选择清晰无遮挡帧，先固定尺寸，仅优化位置/yaw。保存每项 loss 与原图叠加。固定相机不动，避免相机参数与目标姿态互相补偿。

### 四级：人为遮挡消融

对合成完整 mask 遮掉一部分，比较对称 IoU 与包含约束。检查是否出现尺寸缩小、深度漂移；再在真实建筑遮挡、邻车遮挡上验证。评估遮挡标签、污染标签，不能只统计 JSON 格式通过率。

### 五级：轨迹与多视角

固定同一套关联和共同帧，对比无 mesh、普通 IoU、遮挡感知监督、加入独立几何锚点的版本。分别报告：深度/中心误差、尺寸误差、yaw 轴误差与车头方向误差、BEV/3D IoU、轨迹抖动、覆盖率。GT 仅用于离线评价；使用 GT 尺寸/yaw/位置参与优化的实验单独标为 oracle。

| 现象 | 优先检查 |
|---|---|
| 全部目标统一旋转 90° 或躺倒 | 资产轴、box LWH 顺序、yaw 偏置 |
| 左右镜像、旋转方向相反 | 左右手系、矩阵转置、行/列向量 |
| 模型不见或在相机背后 | 外参方向、OpenGL 转换、near/far、单位 |
| 中心基本对，远处误差随距离变化 | 内参分辨率、深度定义、标定与畸变 |
| 全图正常，crop 后偏移 | K_roi、padding、半像素、H/W |
| 遮挡时尺寸缩小或位置跳变 | visible/amodal 混用、对称面积惩罚、mask 污染 |
| 一个相机好，其他相机同时差 | 相机同步、外参方向、目标关联、各相机独立位姿 |
| loss 下降但实际框更差 | 多解、缺少度量锚点、形变过大或监督偏差 |
| Adam 不更新位姿 | 硬阈值/不可微路径、NumPy、detach、参数未注册 |
| 优化帧正确，导出视频不同 | 导出矩阵与渲染矩阵链不一致、归一化重复/遗漏 |

## 13. 可直接给远端实现者的任务说明

> 保留现有 mesh 投影实现，先统一 `T_A_B` 变换命名、资产轴、LWH、像素域和时间戳。用标准立方体及公式投影对照渲染结果，确保全图与 ROI 一致。同一 track 共享尺寸，同一帧跨相机共享世界位姿。完整观测采用轮廓监督，遮挡观测采用可靠可见像素的包含约束，未知遮挡区不当作负样本，污染 mask 单独降权或停用。尺寸/yaw 先验与独立几何证据抑制多解，世界坐标中按真实时间做轨迹平滑。优化、导出、展示共用同一变换链，输出分项 loss、标定与观测可靠性，先通过合成恢复实验，再做真实遮挡消融与固定关联的几何评价。

## 14. 参考

本地实现路径均相对本文件所在的 `mesh/` 目录。

- `utils/register_baseline.py`：`load_unit_model`、`camera_for`、`render_camera`。
- `utils/optimize_9dof.py`：`model_geometry`、`decode_pose`、`load_views`、`render_mask`、`view_iou`、`export_results`。
- `utils/optimize_grounded_temporal.py`：`context_for`、`world_pose`、`optimize_moving`。
- [pyrender 官方文档：相机与离屏渲染](https://pyrender.readthedocs.io/_/downloads/en/latest/pdf/)
- [pyrender 官方离屏渲染说明源码](https://github.com/mmatl/pyrender/blob/master/docs/source/examples/offscreen.rst)
- [trimesh SceneGraph 官方文档](https://trimesh.org/trimesh.scene.transforms.html)
- [SciPy Rotation.from_euler 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html)

本文中的优化方案和排错顺序是针对当前代码及已知遮挡问题提出的工程建议，具体阈值需在远端数据上验证，不是已完成的性能结论。
