# DVP-MVS 安装和运行检查

检查日期：2026-09-05。上游提交：`5b2b31c02941cfa28808b4e37f06a87e8dc0f2d9`。

## 结论

原版能编译，但按 README 准备的 `images/ + cams/ + pair.txt` 不足以可靠运行。
本地已做运行修复，并保留原版可执行文件 `build-original/APD`；当前修复版为 `build/APD`。
这不是论文完整复现：作者没有随仓库提供离线单目深度和区域腐蚀/膨胀生成流程。

[issue #5 作者回复](https://github.com/ZhenlongYuan/DVP-MVS/issues/5#issuecomment-3536290291)
明确说明相关策略在离线 `.dmb` 中。作者提到的无 `.dmb` 降级说法，与当前提交里
仍强制读取 `dep/`、`sfm/` 和 `MVS4/` 的路径不一致。

## 主要 bug

| 问题 | 影响 | 本地处理 |
|---|---|---|
| 主循环 `i < round_num - 1` | 小于等于 800 的图跳过重建；大图缺全分辨率层 | 修正循环上限 |
| 缺少 dep/sfm 后访问空 rates | 段错误；原版大图测试已复现 | 无完整先验时明确采用随机 PatchMatch 初始化 |
| MVS4 标签无同尺寸赋值，且按 float 缩放 | 空标签或标签类型错误 | 无 MVS4 时使用图像生成标签，外部标签必须为 int32 |
| RANSAC 最佳采样点索引一直为 -1 | CUDA 数组越界，普通执行可能掩盖错误 | 保存获胜采样点索引；显存检查已复现原问题 |
| use_APD=false 时邻居映射未初始化 | cudaMemcpy 参数错误被忽略 | 初始化映射 |
| 方向向量把 z 写为 x、数组容量仅 20 | 可见性方向错误，多视图可能越界 | 修正分量和容量 |
| 无几何一致性时仍读取深度纹理指针 | 无效指针访问风险 | 条件读取 |
| 首轮先验没有从深度/世界法线转换成相机平面 | 初始匹配代价可能错误 | 补充转换 |
| 矩阵缩放 x/y 比例写反、同尺寸不赋值 | 非等比例缩放错误 | 修正 |
| 文件打开判断、未排序中位数、未初始化法线边界 | 缺输入崩溃或先验不稳定 | 增加基础检查、排序和初始化 |

更细的记录见 `diagnostics/README.md`；完整修改见 `diagnostics/runtime-fixes.patch`。
这些是本次检查覆盖的问题，不代表对整个科研代码做了完整正确性证明。

## 环境和重新编译

Ubuntu 20.04 / GCC 9.4 / CMake 3.16.3 / CUDA 11.6 / A800 sm_80。
已安装系统 C++ 依赖 OpenCV 4.2 和 Boost 1.71。
Python 测试脚本使用机器已有的 numpy 和 cv2，不需要模型权重。

```bash
cd /mnt/zhoukaixuan_workspace/code/DVP-MVS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
```

其他新机器需要先安装 `libopencv-dev libboost-filesystem-dev libboost-system-dev`，
并按显卡修改 CMakeLists.txt 中 `compute_80/sm_80`。本机无须更换 CUDA 或驱动。

## 使用

```bash
cd /mnt/zhoukaixuan_workspace/code/DVP-MVS
CUDA_VISIBLE_DEVICES=0 ./build/APD /你的场景目录 0
# 重做五视图、多尺度合成测试，并检查深度和点云：
./scripts/run_smoke_test.sh
```

场景目录要求：

- `images/00000000.jpg` 等同尺寸图像。
- `cams/00000000_cam.txt` 等相机参数，深度行需四个数。
- `pair.txt`，源视图也应作为参考视图列出以供融合。
- 可选 `dep/<id>.dmb + sfm/<id>.txt`；缺少完整先验时运行本地降级模式。
- 可选 `MVS4/<id>.dmb`，本地版本要求 `CV_32SC1` 标签；不提供时使用图像标签。

输出在 `<场景目录>/APD/`，融合点云为 `APD.ply`。
不同参数或输入之间建议使用新的场景目录，避免沿用旧的边缘/标签缓存。

## 验证范围

测试是已知相机、深度为 4 的纹理平面，五个视图。它验证编译、GPU 计算、
多尺度/几何一致性迭代、深度写出和点云融合。它不是 ETH3D 或 Tanks & Temples
真实数据集评测，也不能证明修复版达到论文精度。
单目输入分支另外使用人工构造的合法 dep/sfm 冒烟验证，未使用作者的离线先验。

旧 CUDA 11.6 检查器提示设备不支持，不能作为显存检查通过的证据。
随后单独下载解包 CUDA 12.9 Compute Sanitizer 到 diagnostics/tools/，没有改动系统 Toolkit。

## 实测结果

| 测试 | 结果 |
|---|---|
| 原版编译 | 成功 |
| 原版 320×240 五视图 | SIGABRT，跳过估计后融合触发 OpenCV 断言 |
| 原版 832×240 五视图 | SIGSEGV，缺失 dep/sfm 后空数组访问 |
| 本地修复 320×240 | 71,198 个融合点；CUDA 12.9 memcheck 0 errors |
| 最终修复 832×240 多尺度 | 192,183 个融合点；退出码 0；memcheck 0 errors |
| 人工构造 dep/sfm 先验分支 | 成功，75,366 个融合点；非作者离线数据 |

最终多尺度测试的五张深度图均为 240×832，所有值有限且为正；
各视图深度中位数为 3.9998–4.0005，真值为 4。GPU 随机初始化使重跑数值略有变化。

- 最终点云：`diagnostics/scene_verified_large/APD/APD.ply`
- 最终显存检查：`diagnostics/memcheck-large-final.log`
- 最终数值检查：`diagnostics/metrics-large-final.json`
- 原版失败日志：`diagnostics/original-scene_small.log`、`diagnostics/original-scene_large.log`
- 中途定位出的 RANSAC 越界日志：`diagnostics/memcheck-large.log`
- 编译日志：`diagnostics/build-original.log`、`diagnostics/build-fixed.log`

原始文件换行风格已保留；`git -c core.whitespace=cr-at-eol diff --check` 通过。
