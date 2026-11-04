# CPU流水线与19次更新

主仓库已整合 sample_01147 的已测CPU流水线/P3实现。默认仍是 `DVP_CPU_PIPELINE=0`、`DVP_ITERATION_PROFILE=legacy`。原24次、legacy-sync、balanced、openmvs配置保留；`legacy19`为新的显式选项，限两个尺度。

## 编译与检查

```bash
cd /mnt/zhoukaixuan_workspace/code/DVP-MVS
cmake -S . -B build-cpu-pipeline -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu-pipeline --target APD CpuSelfTest IterationScheduleTest InitAudit -j2
(cd build-cpu-pipeline && ctest --output-on-failure)
```

使用独立build目录，原build/APD与build-original不覆盖。仅编译和CPU检查不会启动重建或融合。

## 使用主仓库程序

`SCENE`指新准备的DVP输入目录；它包含images/cams/pair.txt/metric_prior/blocks/metadata及静态edges/labels，不能指向已完成实验以免覆盖已有深度。GPU由调用者选定空闲卡，未自动占用其它卡。

```bash
cd /mnt/zhoukaixuan_workspace/code/DVP-MVS
SCENE=/path/to/new_prepared_scene
GPU=2
CUDA_VISIBLE_DEVICES="$GPU" DVP_CPU_PIPELINE=1 \
DVP_ITERATION_PROFILE=legacy19 DVP_SKIP_FUSION=1 \
DVP_EXEC_OPT=0 DVP_STATIC_CACHE=0 DVP_INPUT_OVERLAP=0 \
DVP_FAST_MODE=0 DVP_TEST_SEED=20261007 \
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 \
./build-cpu-pipeline/APD "$SCENE" 0
```

- `legacy19`：480非几何3＋几何3/3/3，960非几何3＋几何2/2，共7轮19更新，仍按原in-place深度顺序发布。
- `legacy`：480/960各非几何3＋几何3/3/3，共8轮24更新。与CPU开关可独立选择。
- 10源图由pair.txt决定，不由调度或CPU开关决定；沿用原20有序邻图的前10，后过滤应仍用保存的pair20_original.txt。程序不会擅自改pair或融合阈值。
- CPU流水线仅支持单worker的legacy/legacy19；不与cache、exec_opt、独立input_overlap实验开关叠加。关CPU流水线时走主仓库原串行CPU后处理，原mask V2、PNG/JPG兼容保持。
- DVP_TEST_SEED未设置时保留原GPU clock64随机和CPU默认rand；设置后用于同条件审计，但原CUDA非确定性仍存在，不能保证无损。

后台准备下一图静态输入、稳定动态深度和打包；当前源图发布后补读新深度；P3连通域、选图bits及写出与下一图GPU重叠。输入/输出队列限1，GPU参考图和CPU随机调用顺序保持。

## 既有批处理

run_batch_mvsnet_to_openmvs_ply.sh现在识别 `--dvp-profile legacy19` 和环境开关 `DVP_CPU_PIPELINE=1`，并要求一张GPU；resume请求记录新增CPU流水线/测试seed，以避免混用旧完成标记。该批处理保留原MoGe和后处理路线；不能将它直接称为sample_01147实验中定制MoGe后过滤＋1.5%融合的复现。

sample_01147的filter.py/fuse_depth1_5pct.py绑定具体输入、BA场景、mask及结果路径，仍保存在原实验code目录；主仓库已有通用转换/融合脚本，未将这些绑定路径的脚本硬编码为生产默认。

## 验证范围

主仓库整合执行编译、24/19及其它调度测试、13 CPU后处理边界自检、42真实初始化/升尺度CPU审计；P3源码与此前通过707图真实CPU输出核验的版本一致，不自动重跑DVP或点云融合。已测19次10源图主结果32.89分钟946866点、1.5%融合，24次20源图CPU版62.22分钟1319308点、1%融合；不同种子/GPU/背景及融合参数限制因果比较。减少更新不是无损优化。
