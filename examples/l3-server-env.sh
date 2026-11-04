# Source in bash before running the examples on these existing L3 servers.
# Code paths resolve relative to this file so the four components can be moved.
export CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export POSE_ROOT="$CODE_ROOT/pose-repair" DVP_ROOT="$CODE_ROOT/dvp-openmvs"
export MESH_ROOT="$CODE_ROOT/vehicle-mesh" STREETGS_ROOT="$CODE_ROOT/streetgs"
: "${L3_DATA_ROOT:=/data/l3data-reconstruction-bingxing}"
export POSE_PYTHON="$L3_DATA_ROOT/Dongfeng_Pose-main/.venv-smoke/bin/python"
export DVP_RUNTIME="$L3_DATA_ROOT/DVP-MVS/runtime" DVP_PYTHON="$L3_DATA_ROOT/DVP-MVS/runtime/multinode/dvp/bin/python"
export DVP_FULL_BIN="$L3_DATA_ROOT/DVP-MVS/build-cpu-pipeline/APD" DVP_FAR_BIN="$L3_DATA_ROOT/DVP-MVS/build-far-only/APD"
export OPENMVS_BIN_DIR=/data/open-source-projects/openMVS/install/bin/OpenMVS OPENMVS_RUNTIME="$L3_DATA_ROOT/colmap+openmvs/openmvs_runtime"
export DVP_DELIVERY_COLMAP="$L3_DATA_ROOT/Dongfeng_Pose-main/tools/colmap_pose_prior/bin/colmap" DVP_DELIVERY_VISION_PYTHON="$DVP_RUNTIME/multinode/vision/bin/python" COLMAP_RUNTIME=/data/gsapro-main-test/.deps/colmap-runtime
export MESH_ASSET_ROOT="$L3_DATA_ROOT/testdata/3d-assets"
export MESH_PYTHON="$(bash -c 'source /data/envs/3dbbox-mesh/activate.sh >/dev/null && command -v python')"
export GS_PYTHON=/data/l3_data_test/street_gaussians/.venv/bin/python
export GSPLAT_CUDA_SO=/data/l3_data_test/street_gaussians/cache/improvedgs_static/torch_extensions/gsplat_cuda/gsplat_cuda.so
export GROUND_PREPROCESS_DEPS=/data/l3_data_test/street_gaussians/deps/ground_preprocess
export PREPARATION_DEPS="$L3_DATA_ROOT/new-pipeline-1031.before-export-1031/runtime/preparation_deps"
export STREETGS_EXTRA_PYTHONPATH="/data/l3_data_test/street_gaussians/deps/improvedgs_ground:/data/l3_data_test/street_gaussians/deps/improvedgs_dynamic:/data/l3_data_test/street_gaussians/deps/gsplat:$PREPARATION_DEPS"
export ROAD_SEGFORMER_WEIGHTS="${ROAD_SEGFORMER_WEIGHTS:-/root/wyp-streetgs-vla11-original/reference/code/roadmask_1031/segformer_cityscapes}"
export EGL_LIB_DIR="$L3_DATA_ROOT/new-pipeline-1031.before-export-1031/runtime/egl/root/usr/lib/x86_64-linux-gnu"
export PYTHONNOUSERSITE=1
export SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-$L3_DATA_ROOT/multiview_registration/sam3/sam3.pt}"
export SAM_CHECKPOINT="${SAM_CHECKPOINT:-/data/envs/.3dbbox-mesh-cache/sam_vit_h_4b8939.pth}"
export DVP_COLMAP_BIN="$DVP_DELIVERY_COLMAP"
export PYTHON_BIN="$DVP_PYTHON"
export MESH_ENV_ACTIVATE=/data/envs/3dbbox-mesh/activate.sh
export POSE_HF_HOME=/data/mnt/duanxuhao_bingixng/duanxuhao_bingxing/dataconvert_t3/pose_dynamic_compensation/code_copy/code/.models/huggingface
export POSE_TORCH_HOME=/data/mnt/duanxuhao_bingixng/duanxuhao_bingxing/dataconvert_t3/pose_dynamic_compensation/code_copy/code/.models/torch
export POSE_LOMA_SRC=/data/mnt/duanxuhao_bingixng/duanxuhao_bingxing/dataconvert_t3/pose_dynamic_compensation/code_copy/code/.tools/loma/src

export DVP_MOGE_ROOT="$DVP_ROOT/vendor/MoGe" DVP_MOGE_WEIGHTS="$DVP_RUNTIME/weights/moge-3-vitl/model.pt" DVP_BA_DONGFENG_ROOT="$POSE_ROOT"
export DVP_MASK2FORMER_WEIGHTS="$L3_DATA_ROOT/tem-test/colmap-wuda/runtime/mask2former"
export DVP_OLD_IMAGE_PREFIX=/mnt/duanxuhao_bingxing/ DVP_NEW_IMAGE_PREFIX=/data/mnt/duanxuhao_bingixng/duanxuhao_bingxing/
export L3_OLD_PATH_PREFIX=/mnt/nuplan L3_NEW_PATH_PREFIX=/data
