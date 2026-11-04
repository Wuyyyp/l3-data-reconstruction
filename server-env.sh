# Source from Bash. Relative paths in local.env are relative to this checkout.
export CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -f "$CODE_ROOT/local.env" ]]; then source "$CODE_ROOT/local.env"; fi
_l3_path() {
    local key="$1" default="$2" value
    value="${!key:-$default}"
    [[ "$value" == /* ]] || value="$CODE_ROOT/$value"
    printf -v "$key" '%s' "$value"
    export "$key"
}
_l3_path POSE_ROOT pose-repair
_l3_path DVP_ROOT dvp-openmvs
_l3_path MESH_ROOT vehicle-mesh
_l3_path STREETGS_ROOT streetgs
_l3_path POSE_PYTHON .envs/pose/bin/python
_l3_path DVP_PYTHON .envs/dvp/bin/python
_l3_path MESH_PYTHON .envs/mesh/bin/python
_l3_path GS_PYTHON .envs/streetgs/bin/python
_l3_path DVP_RUNTIME runtime/dvp
_l3_path DVP_MOGE_ROOT "$DVP_ROOT/vendor/MoGe"
_l3_path DVP_MOGE_WEIGHTS weights/moge-3-vitl/model.pt
_l3_path DVP_FULL_BIN "$DVP_ROOT/build-cpu-pipeline/APD"
_l3_path DVP_FAR_BIN "$DVP_ROOT/build-far-only/APD"
_l3_path DVP_DELIVERY_COLMAP tools/colmap/bin/colmap
_l3_path DVP_DELIVERY_VISION_PYTHON "$POSE_PYTHON"
_l3_path OPENMVS_BIN_DIR tools/openmvs/bin/OpenMVS
_l3_path OPENMVS_RUNTIME tools/openmvs/lib
_l3_path COLMAP_RUNTIME tools/colmap
_l3_path POSE_HF_HOME weights/huggingface
_l3_path POSE_TORCH_HOME weights/torch
_l3_path POSE_LOMA_SRC tools/loma/src
_l3_path ROAD_SEGFORMER_WEIGHTS weights/segformer-cityscapes
_l3_path DVP_MASK2FORMER_WEIGHTS weights/mask2former
_l3_path MESH_ASSET_ROOT assets/3d-assets
_l3_path SAM3_CHECKPOINT weights/sam3.pt
_l3_path SAM_CHECKPOINT weights/sam_vit_h_4b8939.pth
_l3_path GSPLAT_CUDA_SO "$STREETGS_ROOT/cache/improvedgs_static/torch_extensions/gsplat_cuda/gsplat_cuda.so"
for _l3_key in PREPARATION_DEPS GROUND_PREPROCESS_DEPS EGL_LIB_DIR MESH_ENV_ACTIVATE; do
    if [[ -n "${!_l3_key:-}" ]]; then _l3_path "$_l3_key" "${!_l3_key}"; fi
done
export DVP_COLMAP_BIN="$DVP_DELIVERY_COLMAP" DVP_BA_DONGFENG_ROOT="$POSE_ROOT" PYTHON_BIN="$DVP_PYTHON"
export PYTHONNOUSERSITE=1
unset _l3_key
unset -f _l3_path
