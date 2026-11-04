#!/usr/bin/env bash
# Batch pipeline: prepared multi-view data -> MoGeV3 -> DVP depth maps -> OpenMVS fused PLY.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../mvs_process" && pwd)"
REPO=$(cd -- "$SCRIPT_DIR/.." && pwd)
DVP_PREPARE="$SCRIPT_DIR/prepare_dvp_scene.py"
if [[ -n "${WYS_CASE_ROOT:-}" && -d "$WYS_CASE_ROOT/.moge_overlap_cache" ]]; then
  DVP_PREPARE="$(dirname "${BASH_SOURCE[0]}")/prepare_dvp_scene.py"
fi
DVP_FINALIZE="$SCRIPT_DIR/finalize_dvp_scene.py"
SPARSE_RANGE_DERIVER="$SCRIPT_DIR/derive_sparse_depth_ranges.py"
DMAP_CONVERTER="$SCRIPT_DIR/convert_dvp_to_openmvs_dmap.py"
DENSE_CONFIG="$SCRIPT_DIR/openmvs_dvp_depth_filter.cfg"

DMAP_PYTHON=${PYTHON_BIN:-python3}
MOGE_ROOT=${DVP_MOGE_ROOT:-$REPO/vendor/MoGe}
MOGE_VENV=${DVP_MOGE_VENV:-$MOGE_ROOT/.venv}
MOGE_PYTHON=${DVP_MOGE_PYTHON:-$MOGE_VENV/bin/python}
MOGE_PYTHONPATH="$MOGE_ROOT"
if [[ -d "$MOGE_VENV/lib/python3.11/site-packages" ]]; then
  MOGE_PYTHONPATH="$MOGE_VENV/lib/python3.11/site-packages:$MOGE_PYTHONPATH"
fi
MOGE_WEIGHTS=${DVP_MOGE_WEIGHTS:-$REPO/runtime/weights/moge-3-vitl/model.pt}
DVP_BIN=${DVP_BIN:-$REPO/build/APD}
OPENMVS_BIN_DIR=${OPENMVS_BIN_DIR:-$REPO/vendor/openMVS/install/bin/OpenMVS}
OPENMVS_DENSIFY=${OPENMVS_DENSIFY:-$OPENMVS_BIN_DIR/DensifyPointCloud}
OPENMVS_RUNTIME=${OPENMVS_RUNTIME:-$OPENMVS_BIN_DIR/../lib}
COLMAP_RUNTIME=${COLMAP_RUNTIME:-$REPO/../tools/colmap}
EXTRA_LD_LIBRARY_PATH=${EXTRA_LD_LIBRARY_PATH:-"$OPENMVS_RUNTIME:$COLMAP_RUNTIME/usr/lib:$COLMAP_RUNTIME/usr/lib/x86_64-linux-gnu:$COLMAP_RUNTIME/usr/lib/x86_64-linux-gnu/atlas:$COLMAP_RUNTIME/lib/x86_64-linux-gnu"}

GPU_IDS=${DVP_GPU_IDS:-0,1,2,3,4,5,6}
ITERATION_PROFILE=${DVP_ITERATION_PROFILE:-legacy}
OPENMVS_THREADS=${OPENMVS_THREADS:-8}
RESUME=0
CHECK_ONLY=0
COLMAP_BA=${DVP_COLMAP_BA:-0}
BA_MODE=${DVP_BA_MODE:-absolute_pose_prior}
BA_PYTHON=${DVP_BA_PYTHON:-$MOGE_PYTHON}
DONGFENG_ROOT=${DVP_BA_DONGFENG_ROOT:-$REPO/../pose-repair}
BA_COLMAP=${DVP_BA_COLMAP:-$DONGFENG_ROOT/tools/colmap_pose_prior/bin/colmap}
COLMAP_BIN=${DVP_COLMAP_BIN:-$REPO/../tools/colmap/bin/colmap}
BA_POSITION_SIGMA=${DVP_BA_POSITION_SIGMA_M:-0.1}
BA_ROTATION_SIGMA=${DVP_BA_ROTATION_SIGMA_DEG:-0.5}
BA_SCRIPT="$SCRIPT_DIR/run_colmap_ba.py"
OPENMVS_INTERFACE=${OPENMVS_INTERFACE:-$OPENMVS_BIN_DIR/InterfaceCOLMAP}
FROM_STAGE=1
TO_STAGE=4
BATCH_FILE=
SAMPLE_ROOT=
RESULT_ROOT=

usage() {
  cat <<'EOF'
Usage (one prepared sample):
  bash mvs_process/run_batch_mvsnet_to_openmvs_ply.sh \
    --sample-root /path/to/preprocessed/sample [--result-root DIR] [options]

Usage (batch TSV):
  bash mvs_process/run_batch_mvsnet_to_openmvs_ply.sh \
    --batch-file mvs_process/batch.example.tsv [options]

sample-root must be an output of run_preprocess_1_to_8.sh containing:
  03_converted/
  06_masks/openmvs_masks/
  07_sparse/triangulated_text/images.txt
  08_openmvs_input/global/scene.mvs
  08_openmvs_input/global/neighbors_diverse_*.txt
  audit/neighbors_diverse_*.tsv

The batch TSV columns are:
  sample_root<TAB>result_root
result_root is optional; outputs are written under sample_root when it is empty.

Options:
  --gpu-ids LIST         GPUs for MoGe/DVP (default: 0,1,2,3,4,5,6)
  --dvp-profile NAME    legacy (default), legacy-sync, balanced, openmvs, or legacy19 (experimental)
  --from-stage N         Start at stage N, 1-4 (default: 1)
  --to-stage N           Stop after stage N, 1-4 (default: 4)
  --no-colmap-ba        Skip COLMAP BA; use the original input poses (default)
  --colmap-ba           Enable BA explicitly (default: disabled)
  --ba-mode NAME        absolute_pose_prior (default), or standard
  --ba-position-sigma M Soft position prior standard deviation (default: 0.1 m)
  --ba-rotation-sigma D Soft rotation prior standard deviation (default: 0.5 deg)
  --resume               Reuse stages with completion markers
  --check                Validate dependencies only
  -h, --help             Show this help

Stages:
  0  COLMAP pose-prior BA, optimized camera files and OpenMVS scene (disabled by default; enable with --colmap-ba)
  1  sparse ranges + converted data -> MoGeV3 priors and DVP scene at 960x640
  2  DVP-MVS PatchMatch depth estimation (native DVP fusion disabled)
  3  DVP depth maps -> OpenMVS DMAP
  4  OpenMVS depth filter and dense-fuse -> PLY

This script does not scan or convert PKL files. BA uses the prepared sparse model; feature extraction is not repeated.
Depth bounds are derived per view from visible sparse points; no fixed 0-100 m clip is used.
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
require_exe() { [[ -x "$1" ]] || die "missing executable: $1"; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
mark_done() { printf 'PASS %s\n' "$(date -u '+%FT%TZ')" >"$1"; }
stage_done() { [[ "$RESUME" == 1 && -f "$1" ]]; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-file) BATCH_FILE=$2; shift 2 ;;
    --sample-root) SAMPLE_ROOT=$2; shift 2 ;;
    --result-root) RESULT_ROOT=$2; shift 2 ;;
    --gpu-ids) GPU_IDS=$2; shift 2 ;;
    --dvp-profile) ITERATION_PROFILE=$2; shift 2 ;;
    --from-stage) FROM_STAGE=$2; shift 2 ;;
    --to-stage) TO_STAGE=$2; shift 2 ;;
    --no-colmap-ba) COLMAP_BA=0; shift ;;
    --colmap-ba) COLMAP_BA=1; shift ;;
    --ba-mode) BA_MODE=$2; shift 2 ;;
    --ba-position-sigma) BA_POSITION_SIGMA=$2; shift 2 ;;
    --ba-rotation-sigma) BA_ROTATION_SIGMA=$2; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$FROM_STAGE" =~ ^[1-4]$ && "$TO_STAGE" =~ ^[1-4]$ ]] || die "stages must be integers in 1-4"
(( FROM_STAGE <= TO_STAGE )) || die "--from-stage must not exceed --to-stage"
case "$ITERATION_PROFILE" in legacy|legacy-sync|balanced|openmvs|legacy19) ;; *) die "invalid DVP profile: $ITERATION_PROFILE" ;; esac
IFS=',' read -r -a GPU_ARRAY <<<"$GPU_IDS"
[[ ${#GPU_ARRAY[@]} -gt 0 ]] || die "--gpu-ids is empty"
for gpu in "${GPU_ARRAY[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU ID: $gpu"; done

if [[ "${DVP_CPU_PIPELINE:-0}" == 1 ]]; then
  [[ ${#GPU_ARRAY[@]} == 1 ]] || die "CPU pipeline requires exactly one --gpu-ids entry"
  case "$ITERATION_PROFILE" in legacy|legacy19) ;; *) die "CPU pipeline requires legacy or legacy19" ;; esac
  [[ "${DVP_EXEC_OPT:-0}" == 0 && "${DVP_STATIC_CACHE:-0}" == 0 && "${DVP_INPUT_OVERLAP:-0}" == 0 ]] || die "Disable experimental cache/exec/input-overlap switches for CPU pipeline"
fi

[[ "$COLMAP_BA" == 0 || "$COLMAP_BA" == 1 ]] || die "DVP_COLMAP_BA must be 0 or 1"
case "$BA_MODE" in standard|absolute_pose_prior) ;; *) die "invalid BA mode: $BA_MODE" ;; esac
BA_ARGS=(--mode "$BA_MODE" --colmap "$COLMAP_BIN" --prior-colmap "$BA_COLMAP"
  --interface-colmap "$OPENMVS_INTERFACE" --dongfeng-root "$DONGFENG_ROOT"
  --position-sigma-m "$BA_POSITION_SIGMA" --rotation-sigma-deg "$BA_ROTATION_SIGMA")

preflight() {
  if [[ "$COLMAP_BA" == 1 ]]; then
    require_file "$BA_SCRIPT"; require_exe "$BA_PYTHON"; require_exe "$COLMAP_BIN"; require_exe "$OPENMVS_INTERFACE"
    "$BA_PYTHON" "$BA_SCRIPT" --check "${BA_ARGS[@]}" || die "COLMAP BA preflight failed"
  fi
  if [[ "$DMAP_PYTHON" == */* ]]; then require_exe "$DMAP_PYTHON"; else require_command "$DMAP_PYTHON"; fi
  require_file "$SPARSE_RANGE_DERIVER"
  "$DMAP_PYTHON" -c 'import numpy' || die "range-derivation Python lacks numpy"
  if (( FROM_STAGE <= 1 && TO_STAGE >= 1 )); then
    require_file "$DVP_PREPARE"; require_file "$DVP_FINALIZE"
    require_exe "$MOGE_PYTHON"; require_file "$MOGE_WEIGHTS"
    env PYTHONPATH="$MOGE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" "$MOGE_PYTHON" -c \
      'import cv2, numpy, torch; from moge.model import import_model_class_by_version; import_model_class_by_version("v3")' \
      || die "MoGe Python cannot import the v3 runtime"
  fi
  if (( FROM_STAGE <= 2 && TO_STAGE >= 2 )); then
    require_exe "$DVP_BIN"
    grep -aq 'DVP_SKIP_FUSION' "$DVP_BIN" || die "DVP executable lacks native-fusion skip support; rebuild build/APD from the current source"
    if [[ "$ITERATION_PROFILE" != legacy ]]; then
      grep -aq 'DVP_ITERATION_POLICY_V1' "$DVP_BIN" || die "DVP binary lacks iteration profiles; rebuild build/APD"
    fi
    if [[ "$ITERATION_PROFILE" == legacy19 ]]; then
      grep -aq 'legacy19' "$DVP_BIN" || die "DVP binary lacks legacy19; rebuild the integrated source"
    fi
    if [[ "${DVP_CPU_PIPELINE:-0}" == 1 ]]; then
      grep -aq 'DVP_CPU_PIPELINE' "$DVP_BIN" || die "DVP binary lacks CPU pipeline; rebuild the integrated source"
    fi
    if [[ "$ITERATION_PROFILE" == openmvs ]]; then
      grep -aq 'DVP_OPENMVS_SCHEDULE_V1' "$DVP_BIN" || die "DVP binary lacks OpenMVS schedule; rebuild build/APD"
    fi
    grep -aq 'DVP_DENSE_MASK_V2' "$DVP_BIN" || die "DVP executable lacks dense matching mask support; rebuild build/APD from the current source"
  fi
  if (( FROM_STAGE <= 4 && TO_STAGE >= 3 )); then
    require_file "$DMAP_CONVERTER"
    "$DMAP_PYTHON" -c 'import cv2, numpy' || die "DMAP Python lacks cv2/numpy"
  fi
  if (( FROM_STAGE <= 4 && TO_STAGE >= 4 )); then
    require_file "$DENSE_CONFIG"; require_exe "$OPENMVS_DENSIFY"
    local missing
    missing=$(LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ldd "$OPENMVS_DENSIFY" | awk '/not found/{print $1}' | paste -sd, -)
    [[ -z "$missing" ]] || die "unresolved OpenMVS libraries: $missing"
  fi
}

preflight
if [[ "$CHECK_ONLY" == 1 ]]; then
  cat <<EOF
PASS: MVSNet-to-PLY dependencies found
MoGe Python:    $MOGE_PYTHON
MoGe weights:   $MOGE_WEIGHTS
DVP executable: $DVP_BIN
OpenMVS:        $OPENMVS_DENSIFY
GPUs:           $GPU_IDS
DVP profile:    $ITERATION_PROFILE
COLMAP BA:      $COLMAP_BA ($BA_MODE)
depth range:    per-view sparse camera-Z min/max (0.9/1.1 margins)
resolution:     960x640 (one downsample level from 1920x1280)
fusion:         standard OpenMVS filtering and dense-fuse
EOF
  exit 0
fi

if [[ -n "$BATCH_FILE" ]]; then
  require_file "$BATCH_FILE"
  [[ -z "$SAMPLE_ROOT$RESULT_ROOT" ]] || die "use either --batch-file or --sample-root"
else
  [[ -n "$SAMPLE_ROOT" ]] || { usage >&2; die "--sample-root or --batch-file is required"; }
fi
export LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_moge_prepare() {
  local run_root=$1 source=$2 masks=$3 neighbors_tsv=$4 ranges=$5 state=$6 logs=$7
  local marker="$state/step_09_dvp_prepare.complete"
  if stage_done "$marker"; then echo "reuse stage 1 (MoGe/DVP scene)"; return; fi
  mkdir -p "$run_root" "$logs"
  local prepare_args=(--num-shards "${#GPU_ARRAY[@]}" --depth-ranges-json "$ranges" --unbounded-lidar-calibration)
  local finalize_args=(--depth-ranges-json "$ranges" --save-unbounded-prior)
  local shard status=0
  local -a pids=()
  for shard in "${!GPU_ARRAY[@]}"; do
    env CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$shard]}" PYTHONPATH="$MOGE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" \
      DVP_RUN_ROOT="$run_root" DVP_SOURCE="$source" DVP_OPENMVS_MASKS="$masks" DVP_MOGE_WEIGHTS="$MOGE_WEIGHTS" \
      "$MOGE_PYTHON" -u "$DVP_PREPARE" --shard "$shard" "${prepare_args[@]}" >"$logs/moge-shard-$shard.log" 2>&1 &
    pids+=("$!")
  done
  for shard in "${!pids[@]}"; do
    if ! wait "${pids[$shard]}"; then echo "MoGe shard $shard failed: $logs/moge-shard-$shard.log" >&2; status=1; fi
  done
  [[ "$status" == 0 ]] || return "$status"
  env PYTHONPATH="$MOGE_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}" DVP_RUN_ROOT="$run_root" DVP_SOURCE="$source" DVP_NEIGHBORS_TSV="$neighbors_tsv" \
    "$MOGE_PYTHON" -u "$DVP_FINALIZE" "${finalize_args[@]}" 2>&1 | tee "$logs/dvp-finalize.log"
  require_file "$run_root/manifest.json"; require_file "$run_root/scene/pair.txt"; mark_done "$marker"
}

verify_dvp_depths() {
  local run_root=$1 expected depth_count normal_count
  require_dir "$run_root/scene/APD"
  read -r expected <"$run_root/scene/pair.txt"
  [[ "$expected" =~ ^[0-9]+$ && "$expected" -gt 0 ]] || die "invalid image count in DVP pair.txt"
  depth_count=$(find "$run_root/scene/APD" -mindepth 2 -maxdepth 2 -type f -name 'depths.dmb' | wc -l)
  normal_count=$(find "$run_root/scene/APD" -mindepth 2 -maxdepth 2 -type f -name 'APD_normals.dmb' | wc -l)
  [[ "$depth_count" -eq "$expected" ]] || die "expected $expected DVP depth maps, found $depth_count"
  [[ "$normal_count" -eq "$expected" ]] || die "expected $expected DVP normal maps, found $normal_count"
}

run_dvp_depths() {
  local run_root=$1 state=$2 logs=$3 marker="$2/step_10_dvp_depths.complete"
  if stage_done "$marker"; then echo "reuse stage 2 (DVP-MVS depth maps)"; return; fi
  rm -rf "$run_root/scene/APD" "$run_root/scene/APD_parallel_barrier"
  mkdir -p "$run_root/scene/APD" "$logs"
  local worker status=0 count=${#GPU_ARRAY[@]}
  local -a pids=()
  for worker in "${!GPU_ARRAY[@]}"; do
    (cd "$REPO"; env CUDA_VISIBLE_DEVICES="${GPU_ARRAY[$worker]}" DVP_SKIP_FUSION=1 DVP_ITERATION_PROFILE="$ITERATION_PROFILE" \
      "$DVP_BIN" "$run_root/scene" 0 "$worker" "$count" \
      >"$logs/dvp-worker-$worker.log" 2>&1) &
    pids+=("$!")
  done
  for worker in "${!pids[@]}"; do
    if ! wait "${pids[$worker]}"; then echo "DVP worker $worker failed: $logs/dvp-worker-$worker.log" >&2; status=1; fi
  done
  [[ "$status" == 0 ]] || return "$status"
  cat "$logs"/dvp-worker-*.log >"$logs/dvp.log"
  verify_dvp_depths "$run_root"
  [[ ! -e "$run_root/scene/APD/APD.ply" ]] || die "DVP native fusion was not disabled"
  mark_done "$marker"
}

convert_dmaps() {
  local run_root=$1 source=$2 scene=$3 images_txt=$4 neighbors=$5 ranges=$6 fusion_root=$7 state=$8 logs=$9 reset=${10}
  local marker="$state/step_11_dmap_conversion.complete" dmaps="$fusion_root/dmaps"
  if [[ "$reset" == 0 ]] && stage_done "$marker"; then echo "reuse stage 3 (DMAP)"; return; fi
  [[ "$reset" == 1 ]] || rm -rf "$dmaps"
  mkdir -p "$dmaps" "$logs"
  "$DMAP_PYTHON" "$DMAP_CONVERTER" --run "$run_root" --source "$source" --scene "$scene" --images-txt "$images_txt" \
    --neighbors "$neighbors" --output "$dmaps" --depth-ranges-json "$ranges" 2>&1 | tee "$logs/dmap-convert.log"
  local count expected; count=$(find "$dmaps" -maxdepth 1 -name 'depth*.dmap' | wc -l); read -r expected <"$run_root/scene/pair.txt"
  [[ "$count" -eq "$expected" ]] || die "expected $expected DMAP files, found $count"
  [[ "$reset" == 1 ]] || mark_done "$marker"
}

run_openmvs() {
  local run_root=$1 source=$2 scene=$3 images_txt=$4 neighbors=$5 masks=$6 ranges=$7 fusion_root=$8 state=$9 logs=${10}
  local marker="$state/step_12_openmvs_fusion.complete"
  if stage_done "$marker"; then echo "reuse stage 4 (OpenMVS fusion)"; return; fi
  mkdir -p "$fusion_root" "$logs"
  convert_dmaps "$run_root" "$source" "$scene" "$images_txt" "$neighbors" "$ranges" "$fusion_root" "$state" "$logs" 1
  local output="$fusion_root/openmvs_filtered_fused.mvs"
  "$OPENMVS_DENSIFY" -w "$fusion_root/dmaps" -i "$scene" -o "$output" --dense-config-file "$DENSE_CONFIG" \
    --mask-path "$masks" --ignore-mask-label 255 --view-neighbors-file "$neighbors" --cuda-device -2 --max-threads "$OPENMVS_THREADS" \
    --resolution-level 1 --max-resolution 960 --min-resolution 640 --sub-resolution-levels 2 --number-views 20 \
    --number-views-fuse 2 --iters 4 --geometric-iters 0 --fusion-mode 0 --fusion-filter 2 \
    --fusion-depth-diff-threshold 0.010 --fusion-reprojection-threshold 2.5 --postprocess-dmaps 1 \
    --estimate-colors 2 --estimate-normals 2 --tower-mode 0 --estimate-roi 0 --crop-to-roi 0 \
    --filter-point-cloud 0 --remove-dmaps 0 --archive-type 2 -v 3 2>&1 | tee "$logs/openmvs-depth-filter-fusion.log"
  [[ -s "$fusion_root/openmvs_filtered_fused.ply" ]] || die "OpenMVS did not produce the fused PLY"
  mark_done "$marker"
}

process_sample() {
  local sample_root=$1 result_root=$2
  require_dir "$sample_root"; sample_root=$(cd -- "$sample_root" && pwd -P)
  [[ -n "$result_root" ]] || result_root="$sample_root"
  mkdir -p "$result_root"; result_root=$(cd -- "$result_root" && pwd -P)
  local sample_name=${sample_root##*/} source="$sample_root/03_converted" masks="$sample_root/06_masks/openmvs_masks"
  local scene="$sample_root/08_openmvs_input/global/scene.mvs" images_txt="$sample_root/07_sparse/triangulated_text/images.txt"
  local sparse_model="$sample_root/07_sparse/triangulated_text"
  local run_root="$result_root/09_dvp" fusion_root="$result_root/10_openmvs_fusion" state="$result_root/.state" logs="$result_root/logs"
  local ranges="$result_root/sparse_depth_ranges_openmvs.json"
  local -a neighbor_files=()
  mapfile -t neighbor_files < <(find "$sample_root/08_openmvs_input/global" -maxdepth 1 -name 'neighbors_diverse_pm*_top20.txt' -type f | sort)
  [[ ${#neighbor_files[@]} -eq 1 ]] || die "$sample_name: expected one top-20 neighbor file, found ${#neighbor_files[@]}"
  local neighbors=${neighbor_files[0]} neighbors_tsv="$sample_root/audit/$(basename "${neighbor_files[0]%.txt}").tsv"
  require_dir "$source"; require_dir "$masks"; require_file "$scene"; require_file "$images_txt"; require_file "$sparse_model/points3D.txt"; require_file "$neighbors_tsv"
  require_file "$sample_root/06_masks/dynamic_mask_audit.json"
  "$DMAP_PYTHON" -c 'import json,sys; sys.path.insert(0,sys.argv[3]); from M18proc.dynamic_policy import policy_version; r=json.load(open(sys.argv[1])); threshold=float(sys.argv[2]); raise SystemExit(0 if r.get("status") == "PASS" and r.get("image_count") == r.get("dynamic_mask_count") and r.get("dynamic_policy") == "disabled-preserve-all-objects" and r.get("dynamic_threshold_m") is None else 1)' \
    "$sample_root/06_masks/dynamic_mask_audit.json" "${DVP_DYNAMIC_THRESHOLD_M:-3}" "$REPO/data_preprocess" || die "$sample_name: dynamic masks do not match the requested map XY displacement threshold ${DVP_DYNAMIC_THRESHOLD_M:-3} m"
  mkdir -p "$state" "$logs"
  # A switch change must not reuse depths estimated with a different pose set.
  local old_request="$state/dvp_openmvs_request.txt"
  if [[ -f "$old_request" ]]; then
    local previous_ba=0
    grep -q '^colmap_ba=1$' "$old_request" && previous_ba=1
    [[ "$previous_ba" == "$COLMAP_BA" ]] || die "$sample_name: BA switch differs from saved results; use a fresh --result-root"
  fi
  local ba_digest=
  if [[ "$COLMAP_BA" == 1 ]]; then
    local image_anchor="$result_root/03_converted"
    if [[ ! -e "$image_anchor" && ! -L "$image_anchor" ]]; then ln -s -- "$source" "$image_anchor"; fi
    local -a ba_resume=()
    [[ "$RESUME" != 1 ]] || ba_resume=(--resume)
    (( FROM_STAGE <= 1 )) || ba_resume=(--resume-only)
    "$BA_PYTHON" "$BA_SCRIPT" --sample-root "$sample_root" --output "$result_root/colmap_ba" \
      --image-folder "$image_anchor/images" --neighbors-name "$(basename "$neighbors")" \
      "${BA_ARGS[@]}" "${ba_resume[@]}" 2>&1 | tee "$logs/colmap-ba.log"
    ba_digest=$("$DMAP_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$result_root/colmap_ba/request.json")
    source="$result_root/colmap_ba/converted"
    sparse_model="$result_root/colmap_ba/sparse_text"
    images_txt="$sparse_model/images.txt"
    scene="$result_root/colmap_ba/global/scene.mvs"
    neighbors="$result_root/colmap_ba/global/$(basename "$neighbors")"
    neighbors_tsv="$result_root/colmap_ba/global/neighbors.tsv"
    require_file "$scene"; require_file "$neighbors_tsv"
  fi
  if [[ "${DVP_USE_PRECOMPUTED_RANGES:-0}" == 1 ]]; then
    require_file "$ranges"
  else
    "$DMAP_PYTHON" "$SPARSE_RANGE_DERIVER" --model "$sparse_model" --output "$ranges" \
      2>&1 | tee "$logs/sparse-depth-ranges.log"
  fi
  local mask_profile=${DVP_CAMERA10_RIG_PROFILE:-$sample_root/06_masks/camera10_rig_profile.json}
  require_file "$mask_profile"
  local mask_digest sky_masks=${DVP_SKY_MASKS:-$sample_root/06_masks/safe_sky}
  local -a sky_args=()
  [[ ! -d "$sky_masks" ]] || sky_args=(--sky-masks "$sky_masks")
  mask_digest=$("$DMAP_PYTHON" "$SCRIPT_DIR/build_effective_masks.py" --images "$source/images" \
    --input-masks "$masks" --output "$result_root/06_masks/effective_masks" --rig-profile "$mask_profile" "${sky_args[@]}")
  masks="$result_root/06_masks/effective_masks"
  local range_digest
  range_digest=$("$DMAP_PYTHON" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$ranges")

  local request="$state/dvp_openmvs_request.txt" request_tmp="$state/dvp_openmvs_request.tmp"
  cat >"$request_tmp" <<EOF
pipeline_version=6
mask_policy=dense-mask-v2
masks_sha256=$mask_digest
source=$sample_root
output_size=960x640
depth_range=per_view_sparse_camera_z_minmax_0.9_1.1
depth_ranges_sha256=$range_digest
moge_lidar_calibration=unbounded
fusion_profile=openmvs_standard
EOF
  if [[ "$COLMAP_BA" == 1 ]]; then
    printf 'colmap_ba=1\ncolmap_ba_sha256=%s\n' "$ba_digest" >>"$request_tmp"
  fi
  if [[ "$ITERATION_PROFILE" != legacy ]]; then
    printf 'dvp_iteration_policy=1\ndvp_iteration_profile=%s\n' "$ITERATION_PROFILE" >>"$request_tmp"
  fi
  if [[ "${DVP_CPU_PIPELINE:-0}" != 0 ]]; then
    printf 'dvp_cpu_pipeline=%s\n' "$DVP_CPU_PIPELINE" >>"$request_tmp"
  fi
  if [[ -n "${DVP_TEST_SEED:-}" ]]; then
    printf 'dvp_test_seed=%s\n' "$DVP_TEST_SEED" >>"$request_tmp"
  fi
  if [[ -f "$request" ]] && ! cmp -s "$request" "$request_tmp"; then
    rm -f "$request_tmp"; die "$sample_name: pipeline settings differ; choose another --result-root or remove stages 1-4"
  fi
  mv "$request_tmp" "$request"

  printf '\n[%s] %s, stages %s-%s\n' "$(date '+%F %T')" "$sample_name" "$FROM_STAGE" "$TO_STAGE"
  if (( FROM_STAGE <= 1 && TO_STAGE >= 1 )); then run_moge_prepare "$run_root" "$source" "$masks" "$neighbors_tsv" "$ranges" "$state" "$logs"; fi
  if (( FROM_STAGE <= 2 && TO_STAGE >= 2 )); then require_file "$run_root/scene/pair.txt"; run_dvp_depths "$run_root" "$state" "$logs"; fi
  if (( FROM_STAGE <= 3 && TO_STAGE >= 3 )); then
    verify_dvp_depths "$run_root"; convert_dmaps "$run_root" "$source" "$scene" "$images_txt" "$neighbors" "$ranges" "$fusion_root" "$state" "$logs" 0
  fi
  if (( FROM_STAGE <= 4 && TO_STAGE >= 4 )); then
    # InterfaceCOLMAP stores image paths relative to the archive output root.
    # Keep that path valid when results are written outside sample_root.
    if [[ "$result_root" != "$sample_root" ]]; then
      local scene_source_link="$result_root/03_converted"
      if [[ -L "$scene_source_link" && "$COLMAP_BA" == 0 ]]; then
        ln -sfn -- "$source" "$scene_source_link"
      elif [[ ! -e "$scene_source_link" ]]; then
        ln -s -- "$source" "$scene_source_link"
      fi
      require_dir "$scene_source_link/images"
    fi
    require_file "$state/step_11_dmap_conversion.complete"; run_openmvs "$run_root" "$source" "$scene" "$images_txt" "$neighbors" "$masks" "$ranges" "$fusion_root" "$state" "$logs"
  fi
  echo "PASS: $sample_name completed through stage $TO_STAGE"
  [[ ! -s "$fusion_root/openmvs_filtered_fused.ply" ]] || echo "  OpenMVS point cloud: $fusion_root/openmvs_filtered_fused.ply"
}

declare -A SEEN_RESULTS=()
PROCESSED=0
if [[ -n "$BATCH_FILE" ]]; then
  while IFS=$'\t' read -r sample_root result_root extra; do
    [[ -z "${sample_root//[[:space:]]/}" || "$sample_root" == \#* || "$sample_root" == sample_root ]] && continue
    [[ -z "$extra" ]] || die "unexpected third TSV column for $sample_root"
    [[ -n "$result_root" ]] || result_root="$sample_root"
    [[ -z "${SEEN_RESULTS[$result_root]:-}" ]] || die "duplicate result_root: $result_root"
    SEEN_RESULTS[$result_root]=1
    process_sample "$sample_root" "$result_root"; PROCESSED=$((PROCESSED + 1))
  done <"$BATCH_FILE"
else
  process_sample "$SAMPLE_ROOT" "$RESULT_ROOT"; PROCESSED=1
fi
(( PROCESSED > 0 )) || die "batch file contains no samples"
printf '\nPASS: all requested samples completed\n'
