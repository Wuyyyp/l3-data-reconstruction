#!/usr/bin/env bash
# One-command L3 PKL -> OpenMVS scene.mvs preprocessing for one 101-frame segment.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
STEPS="$SCRIPT_DIR/steps"

MAIN_CLIPS=${MAIN_CLIPS:-/mnt/nuplan/l3_data_test/L3-data-pkl-paths/available_clips.txt}
PKL_PATH_LISTS=${PKL_PATH_LISTS:-"/mnt/nuplan/l3_data_test/L3-data-pkl-paths/available_pkl_paths_L3_V5data.txt /mnt/nuplan/l3_data_test/L3-data-pkl-paths/available_pkl_paths_L3_V5data_3.txt"}
DATA_ROOT=${DATA_ROOT:-/mnt/l3-labeled-data/prd_data/ALL}
COLMAP_BIN=${COLMAP_BIN:-/mnt/nuplan/l3_data_test/street_gaussians-main-local-v2/tem-test/tools/colmap-gpu/bin/colmap}
OPENMVS_BIN_DIR=${OPENMVS_BIN_DIR:-/mnt/nuplan/open-source-projects/openMVS/install/bin/OpenMVS}
SEGFORMER_CACHE=${SEGFORMER_CACHE:-/mnt/nuplan/l3data-reconstruction-bingxing/colmap+openmvs/pipeline_tools/segformer_b0_ade512}
PYTHON_DEPS=${PYTHON_DEPS:-/mnt/nuplan/l3data-reconstruction-bingxing/colmap+openmvs/pipeline_tools/python_deps}
OPENMVS_RUNTIME=${OPENMVS_RUNTIME:-/mnt/nuplan/l3data-reconstruction-bingxing/colmap+openmvs/openmvs_runtime}
COLMAP_RUNTIME=${COLMAP_RUNTIME:-/mnt/nuplan/gsapro-main-test/.deps/colmap-runtime}
PYTHON_BIN=${PYTHON_BIN:-python3}
SCAN_WORKERS=${SCAN_WORKERS:-8}
CONVERT_WORKERS=${CONVERT_WORKERS:-16}
COLMAP_GPUS=${COLMAP_GPUS:-0}
COLMAP_MATCH_GPUS=${COLMAP_MATCH_GPUS:-$COLMAP_GPUS}
SKY_DEVICE=${SKY_DEVICE:-cuda:0}
PAIR_RADIUS=${PAIR_RADIUS:-20}
DENSE_SOURCE_VIEWS=${DENSE_SOURCE_VIEWS:-20}
BUILD_SKY_MASKS=${BUILD_SKY_MASKS:-1}
ADD_CAMERA10_RIG_MASK=${ADD_CAMERA10_RIG_MASK:-1}
EXTRA_LD_LIBRARY_PATH=${EXTRA_LD_LIBRARY_PATH:-"$OPENMVS_RUNTIME:$COLMAP_RUNTIME/usr/lib:$COLMAP_RUNTIME/usr/lib/x86_64-linux-gnu:$COLMAP_RUNTIME/usr/lib/x86_64-linux-gnu/atlas:$COLMAP_RUNTIME/lib/x86_64-linux-gnu"}

export LD_LIBRARY_PATH="$EXTRA_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CLIP_NAME=
SIDE=
FRAME_START=
FRAME_END=
SAMPLE_NAME=
OUTPUT_ROOT="$SCRIPT_DIR/runs"
SEGMENTS_TSV=
FROM_STEP=1
TO_STEP=8
RESUME=0
CHECK_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  bash data_preprocess/run_preprocess_1_to_8.sh \
    --clip-name CLIP --side left|right --frame-start N --frame-end N \
    --output-root DIR [options]

Required segment selectors:
  --clip-name NAME       Clip name, for example clip_M18-2_07_20251202110510_DF
  --side SIDE            left or right
  --frame-start N        Inclusive original PKL frame index
  --frame-end N          Inclusive original PKL frame index; must span 101 frames

Options:
  --output-root DIR      Output root (default: data_preprocess/runs, ignored by Git)
  --sample-name NAME     Output directory name; derived from clip/range/side by default
  --segments-tsv FILE   Reuse a scanner TSV instead of recomputing step 1
  --from-step N          Start at step N, 1-8 (requires earlier outputs)
  --to-step N            Stop after step N, 1-8
  --resume               Reuse stages that have a completed marker
  --no-sky-mask          Create empty masks, retaining only the camera-10 rig mask
  --check                Check paths, Python imports and print the plan without processing data
  -h, --help             Show this message

Most machine-specific paths and GPU choices can be overridden with variables in
data_preprocess/config.example.env. OpenMVS dense PatchMatch is intentionally not
run here; step 8 stops after producing scene.mvs, masks and the neighbor file.
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }
require_exe() { [[ -x "$1" ]] || die "missing executable: $1"; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
require_runtime() {
  local missing
  missing=$(ldd "$1" | awk '/not found/{print $1}' | paste -sd, -)
  [[ -z "$missing" ]] || die "unresolved runtime libraries for $1: $missing"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clip-name) CLIP_NAME=$2; shift 2 ;;
    --side) SIDE=$2; shift 2 ;;
    --frame-start) FRAME_START=$2; shift 2 ;;
    --frame-end) FRAME_END=$2; shift 2 ;;
    --sample-name) SAMPLE_NAME=$2; shift 2 ;;
    --output-root) OUTPUT_ROOT=$2; shift 2 ;;
    --segments-tsv) SEGMENTS_TSV=$2; shift 2 ;;
    --from-step) FROM_STEP=$2; shift 2 ;;
    --to-step) TO_STEP=$2; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --no-sky-mask) BUILD_SKY_MASKS=0; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$FROM_STEP" =~ ^[1-8]$ && "$TO_STEP" =~ ^[1-8]$ && "$FROM_STEP" -le "$TO_STEP" ]] || \
  die "--from-step and --to-step must satisfy 1 <= from <= to <= 8"
if [[ "$CHECK_ONLY" != 1 ]]; then
  [[ -n "$CLIP_NAME" && -n "$SIDE" && -n "$FRAME_START" && -n "$FRAME_END" ]] || {
    usage >&2
    die "clip name, side, frame start and frame end are required"
  }
  [[ "$SIDE" == left || "$SIDE" == right ]] || die "--side must be left or right"
  [[ "$FRAME_START" =~ ^[0-9]+$ && "$FRAME_END" =~ ^[0-9]+$ ]] || die "frame indices must be non-negative integers"
  (( FRAME_END - FRAME_START + 1 == 101 )) || die "the selected segment must contain exactly 101 frames"
fi

preflight() {
  if [[ "$PYTHON_BIN" == */* ]]; then require_exe "$PYTHON_BIN"; else require_command "$PYTHON_BIN"; fi
  for helper in \
    01_collision_detection_v4.py 01_scan_original_clip_windows_v4.py \
    02_convert_m18_pkl.py 03_normalize_segment_0based.py \
    04_prepare_openmvs_input.py 05_prepare_image_lists_and_pairs.py \
    06_prepare_full_sky_masks.py 06_create_empty_masks.py 06_merge_dynamic_masks.py \
    06_add_camera10_rig_mask.py \
    07_camera_params.py 07_create_colmap_template.py 07_build_correct_fixed_model.py \
    07_audit_colmap_database.py 07_validate_fixed_poses.py 08_build_diverse_neighbors.py \
    select_segment.py; do
    require_file "$STEPS/$helper"
  done
  require_file "$SCRIPT_DIR/M18proc/__init__.py"
  if (( FROM_STEP <= 1 && TO_STEP >= 1 )) && [[ -z "$SEGMENTS_TSV" ]]; then
    require_file "$MAIN_CLIPS"
    read -r -a pkl_lists <<<"$PKL_PATH_LISTS"
    [[ ${#pkl_lists[@]} -gt 0 ]] || die "PKL_PATH_LISTS is empty"
    for path in "${pkl_lists[@]}"; do require_file "$path"; done
    require_dir "$DATA_ROOT/Result"
  elif [[ -n "$SEGMENTS_TSV" ]]; then
    require_file "$SEGMENTS_TSV"
  fi
  if (( TO_STEP >= 7 )); then
    require_exe "$COLMAP_BIN"
    require_runtime "$COLMAP_BIN"
  fi
  if (( TO_STEP >= 8 )); then
    require_exe "$OPENMVS_BIN_DIR/InterfaceCOLMAP"
    require_runtime "$OPENMVS_BIN_DIR/InterfaceCOLMAP"
  fi
  if (( FROM_STEP <= 6 && TO_STEP >= 6 && BUILD_SKY_MASKS == 1 )); then
    require_file "$SEGFORMER_CACHE/config.json"
    require_file "$SEGFORMER_CACHE/model.safetensors"
    require_dir "$PYTHON_DEPS/transformers"
  fi

  "$PYTHON_BIN" -c 'import cv2, numpy' || die "PYTHON_BIN lacks cv2/numpy"
  if (( FROM_STEP <= 2 && TO_STEP >= 2 )); then
    "$PYTHON_BIN" -c 'import open3d' || die "PYTHON_BIN lacks open3d required by the PKL converter"
  fi
  if (( FROM_STEP <= 6 && TO_STEP >= 6 && BUILD_SKY_MASKS == 1 )); then
    env PYTHONPATH="$PYTHON_DEPS${PYTHONPATH:+:$PYTHONPATH}" \
      HF_HOME="${TMPDIR:-/tmp}/dvp_preprocess_hf_check" \
      "$PYTHON_BIN" -c 'import PIL, torch, transformers' || \
      die "PYTHON_BIN lacks Pillow/torch/transformers required for sky masks"
  fi
}

preflight
if [[ "$CHECK_ONLY" == 1 ]]; then
  cat <<EOF
PASS: preprocessing dependencies found
steps: $FROM_STEP-$TO_STEP
python: $PYTHON_BIN
COLMAP: $COLMAP_BIN
InterfaceCOLMAP: $OPENMVS_BIN_DIR/InterfaceCOLMAP
sky masks: $BUILD_SKY_MASKS
default output root: $OUTPUT_ROOT
EOF
  exit 0
fi

[[ -n "$SAMPLE_NAME" ]] || SAMPLE_NAME="${CLIP_NAME}_f${FRAME_START}_${FRAME_END}_${SIDE}"
RUN_ROOT="$OUTPUT_ROOT/$SAMPLE_NAME"
STATE="$RUN_ROOT/.state"
LOGS="$RUN_ROOT/logs"
SCAN_DIR="$RUN_ROOT/01_scan"
ROW_JSON="$RUN_ROOT/selected_segment.json"
ABSOLUTE_DIR="$RUN_ROOT/02_converted_absolute"
CONVERTED_DIR="$RUN_ROOT/03_converted"
OPENMVS_INPUT="$RUN_ROOT/04_openmvs_input"
PAIR_WORK="$RUN_ROOT/05_pairs"
MASK_ROOT="$RUN_ROOT/06_masks"
SPARSE_ROOT="$RUN_ROOT/07_sparse"
MVS_ROOT="$RUN_ROOT/08_openmvs_input"
AUDIT="$RUN_ROOT/audit"

if [[ -e "$RUN_ROOT" && "$RESUME" != 1 ]]; then
  die "$RUN_ROOT already exists; use --resume or choose another output root/sample name"
fi
mkdir -p "$RUN_ROOT" "$STATE" "$LOGS" "$AUDIT"

stage_done() { [[ -f "$STATE/step_$1.complete" ]]; }
mark_done() { printf 'PASS %s\n' "$(date -u '+%FT%TZ')" >"$STATE/step_$1.complete"; }
stage_header() { printf '\n[%s] step %s: %s\n' "$(date '+%F %T')" "$1" "$2"; }
run_logged() {
  local log=$1
  shift
  "$@" 2>&1 | tee "$log"
}
json_field() {
  "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"
}

run_step_1() {
  stage_header 1 "scan PKLs and select the requested 101-frame window"
  if stage_done 1; then echo "reuse completed step 1"; return; fi
  if [[ -n "$SEGMENTS_TSV" ]]; then
    mkdir -p "$SCAN_DIR"
    cp "$SEGMENTS_TSV" "$SCAN_DIR/selected_101f_segments.tsv"
  else
    [[ ! -e "$SCAN_DIR" ]] || die "incomplete step 1 directory exists: $SCAN_DIR"
    read -r -a pkl_lists <<<"$PKL_PATH_LISTS"
    run_logged "$LOGS/step01_scan.log" "$PYTHON_BIN" "$STEPS/01_scan_original_clip_windows_v4.py" \
      --main-clips "$MAIN_CLIPS" --pkl-path-lists "${pkl_lists[@]}" \
      --current-pkl-root "$DATA_ROOT/Result" \
      --detector "$STEPS/01_collision_detection_v4.py" --output-dir "$SCAN_DIR" \
      --window-size 101 --workers "$SCAN_WORKERS" --clip-name "$CLIP_NAME"
  fi
  run_logged "$LOGS/step01_select.log" "$PYTHON_BIN" "$STEPS/select_segment.py" \
    --input "$SCAN_DIR/selected_101f_segments.tsv" --clip-name "$CLIP_NAME" \
    --side "$SIDE" --frame-start "$FRAME_START" --frame-end "$FRAME_END" \
    --data-root "$DATA_ROOT" --output "$ROW_JSON"
  mark_done 1
}

run_step_2() {
  stage_header 2 "convert the PKL range into absolute-index sensor files"
  if stage_done 2; then echo "reuse completed step 2"; return; fi
  require_file "$ROW_JSON"
  [[ ! -e "$ABSOLUTE_DIR" ]] || die "incomplete step 2 directory exists: $ABSOLUTE_DIR"
  local pkl_path
  pkl_path=$(json_field "$ROW_JSON" local_pkl_path)
  require_file "$pkl_path"
  run_logged "$LOGS/step02_convert.log" env PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$STEPS/02_convert_m18_pkl.py" \
      --input "$pkl_path" --output "$ABSOLUTE_DIR" --data_root "$DATA_ROOT" --downsample \
      --start_frame "$FRAME_START" --end_frame "$((FRAME_END + 1))" \
      --num_workers "$CONVERT_WORKERS" --proc_camera_idx 0 1 2 3 4 9 10
  mark_done 2
}

run_step_3() {
  stage_header 3 "normalize segment frame indices to 000000-000100"
  if stage_done 3; then echo "reuse completed step 3"; return; fi
  require_dir "$ABSOLUTE_DIR"
  [[ ! -e "$CONVERTED_DIR" ]] || die "incomplete step 3 directory exists: $CONVERTED_DIR"
  run_logged "$LOGS/step03_normalize.log" "$PYTHON_BIN" "$STEPS/03_normalize_segment_0based.py" \
    --src "$ABSOLUTE_DIR" --dst "$CONVERTED_DIR" --offset "$FRAME_START"
  mark_done 3
}

run_step_4() {
  stage_header 4 "create the fixed-pose input manifest"
  if stage_done 4; then echo "reuse completed step 4"; return; fi
  require_dir "$CONVERTED_DIR"
  [[ ! -e "$OPENMVS_INPUT" ]] || die "incomplete step 4 directory exists: $OPENMVS_INPUT"
  run_logged "$LOGS/step04_manifest.log" "$PYTHON_BIN" "$STEPS/04_prepare_openmvs_input.py" \
    --source "$CONVERTED_DIR" --output "$OPENMVS_INPUT"
  mark_done 4
}

run_step_5() {
  stage_header 5 "validate images and create COLMAP image/pair lists"
  if stage_done 5; then echo "reuse completed step 5"; return; fi
  require_file "$OPENMVS_INPUT/input_manifest.json"
  [[ ! -e "$PAIR_WORK" ]] || die "incomplete step 5 directory exists: $PAIR_WORK"
  run_logged "$LOGS/step05_pairs.log" "$PYTHON_BIN" "$STEPS/05_prepare_image_lists_and_pairs.py" \
    --source "$OPENMVS_INPUT" --output "$PAIR_WORK" --pair-radius "$PAIR_RADIUS"
  mark_done 5
}

run_step_6() {
  stage_header 6 "build OpenMVS ignore masks"
  if stage_done 6; then echo "reuse completed step 6"; return; fi
  require_dir "$OPENMVS_INPUT/images"
  [[ ! -e "$MASK_ROOT" ]] || die "incomplete step 6 directory exists: $MASK_ROOT"
  if [[ "$BUILD_SKY_MASKS" == 1 ]]; then
    mkdir -p "$RUN_ROOT/huggingface_cache"
    run_logged "$LOGS/step06_sky_masks.log" env \
      PYTHONPATH="$PYTHON_DEPS${PYTHONPATH:+:$PYTHONPATH}" HF_HOME="$RUN_ROOT/huggingface_cache" \
      "$PYTHON_BIN" "$STEPS/06_prepare_full_sky_masks.py" \
      --images "$OPENMVS_INPUT/images" --output "$MASK_ROOT" --cache "$SEGFORMER_CACHE" \
      --device "$SKY_DEVICE" --batch-size 8 --sky-prob 0.60 --sky-margin 0.25 --erosion-px 5
  else
    run_logged "$LOGS/step06_empty_masks.log" "$PYTHON_BIN" "$STEPS/06_create_empty_masks.py" \
      --images "$OPENMVS_INPUT/images" --output "$MASK_ROOT" --frame-count 101
  fi
  if [[ -d "$OPENMVS_INPUT/dynamic_mask" ]]; then
    run_logged "$LOGS/step06_dynamic_masks.log" "$PYTHON_BIN" "$STEPS/06_merge_dynamic_masks.py" \
      --images "$OPENMVS_INPUT/images" --dynamic-masks "$OPENMVS_INPUT/dynamic_mask" \
      --mask-root "$MASK_ROOT"
  fi
  if [[ "$ADD_CAMERA10_RIG_MASK" == 1 ]]; then
    run_logged "$LOGS/step06_camera10_mask.log" "$PYTHON_BIN" "$STEPS/06_add_camera10_rig_mask.py" \
      --images "$OPENMVS_INPUT/images" --mask-root "$MASK_ROOT" --frame-count 101
  fi
  local mask_count
  mask_count=$(find "$MASK_ROOT/openmvs_masks" -maxdepth 1 -name '*.mask.png' | wc -l)
  [[ "$mask_count" -eq 707 ]] || die "expected 707 masks, found $mask_count"
  mark_done 6
}

run_step_7() {
  stage_header 7 "extract/match COLMAP features and triangulate with fixed poses"
  if stage_done 7; then echo "reuse completed step 7"; return; fi
  require_file "$PAIR_WORK/colmap_match_pairs.txt"
  [[ ! -e "$SPARSE_ROOT" ]] || die "incomplete step 7 directory exists: $SPARSE_ROOT"
  local database="$SPARSE_ROOT/database.db"
  local template="$SPARSE_ROOT/database_template"
  local known="$SPARSE_ROOT/known_pose_input"
  local triangulated="$SPARSE_ROOT/triangulated_binary"
  local triangulated_text="$SPARSE_ROOT/triangulated_text"
  mkdir -p "$SPARSE_ROOT" "$template" "$known" "$triangulated" "$triangulated_text"
  run_logged "$LOGS/step07_database.log" "$COLMAP_BIN" database_creator --database_path "$database"

  local cameras=(00 01 02 03 04 09 10)
  local gpu_values
  IFS=',' read -r -a gpu_values <<<"$COLMAP_GPUS"
  if [[ ${#gpu_values[@]} -eq 1 ]]; then
    gpu_values=("${gpu_values[0]}" "${gpu_values[0]}" "${gpu_values[0]}" "${gpu_values[0]}" "${gpu_values[0]}" "${gpu_values[0]}" "${gpu_values[0]}")
  fi
  [[ ${#gpu_values[@]} -eq 7 ]] || die "COLMAP_GPUS must contain one or seven comma-separated GPU IDs"
  local index camera gpu params
  for index in "${!cameras[@]}"; do
    camera=${cameras[$index]}
    gpu=${gpu_values[$index]}
    params=$("$PYTHON_BIN" "$STEPS/07_camera_params.py" \
      --manifest "$OPENMVS_INPUT/input_manifest.json" --camera "$camera")
    run_logged "$LOGS/step07_feature_camera${camera}.log" "$COLMAP_BIN" feature_extractor \
      --database_path "$database" --image_path "$OPENMVS_INPUT/images" \
      --image_list_path "$PAIR_WORK/images_camera_${camera}.txt" \
      --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --ImageReader.camera_params "$params" \
      --SiftExtraction.use_gpu 1 --SiftExtraction.gpu_index "$gpu" \
      --SiftExtraction.max_image_size 1600 --SiftExtraction.max_num_features 8192
  done
  run_logged "$LOGS/step07_matching.log" "$COLMAP_BIN" matches_importer \
    --database_path "$database" --match_list_path "$PAIR_WORK/colmap_match_pairs.txt" --match_type pairs \
    --SiftMatching.use_gpu 1 --SiftMatching.gpu_index "$COLMAP_MATCH_GPUS" \
    --SiftMatching.guided_matching 1 --SiftMatching.max_error 4
  run_logged "$LOGS/step07_database_audit.log" "$PYTHON_BIN" "$STEPS/07_audit_colmap_database.py" \
    --database "$database" --pairs "$PAIR_WORK/colmap_match_pairs.txt" \
    --expected-images 707 --output "$AUDIT/matching_inventory.json"
  run_logged "$LOGS/step07_template.log" "$PYTHON_BIN" "$STEPS/07_create_colmap_template.py" \
    --database "$database" --manifest "$OPENMVS_INPUT/input_manifest.json" --output "$template"
  run_logged "$LOGS/step07_fixed_model.log" "$PYTHON_BIN" "$STEPS/07_build_correct_fixed_model.py" \
    --manifest "$OPENMVS_INPUT/input_manifest.json" --extrinsics "$OPENMVS_INPUT/extrinsics" \
    --template-model "$template" --output "$known" --audit "$AUDIT/corrected_pose_audit.json"
  run_logged "$LOGS/step07_triangulate.log" "$COLMAP_BIN" point_triangulator \
    --database_path "$database" --image_path "$OPENMVS_INPUT/images" \
    --input_path "$known" --output_path "$triangulated" --Mapper.fix_existing_images 1 \
    --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0 \
    --Mapper.tri_ignore_two_view_tracks 0 --Mapper.filter_max_reproj_error 4 \
    --Mapper.tri_merge_max_reproj_error 4 --Mapper.tri_complete_max_reproj_error 4 \
    --Mapper.filter_min_tri_angle 0.2
  run_logged "$LOGS/step07_model_text.log" "$COLMAP_BIN" model_converter \
    --input_path "$triangulated" --output_path "$triangulated_text" --output_type TXT
  run_logged "$LOGS/step07_pose_validation.log" "$PYTHON_BIN" "$STEPS/07_validate_fixed_poses.py" \
    --known-images "$known/images.txt" --triangulated-images "$triangulated/images.bin" \
    --output "$AUDIT/fixed_pose_validation.json"
  mark_done 7
}

run_step_8() {
  stage_header 8 "convert the COLMAP model to OpenMVS scene.mvs and build dense neighbors"
  if stage_done 8; then echo "reuse completed step 8"; return; fi
  require_file "$SPARSE_ROOT/triangulated_binary/images.bin"
  [[ ! -e "$MVS_ROOT" ]] || die "incomplete step 8 directory exists: $MVS_ROOT"
  local sparse_input="$MVS_ROOT/colmap_input/sparse"
  local global="$MVS_ROOT/global"
  mkdir -p "$sparse_input" "$global"
  local name
  for name in cameras.bin images.bin points3D.bin; do
    ln -s "$SPARSE_ROOT/triangulated_binary/$name" "$sparse_input/$name"
  done
  run_logged "$LOGS/step08_interface_colmap.log" \
    "$OPENMVS_BIN_DIR/InterfaceCOLMAP" -w "$global" -i "$MVS_ROOT/colmap_input" -o scene.mvs \
      --image-folder "$OPENMVS_INPUT/images" --binary 1 --archive-type 2 -v 2
  local neighbors="$global/neighbors_diverse_pm${PAIR_RADIUS}_top${DENSE_SOURCE_VIEWS}.txt"
  run_logged "$LOGS/step08_neighbors.log" "$PYTHON_BIN" "$STEPS/08_build_diverse_neighbors.py" \
    --scene "$global/scene.mvs" --images-txt "$SPARSE_ROOT/triangulated_text/images.txt" \
    --output "$neighbors" --audit "$AUDIT/neighbors_diverse_pm${PAIR_RADIUS}_top${DENSE_SOURCE_VIEWS}.json" \
    --radius "$PAIR_RADIUS" --sources "$DENSE_SOURCE_VIEWS" --cross-camera-quota 6
  require_file "$global/scene.mvs"
  require_file "$neighbors"
  mark_done 8
  cat <<EOF

OpenMVS preprocessing complete:
  scene:     $global/scene.mvs
  images:    $OPENMVS_INPUT/images
  masks:     $MASK_ROOT/openmvs_masks
  neighbors: $neighbors
EOF
}

for step in $(seq "$FROM_STEP" "$TO_STEP"); do
  "run_step_$step"
done

printf '\nPASS: steps %s-%s completed for %s\n' "$FROM_STEP" "$TO_STEP" "$SAMPLE_NAME"
