param(
    [string]$Python = "D:\DongFeng\Code\.conda\envs\dongfeng-registration\python.exe",
    [string]$Root = "D:\DongFeng"
)

$ErrorActionPreference = "Stop"
Set-Location $Root
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$RunId = Get-Date -Format "yyyyMMdd_HHmmss"
$TimingRoot = Join-Path $Root "outputs\full_pipeline_timing_$RunId"
$LogRoot = Join-Path $TimingRoot "logs"
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$StageRows = New-Object System.Collections.Generic.List[object]

function Get-TreeStats {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return [PSCustomObject]@{ FileCount = 0; DirectoryCount = 0; Bytes = 0 }
    }
    $items = Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue
    $files = @($items | Where-Object { -not $_.PSIsContainer })
    $dirs = @($items | Where-Object { $_.PSIsContainer })
    $bytes = 0
    foreach ($file in $files) { $bytes += [int64]$file.Length }
    [PSCustomObject]@{ FileCount = $files.Count; DirectoryCount = $dirs.Count; Bytes = $bytes }
}

function Invoke-Stage {
    param(
        [string]$Name,
        [string[]]$CommandArgs,
        [string]$PrimaryPath = ""
    )
    $safeName = $Name -replace '[^A-Za-z0-9_.-]', '_'
    $stdout = Join-Path $LogRoot "$safeName.stdout.log"
    $stderr = Join-Path $LogRoot "$safeName.stderr.log"
    $before = if ($PrimaryPath) { Get-TreeStats $PrimaryPath } else { [PSCustomObject]@{FileCount=0;DirectoryCount=0;Bytes=0} }
    $start = Get-Date
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @CommandArgs > $stdout 2> $stderr
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    $sw.Stop()
    $end = Get-Date
    $after = if ($PrimaryPath) { Get-TreeStats $PrimaryPath } else { [PSCustomObject]@{FileCount=0;DirectoryCount=0;Bytes=0} }
    $StageRows.Add([PSCustomObject]@{
        stage = $Name
        command = "$Python " + ($CommandArgs -join " ")
        start_time = $start.ToString("o")
        end_time = $end.ToString("o")
        elapsed_seconds = [Math]::Round($sw.Elapsed.TotalSeconds, 3)
        exit_code = $exitCode
        primary_path = $PrimaryPath
        files_before = $before.FileCount
        files_after = $after.FileCount
        files_delta = $after.FileCount - $before.FileCount
        dirs_before = $before.DirectoryCount
        dirs_after = $after.DirectoryCount
        bytes_before = $before.Bytes
        bytes_after = $after.Bytes
        bytes_delta = $after.Bytes - $before.Bytes
        stdout_log = $stdout
        stderr_log = $stderr
    }) | Out-Null
    $StageRows | Export-Csv (Join-Path $TimingRoot "stage_timing.csv") -NoTypeInformation -Encoding UTF8
    if ($exitCode -ne 0) {
        throw "Stage failed: $Name, exit code $exitCode. See $stderr"
    }
}

$Cameras = @(
    "center_camera_fov30",
    "left_front_camera",
    "right_front_camera",
    "rear_camera",
    "left_rear_camera",
    "right_rear_camera"
)

foreach ($camera in $Cameras) {
    Invoke-Stage -Name "semantic_$camera" -CommandArgs @(
        "-m", "dongfeng_registration.cli", "segment-dataset",
        "--config", "configs\t3.yaml",
        "--camera", $camera,
        "--start-frame", "0",
        "--frame-count", "600",
        "--cache-root", "D:\DongFeng\data_cache\semantic_masks",
        "--image-size", "2048",
        "--confidence", "0.25",
        "--batch-size", "24",
        "--preview-ratio", "0.1",
        "--device", "cuda:0",
        "--force"
    ) -PrimaryPath "D:\DongFeng\data_cache\semantic_masks"
}

foreach ($camera in $Cameras) {
    Invoke-Stage -Name "intra_match_$camera" -CommandArgs @(
        "-m", "dongfeng_registration.cli", "match-dataset",
        "--config", "configs\t3.yaml",
        "--camera", $camera,
        "--start-frame", "0",
        "--frame-count", "600",
        "--baseline-m", "2",
        "--baseline-m", "4",
        "--baseline-m", "6",
        "--cache-root", "D:\DongFeng\data_cache\semantic_masks",
        "--match-cache-root", "D:\DongFeng\data_cache\matches",
        "--mask-image-size", "2048",
        "--mask-confidence", "0.25"
    ) -PrimaryPath "D:\DongFeng\data_cache\matches"
}

$SameIndexPairs = @(
    @("center_camera_fov30", "left_front_camera"),
    @("center_camera_fov30", "right_front_camera"),
    @("rear_camera", "left_rear_camera"),
    @("rear_camera", "right_rear_camera")
)

foreach ($pair in $SameIndexPairs) {
    Invoke-Stage -Name "inter_same_$($pair[0])__$($pair[1])" -CommandArgs @(
        "-m", "dongfeng_registration.cli", "match-multicamera",
        "--config", "configs\t3.yaml",
        "--semantic-cache-root", "D:\DongFeng\data_cache\semantic_masks",
        "--match-cache-root", "D:\DongFeng\data_cache\matches",
        "--cameras", $pair[0],
        "--cameras", $pair[1],
        "--start-frame", "0",
        "--frame-count", "600",
        "--relation", "same_index",
        "--mask-image-size", "2048",
        "--mask-confidence", "0.25"
    ) -PrimaryPath "D:\DongFeng\data_cache\matches"
}

$TemporalBaselinePairs = @(
    @("center_camera_fov30", "left_front_camera"),
    @("center_camera_fov30", "right_front_camera"),
    @("rear_camera", "left_rear_camera"),
    @("rear_camera", "right_rear_camera")
)

foreach ($pair in $TemporalBaselinePairs) {
    Invoke-Stage -Name "inter_temporal_$($pair[0])__$($pair[1])" -CommandArgs @(
        "-m", "dongfeng_registration.cli", "match-multicamera",
        "--config", "configs\t3.yaml",
        "--semantic-cache-root", "D:\DongFeng\data_cache\semantic_masks",
        "--match-cache-root", "D:\DongFeng\data_cache\matches",
        "--cameras", $pair[0],
        "--cameras", $pair[1],
        "--start-frame", "0",
        "--frame-count", "600",
        "--relation", "temporal_overlap",
        "--mask-image-size", "2048",
        "--mask-confidence", "0.25"
    ) -PrimaryPath "D:\DongFeng\data_cache\matches"
}

Invoke-Stage -Name "inter_temporal_left_front__left_rear_8m" -CommandArgs @(
    "-m", "dongfeng_registration.cli", "match-multicamera",
    "--config", "configs\left_front_left_rear_temporal_8m_k1.yaml",
    "--semantic-cache-root", "D:\DongFeng\data_cache\semantic_masks",
    "--match-cache-root", "D:\DongFeng\data_cache\matches",
    "--cameras", "left_front_camera",
    "--cameras", "left_rear_camera",
    "--start-frame", "0",
    "--frame-count", "600",
    "--relation", "temporal_overlap",
    "--mask-image-size", "2048",
    "--mask-confidence", "0.25"
) -PrimaryPath "D:\DongFeng\data_cache\matches"

Invoke-Stage -Name "inter_temporal_right_front__right_rear_8m" -CommandArgs @(
    "-m", "dongfeng_registration.cli", "match-multicamera",
    "--config", "configs\right_front_right_rear_temporal_8m_k1.yaml",
    "--semantic-cache-root", "D:\DongFeng\data_cache\semantic_masks",
    "--match-cache-root", "D:\DongFeng\data_cache\matches",
    "--cameras", "right_front_camera",
    "--cameras", "right_rear_camera",
    "--start-frame", "0",
    "--frame-count", "600",
    "--relation", "temporal_overlap",
    "--mask-image-size", "2048",
    "--mask-confidence", "0.25"
) -PrimaryPath "D:\DongFeng\data_cache\matches"

Invoke-Stage -Name "joint_optimization" -CommandArgs @(
    "-m", "dongfeng_registration.cli", "optimize-multicamera-extrinsics",
    "--config", "configs\t3.yaml",
    "--match-cache-root", "D:\DongFeng\data_cache\matches\loma__mask2former",
    "--output", "D:\DongFeng\outputs\multicamera_joint_full_rerun",
    "--camera", "center_camera_fov30",
    "--camera", "left_front_camera",
    "--camera", "right_front_camera",
    "--camera", "rear_camera",
    "--camera", "left_rear_camera",
    "--camera", "right_rear_camera",
    "--intra-baseline-m", "2",
    "--intra-baseline-m", "4",
    "--intra-baseline-m", "6",
    "--inter-pair", "center_camera_fov30__left_front_camera",
    "--inter-pair", "center_camera_fov30__right_front_camera",
    "--inter-pair", "rear_camera__left_rear_camera",
    "--inter-pair", "rear_camera__right_rear_camera",
    "--inter-pair", "left_front_camera__left_rear_camera",
    "--inter-pair", "right_front_camera__right_rear_camera",
    "--inter-relation", "same_index",
    "--inter-relation", "temporal_overlap_12m",
    "--inter-relation", "temporal_overlap_7m",
    "--inter-relation", "temporal_overlap_8m",
    "--optimizer-method", "torch_sparse_graph",
    "--torch-device", "cuda:0",
    "--torch-learning-rate", "0.003",
    "--torch-lr-schedule", "step",
    "--torch-lr-step-size", "300",
    "--torch-lr-decay", "0.5",
    "--torch-lr-min-factor", "0.1",
    "--pair-sample-seed", "20260704",
    "--adaptive-weight-mode", "group_static",
    "--adaptive-weight-max", "2.0",
    "--max-nfev", "1500",
    "--translation-bound-m", "1.0",
    "--rotation-bound-deg", "10.0",
    "--fixed-translation-l2-weight", "50.0",
    "--fixed-rotation-l2-weight", "300.0",
    "--optimize-time-varying-disturbance",
    "--time-varying-disturbance-dof", "6",
    "--optimize-body-time-varying-disturbance",
    "--body-time-varying-disturbance-dof", "6",
    "--time-varying-rotation-bound-deg", "15.0",
    "--time-varying-translation-bound-m", "0.15",
    "--time-varying-translation-l2-weight", "200.0",
    "--time-varying-rotation-l2-weight", "5.0",
    "--time-varying-translation-smooth-l2-weight", "20.0",
    "--time-varying-rotation-smooth-l2-weight", "0.5"
) -PrimaryPath "D:\DongFeng\outputs\multicamera_joint_full_rerun"

$StageRows | Export-Csv (Join-Path $TimingRoot "stage_timing.csv") -NoTypeInformation -Encoding UTF8
Write-Output "TIMING_ROOT=$TimingRoot"


