param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$conda = Join-Path $ProjectRoot ".tools\miniforge3\Scripts\conda.exe"
$prefix = Join-Path $ProjectRoot ".conda\envs\dongfeng-registration"
$environmentFile = Join-Path $ProjectRoot "environment.yml"
$lomaSource = Join-Path $ProjectRoot ".tools\loma"

if (-not (Test-Path $conda)) {
    throw "Miniforge not found at $conda"
}

& $conda env create --prefix $prefix --file $environmentFile --yes
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not (Test-Path $lomaSource)) {
    throw "LoMa source not found at $lomaSource"
}

$python = Join-Path $prefix "python.exe"
& $python -m pip install --no-build-isolation --no-deps --editable $ProjectRoot
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m pip install --no-build-isolation --no-deps --editable $lomaSource
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Output "Environment created at $prefix"
