param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".conda\envs\dongfeng-registration\python.exe"

& $python -m dongfeng_registration.pose_quality_cli @Arguments
exit $LASTEXITCODE
