param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".conda\envs\dongfeng-registration\python.exe"

if (-not (Test-Path $python)) {
    throw "Environment missing. Run scripts\bootstrap.ps1 first."
}

& $python -m dongfeng_registration.semantic_cli @Arguments
exit $LASTEXITCODE
