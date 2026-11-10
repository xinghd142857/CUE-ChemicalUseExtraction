param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $RemainingArgs
)

$ErrorActionPreference = "Stop"
$AppDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $AppDir

$VenvPython = Join-Path $AppDir ".venv\Scripts\python.exe"
if (!(Test-Path $VenvPython)) {
    Write-Error "Missing .venv. Run setup_windows.ps1 first."
}

& $VenvPython -m chem_use_web @RemainingArgs
exit $LASTEXITCODE
