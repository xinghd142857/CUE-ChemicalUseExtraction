$ErrorActionPreference = "Stop"
$AppDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $AppDir

$VenvDir = Join-Path $AppDir ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m venv $VenvDir
} else {
    python -m venv $VenvDir
}
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $AppDir "requirements.txt")

if (!(Test-Path (Join-Path $AppDir "config.local.json"))) {
    Copy-Item (Join-Path $AppDir "config.example.json") (Join-Path $AppDir "config.local.json")
}

Write-Host "Setup complete."
Write-Host "App folder: $AppDir"
Write-Host "Edit config.local.json if you need model paths."
Write-Host "Start with: .\scripts\run_windows.ps1"
