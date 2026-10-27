# One-shot local setup for Windows PowerShell (README Steps 2-3).
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1            # dev install
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Runtime   # runtime-only install
# Every tool is run as "python -m ..." so it also works where Device Guard blocks .exe launchers.
param([switch]$Runtime)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
& $py -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 'Python 3.10+ required')"
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "==> Creating virtual environment in .venv"
& $py -m venv .venv
$venvPy = ".\.venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip

Write-Host "==> Installing CPU-only PyTorch"
& $venvPy -m pip install --index-url https://download.pytorch.org/whl/cpu torch

if ($Runtime) {
    & $venvPy -m pip install -r requirements.txt
} else {
    & $venvPy -m pip install -r requirements-dev.txt
    if (Test-Path .git) { & $venvPy -m pre_commit install }
}
& $venvPy -m pip install -e . --no-deps
if ($LASTEXITCODE -ne 0) { Write-Host "Install failed - see the messages above."; exit 1 }

if (-not (Test-Path .env)) { Copy-Item .env.example .env }
New-Item -ItemType Directory -Force -Path data, artifacts, logs | Out-Null

Write-Host ""
Write-Host "Done. Next steps (README Steps 4-8):"
Write-Host "  .venv\Scripts\activate"
Write-Host "  python -m pytest"
Write-Host "  python -m bookmind.pipeline.cli download"
Write-Host "  python -m bookmind.pipeline.cli all --classifier keyword"
Write-Host "  run.bat"
