$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/environment.ps1"
if (-not (Test-Path -LiteralPath $dqPython)) { throw 'Create a Python 3.12 venv in .venv first.' }
& $dqPython -m pip install 'torch==2.9.1' --index-url 'https://download.pytorch.org/whl/cu128' --progress-bar off
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }
& $dqPython -m pip install -r (Join-Path $dqRoot 'requirements.txt') --index-url 'https://pypi.org/simple' --progress-bar off
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $dqPython -m pip freeze | Set-Content -LiteralPath (Join-Path $dqRoot 'requirements-lock.txt') -Encoding utf8
