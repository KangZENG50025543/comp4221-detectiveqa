param(
    [ValidateSet('doctor','audit','manual','pilot','prepare-api','tests')][string]$Action = 'doctor',
    [int]$Limit = 3,
    [string]$RunName = 'dev_retrieval_pilot'
)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot/scripts/environment.ps1"
switch ($Action) {
    'doctor' { & $dqPython "$PSScriptRoot/scripts/doctor.py" }
    'audit' { & $dqPython "$PSScriptRoot/scripts/audit_data.py" }
    'manual' { & $dqPython "$PSScriptRoot/scripts/build_manual_audit.py" }
    'pilot' { & $dqPython "$PSScriptRoot/scripts/run_retrieval_pilot.py" --limit $Limit --run-name $RunName }
    'prepare-api' { & $dqPython "$PSScriptRoot/scripts/prepare_api_inputs.py" --run-name $RunName }
    'tests' { & $dqPython -m unittest discover -s "$PSScriptRoot/tests" -v }
}
if ($LASTEXITCODE -ne 0) { throw "Research command failed: $Action" }
