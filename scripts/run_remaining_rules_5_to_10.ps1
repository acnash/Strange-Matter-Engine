param(
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$gpuPython = 'C:\Users\Anthony\anaconda3\envs\strange-matter-gpu\python.exe'
$reportPython = 'C:\Users\Anthony\anaconda3\envs\pythonProject1\python.exe'
$queueLog = Join-Path $Workspace 'results\challenge_aligned_v5_queue.log'
$rule4Summary = Join-Path $Workspace 'results\production_coupled_map_challenge_aligned_v5\study_summary.json'

Set-Location -LiteralPath $Workspace
"$(Get-Date -Format o) queue waiting for coupled_map" | Add-Content -LiteralPath $queueLog
while (-not (Test-Path -LiteralPath $rule4Summary)) {
    Start-Sleep -Seconds 30
}

$rules = @(
    'damped_symplectic',
    'fitzhugh_nagumo',
    'gray_scott',
    'kuramoto_sakaguchi',
    'conservative_graph_flux',
    'delayed_memory'
)

foreach ($rule in $rules) {
    $summary = Join-Path $Workspace "results\production_${rule}_challenge_aligned_v5\study_summary.json"
    if (Test-Path -LiteralPath $summary) {
        "$(Get-Date -Format o) $rule already complete; skipping" | Add-Content -LiteralPath $queueLog
        continue
    }
    "$(Get-Date -Format o) starting $rule" | Add-Content -LiteralPath $queueLog
    & $gpuPython 'scripts\run_production_transition_study.py' --rule $rule --device cuda --report-python $reportPython *>> $queueLog
    if ($LASTEXITCODE -ne 0) {
        "$(Get-Date -Format o) $rule failed with exit code $LASTEXITCODE; queue stopped" | Add-Content -LiteralPath $queueLog
        exit $LASTEXITCODE
    }
    "$(Get-Date -Format o) completed $rule" | Add-Content -LiteralPath $queueLog
}

"$(Get-Date -Format o) queue complete" | Add-Content -LiteralPath $queueLog
