$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = 'C:\Users\Anthony\anaconda3\envs\strange-matter-gpu\python.exe'
$CampaignName = 'production_hybrid_genetic_structure_search_cyp3a4_v1'
$CampaignDirectory = Join-Path $ProjectRoot "results\$CampaignName"
New-Item -ItemType Directory -Force -Path $CampaignDirectory | Out-Null

$Process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList @('scripts/run_hybrid_genetic_structure_campaign.py') `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput (Join-Path $CampaignDirectory 'controller_stdout.log') `
    -RedirectStandardError (Join-Path $CampaignDirectory 'controller_stderr.log') `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    Campaign = $CampaignName
    ProcessId = $Process.Id
    ParallelCudaWorkers = 5
    PlannedRuns = 242
    OutputDirectory = $CampaignDirectory
}
