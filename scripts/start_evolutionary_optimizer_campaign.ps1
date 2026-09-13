$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = 'C:\Users\Anthony\anaconda3\envs\strange-matter-gpu\python.exe'
$CampaignDirectory = Join-Path $ProjectRoot 'results\production_es_optimizer_tuning_fixed_graph_ca_v1'
New-Item -ItemType Directory -Force -Path $CampaignDirectory | Out-Null

$Process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList @('scripts/run_evolutionary_optimizer_campaign.py') `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput (Join-Path $CampaignDirectory 'controller_stdout.log') `
    -RedirectStandardError (Join-Path $CampaignDirectory 'controller_stderr.log') `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    Campaign = 'production_es_optimizer_tuning_fixed_graph_ca_v1'
    ProcessId = $Process.Id
    ParallelCudaWorkers = 5
    OutputDirectory = $CampaignDirectory
}
