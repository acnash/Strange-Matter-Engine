param([int]$CampaignProcessId)

$ErrorActionPreference = "Continue"
$projectDirectory = Split-Path -Parent $PSScriptRoot
$outputDirectory = Join-Path $projectDirectory "results\structure_dynamics_publication_v1"

while (Get-Process -Id $CampaignProcessId -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 30
}

$errorLog = Join-Path $outputDirectory "campaign.err.log"
if ((Test-Path $errorLog) -and (Get-Item $errorLog).Length -gt 0) {
    Add-Content -Path (Join-Path $outputDirectory "analysis.err.log") -Value "GPU campaign reported an error; statistical analysis was not started."
    exit 1
}

& "C:\Users\Anthony\anaconda3\envs\LSTM_conda_3_10\python.exe" `
    (Join-Path $PSScriptRoot "analyse_structure_dynamics_campaign.py") `
    1> (Join-Path $outputDirectory "analysis.log") `
    2> (Join-Path $outputDirectory "analysis.err.log")
