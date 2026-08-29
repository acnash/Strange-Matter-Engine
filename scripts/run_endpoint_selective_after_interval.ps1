$ErrorActionPreference = "Stop"
$controllerId = 96888
$projectRoot = "C:\Users\Anthony\OneDrive\Documents\ChatGPT\Strange Matter Engine"
$python = "C:\Users\Anthony\anaconda3\envs\pythonProject1\python.exe"
$stdout = Join-Path $projectRoot "results\production_endpoint_selective_ca_ensemble_v1.stdout.log"
$stderr = Join-Path $projectRoot "results\production_endpoint_selective_ca_ensemble_v1.stderr.log"

$controller = Get-Process -Id $controllerId -ErrorAction SilentlyContinue
if ($null -ne $controller) {
    Wait-Process -Id $controllerId
}

$process = Start-Process -FilePath $python `
    -ArgumentList "scripts\run_endpoint_selective_ca_ensemble.py" `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru
$process.WaitForExit()
if ($process.ExitCode -ne 0) {
    throw "Endpoint-selective CA ensemble failed with exit code $($process.ExitCode)"
}
