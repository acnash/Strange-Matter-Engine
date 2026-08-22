$ErrorActionPreference = 'Stop'

$python = 'C:\Users\Anthony\anaconda3\envs\strange-matter-gpu\python.exe'
$runner = Join-Path $PSScriptRoot 'run_graph_ca_visual_prototype.py'
$root = Split-Path $PSScriptRoot -Parent

$configs = @(
    @{ Id='01'; CaLr='0.0005'; CaL2='0.00001'; Ridge='0.0010'; Clip='1.0' },
    @{ Id='02'; CaLr='0.0010'; CaL2='0.00001'; Ridge='0.0010'; Clip='1.0' },
    @{ Id='03'; CaLr='0.0020'; CaL2='0.00001'; Ridge='0.0010'; Clip='1.0' },
    @{ Id='04'; CaLr='0.0010'; CaL2='0.00001'; Ridge='0.0003'; Clip='1.0' },
    @{ Id='05'; CaLr='0.0010'; CaL2='0.00010'; Ridge='0.0030'; Clip='1.0' },
    @{ Id='06'; CaLr='0.0010'; CaL2='0.00001'; Ridge='0.0010'; Clip='2.0' }
)

$started = Get-Date
foreach ($config in $configs) {
    $env:SME_CA_RULE = 'inertial_reaction_diffusion'
    $env:SME_GENERATIONS = '500'
    $env:SME_RUN_NAME = "gpu_500_timing_tuning_$($config.Id)"
    $env:SME_DEVICE = 'cuda'
    $env:SME_CA_LR = $config.CaLr
    $env:SME_CA_L2 = $config.CaL2
    $env:SME_RIDGE = $config.Ridge
    $env:SME_GRAD_CLIP = $config.Clip
    $env:SME_MAX_EPOCHS = '3'
    $env:SME_PATIENCE = '2'
    $env:SME_MIN_DELTA = '0.003'
    $env:SME_TUNING_ONLY = '1'
    $env:SME_TUNING_FIT_MOLECULES = '600'
    $env:SME_TUNING_VAL_MOLECULES = '200'
    & $python $runner train
    if ($LASTEXITCODE -ne 0) { throw "Tuning configuration $($config.Id) failed" }
}

$rows = foreach ($config in $configs) {
    $path = Join-Path $root "results\gpu_500_timing_tuning_$($config.Id)\metrics.json"
    $metric = Get-Content -Raw $path | ConvertFrom-Json
    [pscustomobject]@{
        id = $config.Id
        validation_rmse = $metric.restored_validation_rmse
        epochs = $metric.epochs_run
        training_seconds = $metric.training_seconds
        ca_lr = $config.CaLr
        ca_l2 = $config.CaL2
        ridge = $config.Ridge
        gradient_clip = $config.Clip
    }
}
$summary = [pscustomobject]@{
    purpose = 'timing test only'
    rule = 'inertial_reaction_diffusion'
    generations = 500
    fit_molecules_per_screen = 600
    validation_molecules_per_screen = 200
    wall_seconds = ((Get-Date) - $started).TotalSeconds
    configurations = $rows
}
$summary | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $root 'results\gpu_500_timing_tuning_summary.json')
$rows | Sort-Object validation_rmse | Format-Table -AutoSize
