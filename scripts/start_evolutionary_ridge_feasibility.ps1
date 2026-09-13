param(
    [ValidateRange(0, 4)]
    [int]$Fold = 0,
    [int]$Population = 64,
    [int]$Epochs = 80,
    [int]$BatchMolecules = 1024
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExecutable = 'C:\Users\Anthony\anaconda3\envs\strange-matter-gpu\python.exe'
$RunName = if ($Fold -eq 0) {
    'production_es_ea_cv_cyp_gca_feasibility_v1'
} else {
    "production_es_ea_cv_cyp_gca_feasibility_v1_fold$Fold"
}
$RunDirectory = Join-Path $ProjectRoot "results\$RunName"
New-Item -ItemType Directory -Force -Path $RunDirectory | Out-Null

$env:SME_TRAINING_ALGORITHM = 'evolution_strategy'
$env:SME_RUN_NAME = $RunName
$env:SME_DEVICE = 'cuda'
$env:SME_CA_RULE = 'damped_symplectic'
$env:SME_ACTIVE_CYP = 'CYP3A4'
$env:SME_SPECIALIST_OBJECTIVE = 'endpoint_only'
$env:SME_CV_FOLD = "$Fold"
$env:SME_CV_FOLDS = '5'
$env:SME_CV_SPLIT_SEED = '260822'
$env:SME_TUNING_ONLY = '1'
$env:SME_TUNING_FIT_MOLECULES = '2400'
$env:SME_TUNING_VAL_MOLECULES = '600'
$env:SME_MAX_EPOCHS = "$Epochs"
$env:SME_PATIENCE = '20'
$env:SME_MIN_DELTA = '0.002'
$env:SME_ES_POPULATION = "$Population"
$env:SME_ES_SIGMA = '0.02'
$env:SME_ES_LR = '0.003'
$env:SME_ES_BATCH_MOLECULES = "$BatchMolecules"
$env:SME_GENERATIONS = '32'
$env:SME_HIDDEN_CHANNELS = '16'
$env:SME_RIDGE = '0.1'
$env:SME_CA_L2 = '0.000001'
$env:SME_UPDATE_SCALE = '0.25'
$env:SME_INIT_SCALE = '1.5'
$env:SME_INITIAL_NOISE = '0.0'
$env:SME_SUPPORT_FRACTION = '0.6'
$env:SME_BOND_TEMPERATURE = '1.0'
$env:SME_DYN_A = '0.5'
$env:SME_DYN_B = '0.2'
$env:SME_DYN_C = '0.2'
$env:SME_DYN_D = '0.15'
$env:SME_ATOM_FEATURE_PROFILE = 'periodic_electronic'
$env:SME_TRAJECTORY_POOLING = 'multiscale'
$env:SME_RIDGE_MODE = 'shared'

$Process = Start-Process `
    -FilePath $PythonExecutable `
    -ArgumentList @('scripts/run_graph_ca_visual_prototype.py', 'train') `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput (Join-Path $RunDirectory 'campaign_stdout.log') `
    -RedirectStandardError (Join-Path $RunDirectory 'campaign_stderr.log') `
    -WindowStyle Hidden `
    -PassThru

[pscustomobject]@{
    RunName = $RunName
    ProcessId = $Process.Id
    OutputDirectory = $RunDirectory
    TrainingAlgorithm = $env:SME_TRAINING_ALGORITHM
    ScaffoldFold = $Fold
    Population = $Population
    Epochs = $Epochs
    BatchMolecules = $BatchMolecules
}
