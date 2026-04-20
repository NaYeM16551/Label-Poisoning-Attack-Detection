param(
    [string]$Config = "configs/default.yaml",
    [switch]$IncludeRandomBaseline
)

$ErrorActionPreference = "Stop"

Write-Host "================================================================"
Write-Host " 1. Generate poisoned datasets"
Write-Host "================================================================"
$Rates = @(250, 500, 1000, 2500)
foreach ($n in $Rates) {
    Write-Host "==> Generating FLIP poisoned dataset with num_flips=$n"
    python -m src.attacks.generate_poisoned --config $Config --attack flip --num_flips $n
}

if ($IncludeRandomBaseline) {
    Write-Host "Generating optional random-flip baseline (num_flips=1000)"
    python -m src.attacks.generate_poisoned --config $Config --attack random --num_flips 1000
}

Write-Host "================================================================"
Write-Host " 2. Extract SSL features"
Write-Host "================================================================"
python scripts/03_extract_features.py --config $Config

Write-Host "================================================================"
Write-Host " 3. Run detection on FLIP-poisoned datasets"
Write-Host "================================================================"
$PoisonedFiles = Get-ChildItem -Path "data/poisoned" -Filter "flip_*.pt"
foreach ($File in $PoisonedFiles) {
    Write-Host "--> Detection on $($File.FullName)"
    python scripts/04_run_detection.py --config $Config --poisoned $File.FullName --skip_validation
}

if ($IncludeRandomBaseline) {
    $RandomFiles = Get-ChildItem -Path "data/poisoned" -Filter "random_*.pt"
    foreach ($File in $RandomFiles) {
        Write-Host "--> Detection on optional baseline $($File.FullName)"
        python scripts/04_run_detection.py --config $Config --poisoned $File.FullName --skip_validation
    }
}

Write-Host "================================================================"
Write-Host " 4. Mitigation on the canonical 2% FLIP dataset"
Write-Host "================================================================"
# In this specific project, data is cifar10
$DatasetName = python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['dataset']['name'])" $Config
$NumFlips = python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['attack']['num_flips'])" $Config

$ScoresFile = "results/detection/flip_${DatasetName}_${NumFlips}_scores.npz"
$PoisonedFile = "data/poisoned/flip_${DatasetName}_${NumFlips}.pt"

if ((Test-Path $ScoresFile) -and (Test-Path $PoisonedFile)) {
    python scripts/05_run_mitigation.py --config $Config --poisoned $PoisonedFile --scores $ScoresFile --mode remove
    python scripts/05_run_mitigation.py --config $Config --poisoned $PoisonedFile --scores $ScoresFile --mode none
} else {
    Write-Host "Skipping mitigation: missing $ScoresFile or $PoisonedFile"
}

Write-Host "================================================================"
Write-Host " 5. Aggregate plots"
Write-Host "================================================================"
python scripts/06_generate_plots.py --config $Config

Write-Host "Pipeline complete. See results/ for outputs."
