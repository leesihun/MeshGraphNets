# Sweep 5 launcher — runs all 16 training configs simultaneously.
# Each config carries its own gpu_ids; 2 processes share each GPU.
#
# GPU assignment (gpu_ids comes from each config file):
#   GPU 0: train1  (anchor)               + train11 (LR 0.001)
#   GPU 1: train2  (LR 0.00005)           + train12 (grad_accum 4 + LR 0.0002)
#   GPU 2: train3  (LR 0.0002)            + train13 (epochs 20000)
#   GPU 3: train4  (LR 0.0005)            + train14 (std_noise 0.01)
#   GPU 4: train5  (grad_accum 4)         + train15 (L=3 aggressive 2500/500/100)
#   GPU 5: train6  (epochs 5000)          + train16 (L=3 + bigger mp 32 blocks)
#   GPU 6: train7  (std_noise 0.005)      + train17 (mp 28 blocks at L=2)
#   GPU 7: train8  (L=3 5000/1000/200)    + train18 (Latent 192)
#
# Usage:
#   .\ex1\run_sweep5.ps1                # launch all 16
#   .\ex1\run_sweep5.ps1 -DryRun        # print commands without launching
#   .\ex1\run_sweep5.ps1 -Configs 1,5,11   # launch only selected configs
#   .\ex1\run_sweep5.ps1 -PlotOnly      # skip training, just plot existing logs
#
# After training completes (or at any time during), generate loss plots with:
#   python ex1\plot_sweep5.py --combined

param(
    [int[]]$Configs = @(1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18),
    [switch]$DryRun,
    [switch]$PlotOnly
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Resolve venv python (override with $env:PYTHON_BIN)
if ($env:PYTHON_BIN) {
    $Python = $env:PYTHON_BIN
} elseif (Test-Path (Join-Path $ProjectRoot "venv\Scripts\python.exe")) {
    $Python = Join-Path $ProjectRoot "venv\Scripts\python.exe"
} elseif (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")) {
    $Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
} elseif (Test-Path (Join-Path $ProjectRoot "venv\bin\python")) {
    $Python = Join-Path $ProjectRoot "venv\bin\python"
} else {
    Write-Host "ERROR: no venv python found. Looked for:" -ForegroundColor Red
    Write-Host "  $ProjectRoot\venv\Scripts\python.exe"
    Write-Host "  $ProjectRoot\.venv\Scripts\python.exe"
    Write-Host "  $ProjectRoot\venv\bin\python"
    Write-Host 'Set $env:PYTHON_BIN to override.'
    exit 1
}
Write-Host "Using python: $Python"
Write-Host ""

if ($PlotOnly) {
    Write-Host "Plot-only mode: generating loss plots from existing logs"
    $configArgs = $Configs | ForEach-Object { "$_" }
    & $Python (Join-Path $ScriptDir "plot_sweep5.py") @configArgs --combined
    exit
}

Write-Host "Sweep 5 launcher"
Write-Host "  Project root: $ProjectRoot"
Write-Host "  Configs:      $($Configs -join ', ')"
Write-Host "  Dry run:      $DryRun"
Write-Host ""

$launched = 0
$missing = @()

foreach ($n in $Configs) {
    $configRel = "ex1/config_train$n.txt"
    $configAbs = Join-Path $ProjectRoot $configRel
    $stdoutLog = Join-Path $ProjectRoot "ex1/train$n.stdout.log"
    $stderrLog = Join-Path $ProjectRoot "ex1/train$n.stderr.log"

    if (-not (Test-Path $configAbs)) {
        Write-Host "  SKIP train${n}: $configRel not found" -ForegroundColor Yellow
        $missing += $n
        continue
    }

    if ($DryRun) {
        Write-Host "  [dry] $Python MeshGraphNets_main.py --config $configRel  > $stdoutLog 2> $stderrLog"
        continue
    }

    Write-Host "  launching train$n -> $stdoutLog"
    Start-Process -FilePath $Python `
        -ArgumentList "MeshGraphNets_main.py", "--config", $configRel `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -WindowStyle Hidden | Out-Null
    $launched++
}

Write-Host ""
if ($DryRun) {
    Write-Host "Dry run complete. $($Configs.Count - $missing.Count) commands would launch."
} else {
    Write-Host "Launched $launched job(s)."
    if ($missing.Count -gt 0) {
        Write-Host "Missing configs (skipped): $($missing -join ', ')" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Monitor with:"
    Write-Host "  nvidia-smi -l 5"
    Write-Host "  Get-Content ex1/train1.log -Wait"
    Write-Host "  Get-Content ex1/train1.stdout.log -Wait"
    Write-Host ""
    Write-Host "Generate loss plots (anytime — re-run as runs progress):"
    Write-Host "  $Python ex1/plot_sweep5.py --combined"
    Write-Host "  $Python ex1/plot_sweep5.py 1 11          # specific configs"
    Write-Host "  .\ex1\run_sweep5.ps1 -PlotOnly           # same, via this script"
}
