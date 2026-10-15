<#
.SYNOPSIS
  Reproduce every AlphaLab result: data -> Rust build -> tests -> NSE study -> market-making ablation
  -> engine benchmark (quick) -> MATLAB cross-check.
.EXAMPLE
  .\run.ps1
  .\run.ps1 -SkipMatlab -SkipDownload
#>
param([switch]$SkipMatlab, [switch]$SkipDownload, [switch]$SkipBench)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { throw "Create the venv first: python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -e .[dev]" }

function Step($name, [scriptblock]$block) {
    Write-Host "== $name" -ForegroundColor Cyan
    $sw = [Diagnostics.Stopwatch]::StartNew()
    & $block
    if ($LASTEXITCODE -ne 0) { throw "$name failed (exit $LASTEXITCODE)" }
    Write-Host ("   done in {0:N0}s" -f $sw.Elapsed.TotalSeconds)
}

if (-not $SkipDownload) {
    Step 'NSE data (Nifty 200, Yahoo daily)' { & $py -c "from alphalab.data import nse_loader as L; L.build(force=True)" }
}
Step 'Build Rust engine (maturin)' {
    Push-Location (Join-Path $root 'lob_rs')
    $env:VIRTUAL_ENV = Join-Path $root '.venv'
    $env:PATH = "$env:VIRTUAL_ENV\Scripts;$env:PATH"
    maturin develop --release
    Pop-Location
}
Step 'NSE research study' { & $py -m alphalab.study }
Step 'Market-making ablation' { & $py -m alphalab.lob.ablation }
if (-not $SkipBench) { Step 'Engine benchmark (quick)' { & $py (Join-Path $root 'bench\bench_engines.py') --quick } }
if (-not $SkipMatlab) { Step 'MATLAB cross-check' { & $py -m alphalab.matlab_parity } }
Step 'Test suite' { & $py -m pytest -q }
Write-Host 'All steps completed. Results: reports\study\STUDY.md, reports\mm\MM_RESULTS.md, reports\matlab\MATLAB_RESULTS.md' -ForegroundColor Green
