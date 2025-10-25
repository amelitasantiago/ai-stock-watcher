param(
  [string[]] $Tickers = @("AMZN","NVDA"),
  [int] $Lookback = 800,
  [int] $Horizon = 5,
  [int] $Window = 252,
  [int] $Step = 21,
  [switch] $Clean,         # remove backtests/signals/models
  [switch] $Redownload     # also delete cached price/features CSVs
)

# Activate venv if present
$activate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activate) { & $activate }

# Timestamped log folder
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logdir = Join-Path "logs" $ts
New-Item -ItemType Directory -Path $logdir -Force | Out-Null

Write-Host "Logs -> $logdir`n"

# Optional: clean artifacts
if ($Clean) {
  Write-Host "Cleaning /data/backtests, /data/signals, /models/final ..."
  Remove-Item -Force -Recurse .\data\backtests, .\data\signals -ErrorAction SilentlyContinue
  Remove-Item -Force -Recurse .\models\final -ErrorAction SilentlyContinue
}
if ($Redownload) {
  foreach ($t in $Tickers) {
    Remove-Item -Force ".\data\my_stock_data\${t}_1d.csv"," .\data\my_stock_data\${t}_features.csv" -ErrorAction SilentlyContinue
  }
}

function Run-Step {
  param([string]$Name, [string[]]$Args, [string]$LogFile)
  Write-Host "`n=== $Name ==="
  python -u -m aisw.cli @Args *>&1 | Tee-Object -FilePath $LogFile
}

# 1) collect
Run-Step "collect" @("collect","--tickers") + $Tickers + @("--lookback",$Lookback) `
  (Join-Path $logdir "01_collect_$($Tickers -join '_')_lb$Lookback.log")

# 2) featurize
Run-Step "featurize" @("featurize","--tickers") + $Tickers `
  (Join-Path $logdir "02_featurize_$($Tickers -join '_').log")

# 3) train
Run-Step "train" @("train","--tickers") + $Tickers + @("--horizon",$Horizon,"--window",$Window) `
  (Join-Path $logdir "03_train_$($Tickers -join '_')_h$Horizon_w$Window.log")

# 4) backtest
Run-Step "backtest" @("backtest","--tickers") + $Tickers + @("--horizon",$Horizon,"--window",$Window,"--step",$Step) `
  (Join-Path $logdir "04_backtest_$($Tickers -join '_')_h$Horizon_w$Window_s$Step.log")

# 5) signal
Run-Step "signal" @("signal","--tickers") + $Tickers + @("--horizon",$Horizon) `
  (Join-Path $logdir "05_signal_$($Tickers -join '_')_h$Horizon.log")

Write-Host "`nAll done. Logs saved in: $logdir"
