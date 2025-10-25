# scripts/daily.ps1
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Resolve-Path "$root\..")  # project root

$venv = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venv) { . $venv }

$tickersFile = "tickers.txt"
if (Test-Path $tickersFile) { $tickers = (Get-Content $tickersFile) -join " " } else { $tickers = "AMZN NVDA AAPL MSFT TSLA META" }

$ts  = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$logDir = "logs\$($ts.Split('_')[0])"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = "$logDir\daily_$ts.log"

function Step($title, $cmd) {
  Write-Host "== $title ==" -ForegroundColor Cyan
  Invoke-Expression $cmd 2>&1 | Tee-Object -FilePath $log -Append
  if ($LASTEXITCODE -ne 0) { throw "Failed at step: $title" }
}

# params
$lookback = 800
$horizon  = 5
$window   = 252
$stepDays = 21

Step "Collect"   "python -m aisw.cli collect   --tickers $tickers --lookback $lookback"
Step "Featurize" "python -m aisw.cli featurize --tickers $tickers"
Step "Train"     "python -m aisw.cli train     --tickers $tickers --horizon $horizon --window $window"
Step "Backtest"  "python -m aisw.cli backtest  --tickers $tickers --horizon $horizon --window $window --step $stepDays"
Step "Signals"   "python -m aisw.cli signal    --tickers $tickers --horizon $horizon"
Step "Sentiment" "python -m aisw.cli sentiment --tickers $tickers --window-days 180"
Step "Calibrate" "python -m aisw.cli calibrate --tickers $tickers --horizon $horizon"

Write-Host "`nAll done ✅ Logs: $log" -ForegroundColor Green
