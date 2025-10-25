#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

TICKERS=${TICKERS:-"AMZN NVDA AAPL MSFT TSLA META"}
[[ -f tickers.txt ]] && TICKERS="$(tr '\n' ' ' < tickers.txt)"

TS=$(date +"%Y-%m-%d_%H-%M-%S")
LOGDIR="logs/${TS%%_*}"; mkdir -p "$LOGDIR"
LOG="$LOGDIR/daily_$TS.log"

step(){ echo "== $1 ==" | tee -a "$LOG"; bash -lc "$2" 2>&1 | tee -a "$LOG"; }

LOOKBACK=800 HORIZON=5 WINDOW=252 STEP=21

step "Collect"   "python -m aisw.cli collect   --tickers $TICKERS --lookback $LOOKBACK"
step "Featurize" "python -m aisw.cli featurize --tickers $TICKERS"
step "Train"     "python -m aisw.cli train     --tickers $TICKERS --horizon $HORIZON --window $WINDOW"
step "Backtest"  "python -m aisw.cli backtest  --tickers $TICKERS --horizon $HORIZON --window $WINDOW --step $STEP"
step "Signals"   "python -m aisw.cli signal    --tickers $TICKERS --horizon $HORIZON"
step "Sentiment" "python -m aisw.cli sentiment --tickers $TICKERS --window-days 180"
step "Calibrate" "python -m aisw.cli calibrate --tickers $TICKERS --horizon $HORIZON"

echo -e "\nAll done ✅ Logs: $LOG"
