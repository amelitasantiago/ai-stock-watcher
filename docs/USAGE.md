# 📖 The Night's Watch - Usage Guide

This guide provides detailed instructions on how to use The Night's Watch AI Stock Watcher.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Collection](#data-collection)
3. [Feature Engineering](#feature-engineering)
4. [Model Training](#model-training)
5. [Backtesting](#backtesting)
6. [Signal Generation](#signal-generation)
7. [Dashboard Usage](#dashboard-usage)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

---

## Getting Started

### First-Time Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
nano .env  # Edit with your API keys
```

3. **Verify Installation**
```bash
python -m aisw.cli --help
```

---

## Data Collection

### Download Historical Data

```bash
# Single ticker
python -m aisw.cli data --tickers AAPL --start 2023-01-01 --end 2024-01-01

# Multiple tickers
python -m aisw.cli data --tickers AAPL GOOGL MSFT TSLA --start 2023-01-01

# Using config file
python -m aisw.cli data --config config/config.yaml
```

### Data Update Schedule

For real-time monitoring, set up a cron job (Linux/macOS):

```bash
# Edit crontab
crontab -e

# Add this line to update every 5 minutes during market hours
*/5 9-16 * * 1-5 cd /path/to/ai-stock-watcher && python -m aisw.cli data --tickers AAPL GOOGL
```

---

## Feature Engineering

### Calculate Technical Indicators

```bash
# Basic indicators
python -m aisw.cli features --tickers AAPL

# Specify indicators
python -m aisw.cli features --tickers AAPL  --horizon 5 --window 252 --step 21

# Custom parameters
python -m aisw.cli features --tickers AAPL  --horizon 5 --window 252 --step 21
```

### Available Indicators

- **Moving Averages**: SMA, EMA, WMA
- **Momentum**: RSI, MACD, Stochastic
- **Volatility**: Bollinger Bands, ATR
- **Volume**: OBV, Volume SMA
- **Trend**: ADX, Aroon

---

## Model Training

### Train Individual Models

```bash
# Naive baseline
python -m aisw.cli train --tickers AAPL --models naive

# ARIMA
python -m aisw.cli train --tickers AAPL --models arima

# Train all models
python -m aisw.cli train --tickers AAPL --horizon 5 --window 252
```

### Ensemble Training

```bash
# Auto-weighted ensemble (inverse RMSE)
python -m aisw.cli train --tickers AAPL --ensemble auto

# Manual weights
python -m aisw.cli train --tickers AAPL --ensemble manual --weights 0.3,0.4,0.3
```

### Model Persistence

Trained models are automatically saved to:
```
models/final/{TICKER}/
├── naive_model.joblib
├── naive_model.meta.json
├── arima_model.joblib
├── arima_model.meta.json
├── rf_model.joblib
└── rf_model.meta.json
```

---

## Backtesting

### Walk-Forward Backtesting

```bash
# Basic backtest
python -m aisw.cli backtest --tickers AAPL GOOGL MSFT

# Custom windows
python -m aisw.cli backtest --tickers AAPL --horizon 5 --window 252 --step 21
```

### Backtest Results

Results are saved to `data/backtests/{TICKER}/`:
```
data/backtests/AAPL/
├── predictions.csv          # Model predictions
├── metrics.json             # Performance metrics
├── equity_curve.csv         # Portfolio value over time
└── trades.csv               # Individual trades
```

### Performance Metrics

- **Returns**: Total return, annualized return, CAGR
- **Risk**: Volatility, beta, max drawdown
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Trading**: Win rate, profit factor, average trade

---

## Signal Generation

### Generate Current Signals

```bash
# Generate signals for watchlist
python -m aisw.cli signals --tickers AAPL GOOGL MSFT

# Custom thresholds
python -m aisw.cli signals --tickers AAPL --horizon 5

# Export to CSV
python -m aisw.cli signals --tickers AAPL --export signals.csv
```

### Signal Interpretation

```
BUY:  Predicted return > buy_threshold (default: 2%)
HOLD: Predicted return between -2% and 2%
SELL: Predicted return < sell_threshold (default: -2%)
```

### Signal Confidence

Each signal includes a confidence score (0-1):
- **High Confidence (>0.75)**: Strong agreement across models
- **Medium Confidence (0.5-0.75)**: Moderate agreement
- **Low Confidence (<0.5)**: Weak or conflicting signals

---

## Dashboard Usage

### Launch Streamlit Dashboard

```bash
uvicorn insightfolio.server.server:app --port 8000
```

### Dashboard Features

* **Search bar:** Type tickers (spaces/commas) to set the watchlist.
* **Begin the Watch:** Parse tickers and build the watch grid with an immediate refresh.
* **Auto refresh toggle:** Turn 60-second polling on or off with LIVE/PAUSED status.
* **Manual refresh:** Pull the latest quotes and update visible panels on demand.
* **Watchlist cards:** Show symbol, price, day change, and a sparkline per ticker.
* **Card click:** Open the detail drawer for the selected ticker.
* **Overview – Maester’s Analysis:** Display Attack/Defend/Retreat with brief reasons.
* **Overview – Ancient AI Wisdom:** Show calibrated confidence, Now, Exp%, and Target Price.
* **Overview – Model Chips:** Indicate each model’s current vote or lean.
* **Overview – Model Performance:** Summarize OOS RMSE, MAE, Hit Rate, and sample size.
* **Overview – Sentiment:** Plot recent daily compound sentiment for context.
* **Overview – News preview:** List latest labeled headlines for a quick read.
* **News tab:** Present full live headline feed with source and sentiment badges.
* **Trade Plan tab:** Choose Attack/Defend/Retreat, add notes, and save the plan.
* **Save plan:** Create or update a paper position and log the decision snapshot.
* **Seer’s Chart – Historical:** Visualize strategy equity from walk-forward backtests.
* **Seer’s Chart – Forecast:** Show p10/p50/p90 fan and the horizon target price.
* **Portfolio tile – Realm’s Treasure:** Display total equity of open positions.
* **Portfolio tile – Fortune’s Change:** Show today’s dollar P&L versus prior close.
* **Portfolio tile – The Winds:** Show today’s percentage return for the portfolio.
* **Portfolio tile – Under Watch:** Count how many positions are currently open.
* **Active Positions:** Track entry, market value, and live P&L per position.
* **Portfolio Summary totals:** Aggregate total value, total P&L, and overall return.
* **Your Trade Plans:** Chronicle saved actions with timestamps and note snippets.
* **Keyboard shortcuts:** Provide quick keys for search, tabs, refresh, and close.
* **Error/empty states:** Show clear fallbacks when data is missing or feeds fail.

---

## Advanced Usage

### Custom Configuration

Edit `config/config.yaml`:

```yaml
data:
  tickers: [AAPL, GOOGL, MSFT, TSLA, NVDA]
  start_date: "2020-01-01"
  interval: "1d"

features:
  technical_indicators:
    - sma_20
    - sma_50
    - rsi
    - macd
  lags: [1, 2, 3, 5, 10]

models:
  train_test_split: 0.8
  walk_forward:
    train_window: 252
    test_window: 63

backtest:
  initial_capital: 100000
  transaction_cost_bps: 10
```

### Python API Usage

```python
from aisw.data.market_data import download_stock_data
from aisw.features.technical import calculate_technical_indicators
from aisw.signals.rules import generate_signals

# 1. Download data
data = download_stock_data('AAPL', start='2023-01-01')

# 2. Calculate features
features = calculate_technical_indicators(data)

# 3. Train model
model.train(features)

# 4. Generate predictions
predictions = model.predict(features)

# 5. Generate signals
signals = generate_signals(predictions)
print(signals)
```

### Batch Processing

Process multiple tickers in parallel:

```bash
# Create a ticker list file
echo "AAPL\nGOOGL\nMSFT\nTSLA" > tickers.txt

# Process all tickers
while read ticker; do
  python -m aisw.cli data --tickers $ticker
  python -m aisw.cli features --tickers $ticker
  python -m aisw.cli train --tickers $ticker
done < tickers.txt
```

---

## Troubleshooting

### Common Issues

#### 1. "No data returned for ticker"

**Cause**: Invalid ticker symbol or no data available  
**Solution**: 
- Verify ticker symbol on Yahoo Finance
- Check date range (ensure it's not in the future)
- Try a different data source

#### 2. "API rate limit exceeded"

**Cause**: Too many requests to data provider  
**Solution**:
- Add delays between requests
- Use cached data when possible
- Upgrade to premium API tier

#### 3. "Model training failed"

**Cause**: Insufficient data or invalid features  
**Solution**:
- Ensure at least 100 data points
- Check for NaN values in features
- Verify all required indicators are calculated

#### 4. "Import error: No module named 'aisw'"

**Cause**: Package not installed or PYTHONPATH not set  
**Solution**:
```bash
pip install -e .
# OR
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
python -m aisw.cli data --tickers AAPL
```

### Getting Help

1. Check the [FAQ](FAQ.md)
2. Search [GitHub Issues](https://github.com/amelitasantiago/ai-stock-watcher/issues)
3. Join our [Discord community](#)
4. Email: amelitasantiago@gmail.com

---

## Performance Optimization

### Speed Up Data Download

```bash
# Use parallel downloads
python -m aisw.cli data --tickers AAPL GOOGL MSFT --parallel

# Cache aggressively
python -m aisw.cli data --tickers AAPL --cache-ttl 3600
```

### Reduce Memory Usage

```python
# Use chunking for large datasets
import pandas as pd

chunksize = 10000
for chunk in pd.read_csv('data.csv', chunksize=chunksize):
    # Process chunk
    pass
```

### GPU Acceleration

For LSTM/Neural Network models:

```bash
# Install GPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Best Practices

1. **Always backtest** before live trading
2. **Use stop-losses** to limit downside risk
3. **Diversify** across multiple stocks
4. **Monitor performance** regularly
5. **Update models** periodically with new data
6. **Start small** with paper trading
7. **Keep detailed logs** of all trades
8. **Review signals** before acting on them

---

## Next Steps

- Explore [Advanced Features](ADVANCED.md)
- Read the [API Documentation](API.md)
- Check the [Roadmap](ROADMAP.md) for upcoming features
- Contribute to the project

---

**Happy Trading! 📈**

*Remember: This tool is for educational purposes. Always do your own research and consult with financial advisors.*
