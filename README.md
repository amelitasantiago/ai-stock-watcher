# 🌙 The Night's Watch - AI Stock Watcher

An intelligent, real-time stock market monitoring system powered by AI that identifies critical market signals—unusual volatility, volume spikes, and sentiment shifts—delivering proactive alerts to retail investors.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

NUS-ISS Intelligent Reasoning Systems | 2025

**Developers:**  Lee Fang Hui | Amelita Talavera Santiago | Hong Jin JIe

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Models & Algorithms](#models--algorithms)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

**The Night's Watch** addresses a critical challenge faced by retail investors: the inability to monitor vast amounts of market data across multiple sources simultaneously. By deploying AI-powered surveillance, the system:

- **Monitors** stock markets in real-time with minute-to-hour (default 1h) data refresh rates
- **Analyzes** sentiment from financial news using NLP


### The Problem We Solve

Modern retail investors face three major challenges:
1. **Information Overload**: Impossible to manually process price, volume, news, and news sentiment
2. **Reaction Latency**: By the time investors discover events, institutional algorithms have already acted
3. **Emotional Decision-Making**: Fear and greed drive impulsive decisions without systematic analysis

### Our Solution

A tireless AI sentinel that combines multiple intelligence signals into actionable insights, democratizing institutional-grade market monitoring for individual investors.

---

## ✨ Features

### Core Capabilities

- 🔍 **Real-Time Market Surveillance** - Continuous monitoring of user-specified watchlists (10-50 stocks)
- 💬 **Sentiment Analysis** - Real-time processing of financial news using VADER (baseline) with optional FinBERT (batched)
- 📈 **Technical Indicators** - Automated calculation of RSI, MACD, Bollinger Bands, and more
- 🎯 **Multi-Signal Fusion** - Combines price, volume, and sentiment for comprehensive risk scoring
- 🔄 **Walk-Forward Backtesting** - Validate strategies on historical data with realistic constraints
- 📱 **Night’s Watch UI** - Interactive web UI for real-time monitoring and visualization

### Intelligent Reasoning Techniques

- **Supervised Learning**: LSTM models for time-series forecasting
- **Transfer Learning**: Fine-tuned FinBERT for financial sentiment analysis
- **Ensemble Methods**: Weighted fusion of multiple signals

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│         DATA INGESTION LAYER                        │
│  ┌──────────┐  ┌──────────┐                         │
│  │ Yahoo    │  │ NewsAPI  │                         │
│  │ Finance  │  │          │                         │
│  └────┬─────┘  └────┬─────┘                         │
└───────┼─────────────┼───────────────────────────────┘
        │             │                              
        ▼             ▼                              
┌────────────────────────────────────────────────────┐
│    DATA PROCESSING & FEATURE ENGINEERING           │
│  • Normalization  • Validation  • Indicators       │
│  • Text Preprocessing  • Embedding Generation      │
└──────────────────────┬─────────────────────────────┘
                       │                              
                       ▼                              
┌───────────────────────────────────────────────────┐
│          AI REASONING CORE                        │
│  ┌───────────────────────────────┐                │
│  │ Sentiment Analysis (VADER)    │                │
│  └───────────────┬───────────────┘                │
│                  ▼                                │
│        ┌───────────────────────┐                  │
│        │ Signal Fusion &       │                  │
│        │ Scoring (Ensemble)    │                  │
│        └───────────────────────┘                  │
└───────────────────────────────────────────────────┘
```
**Technology Stack** 

* **Backend:** Python 3.11, FastAPI for REST APIs, Uvicorn ASGI server.
* **Data Ingestion:** yfinance for OHLCV/quotes, Yahoo Finance RSS via feedparser (News).
* **Data Science Core:** Pandas, NumPy, Scikit-learn (pipelines, calibration, RF).
* **Time-Series & Forecasting:** statsmodels (ARIMA), pmdarima (auto-ARIMA), SciPy.
* **NLP / Sentiment:** NLTK VADER (fast), Transformers/FinBERT (ProsusAI/finbert) optional.
* **Storage / Database:** **SQLite** (local .db), Parquet/CSV for data files, joblib for models.
* **Feature Engineering:** TA indicators (SMA/EMA/RSI/MACD), rolling stats, lagged returns.
* **Ensembling & Calibration:** inverse-error weighting, isotonic regression to P(up).
* **Backtesting:** custom walk-forward engine (window/step), metrics (Sharpe, MaxDD, Brier).
* **Frontend/UI:** HTML/CSS/JS (index.html + app.js) and/or React/Next.js (page.tsx), Streamlit optional.
* **Charts/Visualization:** Chart.js or ECharts in web UI; matplotlib for static plots.
* **Config & Secrets:** JSON config (`config/config.json`) + `.env` overrides, python-dotenv for .env handling.
* **Testing & Quality:** pytest, black, isort, mypy (optional typing checks).
* **DevOps / CI:** Git + GitHub Actions (lint/build), Docker optional for deployment.
---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (for cloning the repository)

### Step 1: Clone the Repository

```bash
git clone https://github.com/amelitasantiago/ai-stock-watcher.git
cd ai-stock-watcher
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Core pipeline dependencies
pip install -r requirements.txt

# Optional: UI dependencies for Streamlit dashboard
pip install -r requirements-ui.txt

# Optional: Advanced ML models (FinBERT, LSTM, VADER)
pip install -r requirements-extras-legacy.txt
```

### Step 4: Configuration

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys and preferences
nano .env  # or use your preferred editor
```

---

## ⚡ Quick Start

### Option 1: Run Complete Pipeline (Recommended)

**Linux/macOS:**
```bash
chmod +x run_all.sh
./run_all.sh
```

**Windows:**
```bash
run_all.bat
```

This will:
1. Download historical data
2. Calculate technical indicators
3. Perform sentiment analysis
4. Train models
5. Run backtests
6. Generate signals

### Option 2: Run Individual Modules

```bash
# 1. Download stock data
python -m aisw.cli data --tickers AAPL GOOGL MSFT --start 2023-01-01

# 2. Calculate technical indicators
python -m aisw.cli features --tickers AAPL GOOGL MSFT

# 3. Run sentiment analysis (optional)
python -m aisw.cli sentiment --tickers AAPL GOOGL MSFT

# 4. Train models
python -m aisw.cli train --tickers AAPL GOOGL MSFT --models naive arima rf

# 5. Run backtest
python -m aisw.cli backtest --tickers AAPL GOOGL MSFT

# 6. Generate trading signals
python -m aisw.cli signals --tickers AAPL GOOGL MSFT
```

### Option 3: Launch Web Dashboard

```bash
pip install -r requirements-ui.txt
uvicorn insightfolio.server.server:app --port 8000
```

Navigate to `http://127.0.0.1:8000` in your browser.

---

## 📖 Usage

### Basic Usage Examples

#### Monitor a Single Stock

```python
python -m aisw.cli collect   --tickers AMZN NVDA AAPL MSFT TSLA META --lookback 800
python -m aisw.cli featurize --tickers AMZN NVDA AAPL MSFT TSLA META 
python -m aisw.cli train     --tickers AMZN NVDA AAPL MSFT TSLA META --horizon 5 --window 252
python -m aisw.cli backtest  --tickers AMZN NVDA AAPL MSFT TSLA META --horizon 5 --window 252 --step 21
python -m aisw.cli signal    --tickers AMZN NVDA AAPL MSFT TSLA META --horizon 5
```

# View metrics
print(f"Sharpe Ratio: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Total Return: {results['total_return']:.2%}")
```

print(signals)
# Output:
# AMZN: BUY (confidence: 0.78)
# NVDA: HOLD (confidence: 0.55)
# AAPL: SELL (confidence: 0.82)
```

---

## ⚙️ Configuration

### Main Configuration File: `config/config.json`

```json
{
  "data": {
    "tickers": ["AMZN","NVDA","AAPL","MSFT","TSLA","META"],
    "start_date": "2020-01-01",
    "end_date": "2024-01-01",
    "interval": "1d"
  },
  "features": {
    "technical_indicators": ["sma_20","sma_50","rsi","macd","bollinger_bands"],
    "lags": [1,2,3,5,10]
  },
  "models": {
    "walk_forward": {"train_window": 252, "test_window": 63},
    "ensemble_weights": "auto"
  },
  "backtest": {
    "initial_capital": 100000,
    "transaction_cost_bps": 10,
    "position_size": "equal"
  }
}
```

### Environment Variables: `.env`

```bash
# API Keys
ALPHA_VANTAGE_KEY=your_key_here
NEWS_API_KEY=your_key_here
FINNHUB_KEY=your_key_here
GNEWS_KEY=your_key_here

# Logging
LOG_LEVEL=INFO
```

---

## 📁 Project Structure

```
ai-stock-watcher/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── .gitignore                      # Git ignore rules
├── .env.example                    # Sample environment variables
├── requirements.txt                # Core dependencies
├── requirements-extras-legacy.txt  # Optional ML models runs in Colab
├── run_all.sh                      # Linux/macOS runner
├── run_all.bat                     # Windows runner
│
├── config/
│   ├── config.yaml                 # Main configuration
│   
│
├── src/
│   └── aisw/                       # Main package
│       ├── __init__.py
│       ├── cli.py                  # Command-line interface
│       ├── config.py               # Config loader
│       ├── utils.py                # Utilities
│       │
│       ├── data/                   # Data ingestion
│       │   ├── market_data.py      # Price/volume downloader
│       │   └── news.py             # News fetcher
│       │
│       ├── features/               # Feature engineering
│       │   └── technical.py        # Technical indicators
│       │
│       ├── sentiment/              # Sentiment analysis
│       │   ├── lexicon.py          # Sentiment scoring
|       |   └── collect.py          # Collecting sentiments
│       │
│       ├── models/                 # Prediction models
│       │   ├── base.py             # Base model class
│       │   ├── naive.py            # Baseline model
│       │   ├── arima.py            # ARIMA wrapper
│       │   ├── rf.py               # Random Forest
│       │   └── registry.py         # Model persistence
│       │
│       ├── ensemble/               # Ensemble methods
│       │   └── weighted.py         # Weighted blending
│       │
│       ├── backtest/               # Backtesting engine
│       │   ├── walkforward.py      # Walk-forward analysis
│       │   └── metrics.py          # Performance metrics
│       │
│       ├── signals/                # Trading signals
│       │   └── rules.py            # Signal generation
│       │
│       └── notebook/                 # Original implementations
│           ├── financialSentimentAnalysis.py
│           ├── YFinanceStockDataCollection.py
│           └── multiModelStocksNightsWatch_v2.py
|           └── AI_Stock_Watcher_Colab_Setup.ipynb
|
│
├── insightfolio/                   # Web dashboard
│   └── app/
│       └── app.py                  # Uvicorn UI
│
├── docs/                           # Documentation
│   └── The_Nights_Watch_Documentation.docx
│
├── tests/                          # Unit tests
│   ├── test_sanity.py
│   ├── test_features.py
│   └── test_backtest.py
│
├── logs/                           # Log files (gitignored)
│   └── logging.conf
│
├── models/                         # Trained models (gitignored)
│   └── final/
│       └── {TICKER}/
│
└── data/                           # Generated data (gitignored)
    ├── my_stock_data/
    ├── comprehensive_news_data/
    ├── financial_sentiment_results/
    ├── backtests/
    └── signals/
```

---

## 🧠 Models & Algorithms

### Sentiment Analysis

**FinBERT (Transfer Learning)**
- Pre-trained transformer model fine-tuned on financial texts
- Advantages: 15% accuracy improvement over generic BERT
- Use case: Classifying news sentiment, news sentiment

### Time-Series Forecasting


**ARIMA (AutoRegressive Integrated Moving Average)**
- Statistical baseline for time-series prediction
- Advantages: Interpretable, fast training
- Use case: Baseline comparison, trend analysis

**Random Forest**
- Ensemble learning on technical indicators and lagged features
- Advantages: Robust to overfitting, feature importance
- Use case: Multi-factor prediction, ensemble member

### Ensemble Methods

**Weighted Averaging**
- Combines predictions from multiple models using inverse-RMSE weights
- Aims to reduce false positives versus single models (validated in backtests; results vary)

---

## 📚 Documentation

- **[Complete Project Documentation](docs/The_Nights_Watch_Documentation.docx)** - Comprehensive technical documentation
- **[Usage Guide](docs/USAGE.md)** - Detailed usage instructions

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_features.py

# Run with coverage
pytest --cov=aisw tests/
```

---

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation as needed
- Use meaningful commit messages

---

## 📊 Performance Metrics

> **Note:** The figures below are illustrative demo metrics. Replace with your latest backtest run before submission.

## # Backtesting Results (Q4 2024)

- **Early Alert Success**: Identified 15% price drop 30 minutes before news
- **Sentiment Correlation**: -0.68 with next-hour price movements
- **False Positive Rate**: 18% (acceptable for conservative system)
- **Average Alert Latency**: 47 seconds
- **System Uptime**: 99.7%

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

- **Project Repository**: [github.com/amelitasantiago/ai-stock-watcher](https://github.com/amelitasantiago/ai-stock-watcher)
- **Email**: amelitasantiagot@gmail.com; leefanghui@gmail.com; regenorak@gmail.com


---

## 🙏 Acknowledgments

**NUS-ISS Faculty:** For guidance and supervision from: Prof. Zhu Fangming: Prof. Gary Leung: Prof. Xavier Xie; Prof. Ding Liya; Prof. TIAN Jing; Prof. Barry Shepherd; Prof. Wang Aubo; Prof. Marylyn Xiang
- Open-source community for essential tools:
  - [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance data
  - [scikit-learn](https://scikit-learn.org/) - Machine learning
  - [Hugging Face Transformers](https://huggingface.co/) - NLP models

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.** It does not constitute financial advice, investment recommendations, or trading signals. Always consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results. Trading involves substantial risk of loss.

---

<div align="center">

**Built with ❤️ by The Night's Watch Team**

*"We are the shield that guards the portfolios of retail investors."*

⭐ Star us on GitHub if you find this project useful!

</div>
