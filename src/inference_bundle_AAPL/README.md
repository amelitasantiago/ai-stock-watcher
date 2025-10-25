# AI Stock Watcher — Inference Bundle (AAPL)

This folder contains ready-to-run **local inference** for the models you trained in Colab.

## Contents
- `infer_lstm.py` — next‑day return using **LSTM**.
- `infer_bayes_hybrid.py` — next‑day return using **Bayesian Ridge** on **hybrid (LSTM+Transformer embeddings + last-k lags)**.
- `infer_sarimax_exog.py` — next‑day return forecasting via **SARIMAX** with the same **hybrid exogenous** features.
- `requirements.txt` — minimal deps.
- Artifacts (from your Colab run):  
  - `lstm_savedmodel.keras`  
  - `transformer_savedmodel.keras`  
  - `bayes_hybrid.joblib`  
  - `sarimax.pkl`  
  - `meta.json`

> **Important**: You must also include the scaler used during training:
> - `seq_scaler.joblib` — saved from Colab (Section 4).  
>   Add this cell in your notebook and re-run it once:
>
> ```python
> # Save sequence scaler used to normalize windows
> import joblib, os
> joblib.dump(sc, os.path.join(ART_DIR, "seq_scaler.joblib"))
> print("Saved scaler →", os.path.join(ART_DIR, "seq_scaler.joblib"))
> ```
> Then download `seq_scaler.joblib` and place it next to these scripts.

## Quickstart (Windows PowerShell)
```powershell
# (optional) create venv
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip

# install deps
pip install -r requirements.txt

# LSTM inference (uses scaler + lstm_savedmodel.keras)
python infer_lstm.py `
  --adj-close C:\Users\ameli\ai_stock_watcher\data\curated\adj_close_wide.parquet `
  --ticker AAPL --window 90

# Bayesian Hybrid inference (needs lstm + transformer + scaler + bayes_hybrid.joblib)
python infer_bayes_hybrid.py `
  --adj-close C:\Users\ameli\ai_stock_watcher\data\curated\adj_close_wide.parquet `
  --ticker AAPL --window 90 --lag-k 10

# SARIMAX with exogenous features (needs sarimax.pkl + same hybrid features)
python infer_sarimax_exog.py `
  --adj-close C:\Users\ameli\ai_stock_watcher\data\curated\adj_close_wide.parquet `
  --ticker AAPL --window 90 --lag-k 10
```

## Notes
- **Window (`--window`)** must match the value used during training (default 90).
- **lag-k** must match the value used to build hybrid features in the notebook (default 10).
- All scripts expect `seq_scaler.joblib` and model files to be in the **same folder**.
