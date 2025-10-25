from fastapi import FastAPI, HTTPException, Query
"""Recreate the latest feature sequence for the model."""
df = dp.prepare_features(ticker, pred.config)
if df.empty:
raise RuntimeError("No data for features")
# keep only model features (already numeric-only during training)
feats = df[pred.feature_columns].values
seq_len = pred.config.sequence_length
if len(feats) < seq_len:
raise RuntimeError(f"Insufficient rows for sequence: have {len(feats)}, need {seq_len}")
recent_seq = feats[-seq_len:,:].reshape(1, seq_len, feats.shape[1])
last_close = float(df["close"].iloc[-1]) if "close" in df.columns else float("nan")
return recent_seq, last_close




@app.get("/api/tickers")
def list_tickers():
return CONFIG.get("tickers", [])




@app.get("/api/forecast", response_model=ForecastResponse)
def forecast(ticker: str = Query(..., min_length=1)):
ticker = ticker.upper()
try:
pred = load_predictor(ticker)
dp = DataProcessor() # will read DB paths from config/config.json
recent_seq, last_close = build_recent_sequence(dp, pred, ticker)


# raw per-model predictions
raw = pred.predict(recent_seq)
horizons = list(range(1, pred.config.prediction_horizon+1))


# post-process LSTM to original units if scaler present
models_out = {}
lstm = raw.get("lstm")
if lstm is not None and len(lstm) == pred.config.prediction_horizon and pred.lstm_scalers.get("y") is not None:
sy = pred.lstm_scalers["y"]
lstm = (np.asarray(lstm) * sy.scale_) + sy.mean_ # inverse transform
models_out["lstm"] = lstm.tolist()
elif lstm is not None:
models_out["lstm"] = np.asarray(lstm).tolist()


# pass-through ridge & arima
if raw.get("ridge") is not None:
models_out["ridge"] = np.asarray(raw["ridge"]).tolist()
if raw.get("arima") is not None:
models_out["arima"] = np.asarray(raw["arima"]).tolist()


# recompute ensemble (avoid unit mismatch)
weights = pred.ensemble_weights or {"lstm": 0.5, "ridge": 0.2, "arima": 0.3}
def w(name):
return float(weights.get(name, 0.0))
# fill missing arrays with zeros
def arr(name):
return np.asarray(models_out.get(name, np.zeros(pred.config.prediction_horizon)))
ensemble = w("lstm")*arr("lstm") + w("ridge")*arr("ridge") + w("arima")*arr("arima")


return {
"ticker": ticker,
"horizons": horizons,
"last_close": last_close,
"models": {k: [float(x) for x in v] for k, v in models_out.items()},
"ensemble": [float(x) for x in ensemble.tolist()],
}
except Exception as e:
traceback.print_exc()
raise HTTPException(status_code=500, detail=str(e))




@app.get("/api/history")
def history(ticker: str, days: int = 180):
dp = DataProcessor()
df = dp.load_stock_data(ticker)
if df.empty:
raise HTTPException(status_code=404, detail="No history")
df = df.tail(days)
return {
"ticker": ticker.upper(),
"dates": [str(d.date()) for d in df.index],
"close": [float(x) for x in df["close"].tolist()],
}