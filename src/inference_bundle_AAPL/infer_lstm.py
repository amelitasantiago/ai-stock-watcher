#!/usr/bin/env python3
import argparse, json, os
import numpy as np, pandas as pd, joblib, tensorflow as tf
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="LSTM next-day return inference")
    ap.add_argument("--adj-close", required=True, help="Path to adj_close_wide.parquet")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--art-dir", default=".", help="Dir containing lstm_savedmodel.keras and seq_scaler.joblib")
    args = ap.parse_args()

    art = Path(args.art_dir)
    scaler_path = art / "seq_scaler.joblib"
    model_path  = art / "lstm_savedmodel.keras"
    if not scaler_path.exists(): raise SystemExit(f"Missing scaler: {scaler_path}")
    if not model_path.exists():  raise SystemExit(f"Missing model: {model_path}")

    prices = pd.read_parquet(args.adj_close).sort_index().asfreq("B")
    if args.ticker not in prices.columns:
        raise SystemExit(f"{args.ticker} not in price columns")
    ret = prices[args.ticker].pct_change().dropna()
    if len(ret) < args.window: raise SystemExit(f"Need at least {args.window} rows of returns")

    window = ret.iloc[-args.window:].values.astype("float32")
    sc = joblib.load(scaler_path)
    X = sc.transform(window.reshape(1, -1)).reshape(1, args.window, 1)

    model = tf.keras.models.load_model(model_path)
    pred = float(model.predict(X, verbose=0).ravel()[0])
    print(json.dumps({"ticker": args.ticker, "pred_next_return": pred}))

if __name__ == "__main__":
    main()
