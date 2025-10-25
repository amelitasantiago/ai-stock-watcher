#!/usr/bin/env python3
import argparse, json, os
import numpy as np, pandas as pd, joblib, tensorflow as tf
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Bayesian Ridge (hybrid) next-day return inference")
    ap.add_argument("--adj-close", required=True, help="Path to adj_close_wide.parquet")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--lag-k", type=int, default=10)
    ap.add_argument("--art-dir", default=".", help="Folder with models and scaler")
    args = ap.parse_args()

    print("[info] art-dir:", args.art_dir)

    art = Path(args.art_dir)
    scaler_path = art / "seq_scaler.joblib"
    lstm_path   = art / "lstm_savedmodel.keras"
    tr_path     = art / "transformer_savedmodel.keras"
    bayes_path  = art / "bayes_hybrid.joblib"

    for p in [scaler_path, lstm_path, tr_path, bayes_path]:
        if not p.exists(): raise SystemExit(f"Missing artifact: {p}")

    prices = pd.read_parquet(args.adj_close).sort_index().asfreq("B")
    if args.ticker not in prices.columns:
        raise SystemExit(f"{args.ticker} not in price columns")
    ret = prices[args.ticker].pct_change().dropna()
    if len(ret) < args.window: raise SystemExit(f"Need at least {args.window} rows of returns")

    # Build normalized window
    window = ret.iloc[-args.window:].values.astype("float32")
    sc = joblib.load(scaler_path)
    X = sc.transform(window.reshape(1, -1)).reshape(1, args.window, 1)

    # Embeddings
    lstm = tf.keras.models.load_model(lstm_path)
    tr   = tf.keras.models.load_model(tr_path)
    lstm_embed = tf.keras.Model(lstm.input, lstm.get_layer("lstm_embedding").output)
    tr_embed   = tf.keras.Model(tr.input,   tr.get_layer("tr_embedding").output)
    e1 = lstm_embed.predict(X, verbose=0)       # (1, d1)
    e2 = tr_embed.predict(X, verbose=0)         # (1, d2)
    lags = X.reshape(1, args.window)[:, -args.lag_k:]  # (1, k)
    Z = np.concatenate([e1, e2, lags], axis=1)

    model = joblib.load(bayes_path)
    pred = float(model.predict(Z)[0])
    print(json.dumps({"ticker": args.ticker, "pred_next_return": pred}))

if __name__ == "__main__":
    main()
