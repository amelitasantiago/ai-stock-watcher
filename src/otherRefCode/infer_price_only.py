#!/usr/bin/env python3
"""
Infer next-day return using the price-only Ridge model.
- Reads artifacts from ART_DIR (price_only_ridge.joblib + price_only_meta.json)
- Takes either:
    (A) --features features_price_only.parquet  (wide returns, columns like AAPL_ret)
 OR (B) --adj-close adj_close_wide.parquet      (compute returns on the fly)

Output: JSON to stdout, e.g.
{"ticker":"AAPL","pred_next_return":0.00123,"direction":"up","lags_used":20,"last_date":"2025-09-12"}
"""
import argparse, json, os, sys
import pandas as pd
from pathlib import Path
import joblib

def err(msg: str, code: int = 1):
    print(f"[error] {msg}", file=sys.stderr); sys.exit(code)

def load_artifacts(art_dir: str):
    art = Path(art_dir)
    pipe_path = art / "price_only_ridge.joblib"
    meta_path = art / "price_only_meta.json"
    if not pipe_path.exists():
        err(f"Missing {pipe_path}. Train the price-only notebook first.")
    if not meta_path.exists():
        err(f"Missing {meta_path}. Train the price-only notebook first.")
    pipe = joblib.load(pipe_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ticker = meta.get("ticker")
    lags = int(meta.get("lags", 20))
    return pipe, ticker, lags, meta

def load_returns_from_features(path: str, ticker: str) -> pd.Series:
    if not os.path.exists(path):
        err(f"features file not found: {path}")
    df = pd.read_parquet(path)
    col = f"{ticker}_ret"
    if col not in df.columns:
        err(f"Column {col} not found in {path}. Available: {list(df.columns)[:8]} ...")
    s = df[col].dropna()
    if s.empty:
        err(f"No non-NaN returns found for {ticker} in {path}")
    s.index = pd.to_datetime(s.index)
    return s

def load_returns_from_adj_close(path: str, ticker: str) -> pd.Series:
    if not os.path.exists(path):
        err(f"adj_close file not found: {path}")
    prices = pd.read_parquet(path).sort_index().asfreq("B")
    if ticker not in prices.columns:
        err(f"Ticker {ticker} not found in {path}. Available: {list(prices.columns)[:8]} ...")
    s = prices[ticker].pct_change().dropna()
    s.index = pd.to_datetime(s.index)
    return s

def build_latest_window(returns: pd.Series, lags: int) -> pd.DataFrame:
    if len(returns) < lags:
        err(f"Need at least {lags} rows of returns; got {len(returns)}")
    # last date with a valid return
    last_date = returns.index.max()
    # construct lag_1 = last return, lag_k = k-th previous return
    row = {f"lag_{k}": float(returns.iloc[-k]) for k in range(1, lags+1)}
    X = pd.DataFrame([row], index=[last_date])
    return X, last_date

def main():
    ap = argparse.ArgumentParser(description="Price-only next-day return inference")
    ap.add_argument("--art-dir", required=True, help="Dir with price_only_ridge.joblib + price_only_meta.json")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--features", help="features_price_only.parquet (wide returns)")
    group.add_argument("--adj-close", help="adj_close_wide.parquet (raw prices)")
    ap.add_argument("--ticker", default=None, help="Override ticker (normally taken from artifacts)")
    ap.add_argument("--lags", type=int, default=None, help="Override lags (else from artifacts)")
    args = ap.parse_args()

    pipe, trained_ticker, trained_lags, meta = load_artifacts(args.art_dir)
    ticker = args.ticker or trained_ticker
    lags = args.lags if args.lags is not None else trained_lags

    # load returns series
    if args.features:
        returns = load_returns_from_features(args.features, ticker)
    else:
        returns = load_returns_from_adj_close(args.adj_close, ticker)

    X, last_date = build_latest_window(returns, lags)
    pred = float(pipe.predict(X)[0])

    direction = "up" if pred > 1e-8 else ("down" if pred < -1e-8 else "flat")
    out = {
        "ticker": ticker,
        "pred_next_return": pred,
        "direction": direction,
        "lags_used": int(lags),
        "last_date": last_date.strftime("%Y-%m-%d")
    }
    print(json.dumps(out))

if __name__ == "__main__":
    main()
