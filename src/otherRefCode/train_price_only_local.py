import os, json, argparse, pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj-close", default="data/curated/adj_close_wide.parquet")
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--lags", type=int, default=20)
    ap.add_argument("--test-split", type=float, default=0.2)
    ap.add_argument("--art-dir", default="artifacts")
    args = ap.parse_args()

    Path(args.art_dir).mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(args.adj_close).sort_index().asfreq("B")
    if args.ticker not in prices.columns:
        raise SystemExit(f"{args.ticker} not in price columns")

    rets = prices.pct_change()
    y = rets[args.ticker].rename("y")

    # build lagged features
    df = pd.DataFrame({"y": y})
    for k in range(1, args.lags+1):
        df[f"lag_{k}"] = df["y"].shift(k)
    df = df.dropna()
    X = df.filter(like="lag_")
    y_next = df["y"].shift(-1).dropna()
    X = X.iloc[:-1]
    assert len(X) == len(y_next)

    split = int(len(X) * (1 - args.test_split))
    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    ytr, yte = y_next.iloc[:split], y_next.iloc[split:]

    # Naive-1 for reference
    yhat_naive = Xte["lag_1"].values
    rmse_naive = mean_squared_error(yte, yhat_naive)
    mae_naive = mean_absolute_error(yte, yhat_naive)
    dacc_naive = (np.sign(yhat_naive) == np.sign(yte.values)).mean()

    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))])
    pipe.fit(Xtr, ytr)
    yhat = pipe.predict(Xte)
    rmse = mean_squared_error(yte, yhat)
    mae = mean_absolute_error(yte, yhat)
    dacc = (np.sign(yhat) == np.sign(yte.values)).mean()

    joblib.dump(pipe, os.path.join(args.art_dir, "price_only_ridge.joblib"))
    meta = {            
        "ticker": args.ticker,
        "lags": int(args.lags),
        "features": list(X.columns),
        "metrics": {
            "ridge": {"rmse": float(rmse), "mae": float(mae), "dir_acc": float(dacc)},
            "naive1": {"rmse": float(rmse_naive), "mae": float(mae_naive), "dir_acc": float(dacc_naive)}
        }
    }
    with open(os.path.join(args.art_dir, "price_only_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved artifacts to:", args.art_dir)
    print("Ridge:", meta["metrics"]["ridge"], "Naive-1:", meta["metrics"]["naive1"])

if __name__ == "__main__":
    main()
