import argparse, os, pandas as pd
from pathlib import Path

def ensure_dir(p: str): Path(p).mkdir(parents=True, exist_ok=True)

def load_prices(adj_close_path: str) -> pd.DataFrame:
    if not os.path.exists(adj_close_path):
        raise SystemExit(f"Missing file: {adj_close_path}")
    prices = pd.read_parquet(adj_close_path).sort_index().asfreq("B")
    return prices

def build_price_only_features(prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change()  # simple returns
    # suffix columns as <TICKER>_ret for clarity
    rets = rets.add_suffix("_ret")
    return rets

def load_sent(sent_path: str) -> pd.DataFrame:
    if not os.path.exists(sent_path):
        # return empty with expected columns
        return pd.DataFrame(columns=["ticker","date","sent"])
    df = pd.read_parquet(sent_path)
    return df

def build_with_sentiment(prices: pd.DataFrame, sent_df: pd.DataFrame, target: str,
                         ma7: int = 7, ma14: int = 14) -> pd.DataFrame:
    target_ret = prices[[target]].pct_change().rename(columns={target: "ret"})

    st = sent_df[sent_df["ticker"] == target].copy()
    if st.empty:
        print(f"[warn] No sentiment rows for {target}; returning empty DataFrame.")
        return pd.DataFrame(columns=["ret","sent","sent_ma7","sent_ma14"])

    st["date"] = pd.to_datetime(st["date"])
    st = st.set_index("date").sort_index().asfreq("B")
    st["sent"] = st["sent"].fillna(0.0)
    st["sent_ma7"]  = st["sent"].rolling(ma7,  min_periods=1).mean()
    st["sent_ma14"] = st["sent"].rolling(ma14, min_periods=1).mean()

    # align by index; keep only overlapping dates
    df = target_ret.join(st[["sent","sent_ma7","sent_ma14"]], how="inner").dropna(subset=["ret"])
    return df

def main():
    ap = argparse.ArgumentParser(description="Build features for AI Stock Watcher")
    ap.add_argument("--adj-close", default="data/curated/adj_close_wide.parquet")
    ap.add_argument("--sent", default="data/curated/daily_sentiment_last_30d.parquet")
    ap.add_argument("--outdir", default="data/curated")
    ap.add_argument("--target", default=None, help="Target ticker for sentiment-enhanced features (e.g., AAPL)")
    ap.add_argument("--ma7", type=int, default=7)
    ap.add_argument("--ma14", type=int, default=14)
    args = ap.parse_args()

    ensure_dir(args.outdir)
    prices = load_prices(args.adj_close)

    # 1) 5y price-only (wide returns)
    feat_price_only = build_price_only_features(prices)
    out_price_only = os.path.join(args.outdir, "features_price_only.parquet")
    feat_price_only.to_parquet(out_price_only)
    print("Saved:", out_price_only, "shape=", feat_price_only.shape)

    # 2) Optional: ticker-specific features with sentiment
    sent_df = load_sent(args.sent)
    if args.target:
        if args.target not in prices.columns:
            raise SystemExit(f"Target '{args.target}' not in price columns: {list(prices.columns)[:5]} ...")
        feat_with_sent = build_with_sentiment(prices, sent_df, args.target, args.ma7, args.ma14)
        out_with_sent = os.path.join(args.outdir, f"features_with_sent_{args.target}.parquet")
        feat_with_sent.to_parquet(out_with_sent)
        print("Saved:", out_with_sent, "shape=", feat_with_sent.shape)
    else:
        print("No --target provided → skipped sentiment-enhanced features.")

if __name__ == "__main__":
    main()
