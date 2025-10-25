# src/fetch_prices.py
import argparse, re
from pathlib import Path
import pandas as pd
import yfinance as yf
from tqdm import tqdm

def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p

def load_tickers(path: str) -> list[str]:
    # Split on commas and whitespace; de-duplicate; drop empties
    tokens = []
    for line in Path(path).read_text().splitlines():
        for tok in re.split(r"[,\s]+", line.strip()):
            if tok:
                tokens.append(tok)
    # keep order while de-duping
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _extract_single_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Ensure df has flat columns for one ticker. If yfinance returned MultiIndex
    (e.g., because multiple tickers were passed), slice out the desired ticker.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Typical structure is like: top level: ['Open','High','Low','Close','Adj Close','Volume'],
        # second level: tickers. We want the columns for `ticker`.
        # If `ticker` missing (e.g., user passed malformed symbol), default to first available.
        level_names = df.columns.names
        # Try slicing by second level (ticker names)
        if ticker in df.columns.get_level_values(-1):
            df = df.xs(ticker, axis=1, level=-1)
        else:
            # fallback to the first ticker present
            first = df.columns.get_level_values(-1).unique()[0]
            print(f"[warn] Requested '{ticker}' not found in MultiIndex; using '{first}' slice instead.")
            df = df.xs(first, axis=1, level=-1)
    return df

def fetch_5y_daily_adj_close(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    df = _extract_single_ticker(df, ticker)
    df = df.reset_index()  # 'Date' column appears here
    # Normalize column names
    df = df.rename(columns={"Date": "date", "Adj Close": "adj_close"})

    # If 'adj_close' missing, fall back to 'Close'
    if "adj_close" not in df.columns and "Close" in df.columns:
        df["adj_close"] = df["Close"]

    # Add ticker column (broadcasts a scalar)
    df["ticker"] = ticker

    # Keep consistent set if available
    keep = ["ticker", "date", "adj_close", "Open", "High", "Low", "Close", "Volume"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols]

    # Clean dates
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="config/tickers.txt")
    ap.add_argument("--outdir", default="data/raw/prices")
    ap.add_argument("--curated", default="data/curated/adj_close_wide.parquet")
    args = ap.parse_args()

    ensure_dir(args.outdir)
    ensure_dir("data/curated")

    tickers = load_tickers(args.tickers)
    if not tickers:
        raise SystemExit("No tickers found. Put one ticker per line in config/tickers.txt (no commas).")

    dfs = []
    for t in tqdm(tickers, desc="Fetching prices"):
        df = fetch_5y_daily_adj_close(t)
        if df.empty:
            print(f"[warn] No data for {t}")
            continue
        df.to_parquet(f"{args.outdir}/{t}.parquet", index=False)
        dfs.append(df[["ticker","date","adj_close"]])

    if dfs:
        wide = pd.concat(dfs).pivot(index="date", columns="ticker", values="adj_close").sort_index()
        wide.to_parquet(args.curated, index=True)
        print("Saved:", args.curated)
    else:
        print("[warn] No price data saved.")

if __name__ == "__main__":
    main()
