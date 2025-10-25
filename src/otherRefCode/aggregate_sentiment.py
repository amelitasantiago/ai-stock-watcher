import os, pandas as pd
from datetime import datetime
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path

INP = "data/curated/news_last_30d.parquet"
OUT = "data/curated/daily_sentiment_last_30d.parquet"

def ensure_dir(p): Path(p).parent.mkdir(parents=True, exist_ok=True)

def main():
    if not os.path.exists(INP):
        raise SystemExit(f"Missing input: {INP}")

    df = pd.read_parquet(INP)
    if df.empty:
        # write empty with expected schema
        pd.DataFrame(columns=["ticker","date","sent"]).to_parquet(OUT, index=False)
        print("No news; wrote empty sentiment file:", OUT)
        return

    # Prepare text to score (title + snippet)
    text = (df["title"].fillna("") + ". " + df["snippet"].fillna("")).astype(str)

    # Score with VADER
    sia = SentimentIntensityAnalyzer()
    df["sent"] = text.map(lambda t: sia.polarity_scores(t)["compound"])

    # Normalize timestamp → date (UTC)
    ts = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df["date"] = ts.dt.date
    out = (df.dropna(subset=["date"])
             .groupby(["ticker","date"], as_index=False)["sent"].mean()
          )

    ensure_dir(OUT)
    out.to_parquet(OUT, index=False)
    print("Saved:", OUT, "rows=", len(out))

if __name__ == "__main__":
    main()
