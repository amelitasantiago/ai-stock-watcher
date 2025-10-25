# src/fetch_news_finnhub.py
import os, json, argparse, requests, pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from utils_io import ensure_dir

load_dotenv()
API_KEY = os.getenv("FINNHUB_KEY")

def daterange(days=1825):
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()

def company_news(symbol, _from, _to):
    url = "https://finnhub.io/api/v1/company-news"
    r = requests.get(url, params={"symbol": symbol, "from": _from, "to": _to, "token": API_KEY}, timeout=30)
    r.raise_for_status()
    return r.json()

def normalize(items, ticker, company_query):
    rows = []
    for a in items:
        ts = a.get("datetime")
        published = datetime.utcfromtimestamp(ts).isoformat()+"Z" if ts else None
        rows.append({
            "provider": "finnhub",
            "ticker": ticker,
            "company_query": company_query,
            "published_utc": published,
            "title": a.get("headline"),
            "snippet": a.get("summary"),
            "url": a.get("url"),
            "source": a.get("source"),
            "raw_json": json.dumps(a)
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="config/company_map.csv")
    ap.add_argument("--out", default="data/raw/news/finnhub.parquet")
    args = ap.parse_args()

    ensure_dir("data/raw/news")
    if not API_KEY:
        print("FINNHUB_KEY not set; skipping.")
        return

    _from, _to = daterange()
    m = pd.read_csv(args.mapping)
    all_df = []
    for _, row in m.iterrows():
        try:
            items = company_news(row["ticker"], _from, _to)
        except requests.HTTPError as e:
            print("Finnhub error for", row["ticker"], e)
            continue
        all_df.append(normalize(items, row["ticker"], row["company"]))
    if all_df:
        out = pd.concat(all_df, ignore_index=True).drop_duplicates(subset=["url"]).sort_values("published_utc")
        out.to_parquet(args.out, index=False)
        print("Saved:", args.out)

if __name__ == "__main__":
    main()
