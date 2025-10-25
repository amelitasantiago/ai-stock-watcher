# src/fetch_news_newsapi.py
import os, time, json, argparse, requests, pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv
from utils_io import ensure_dir

load_dotenv()
API_KEY = os.getenv("NEWSAPI_KEY")

SCHEMA = ["provider","ticker","company_query","published_utc","title","snippet","url","source","raw_json"]
BASE = "https://newsapi.org/v2/everything"

def daterange(days:int=1825):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return start.date().isoformat(), end.date().isoformat()

def search_company(company:str, from_date:str, to_date:str, page_size:int=100, max_pages:int=3):
    hdrs = {"X-Api-Key": API_KEY}
    items = []
    for page in range(1, max_pages+1):
        params = {
            "q": f"\"{company}\"",
            "language": "en",
            "from": from_date,
            "to": to_date,
            "sortBy": "publishedAt",
            "pageSize": page_size,
            "page": page
        }
        r = requests.get(BASE, headers=hdrs, params=params, timeout=30)
        r.raise_for_status()
        batch = r.json().get("articles", [])
        items.extend(batch)
        if len(batch) < page_size:
            break
        time.sleep(1)
    return items

def normalize(items, ticker, company) -> pd.DataFrame:
    rows = []
    for a in (items or []):
        published = a.get("publishedAt") or a.get("published_at")
        rows.append({
            "provider": "newsapi",
            "ticker": ticker,
            "company_query": company,
            "published_utc": published,                  # may be None → that’s OK
            "title": a.get("title"),
            "snippet": a.get("description"),
            "url": a.get("url"),
            "source": (a.get("source") or {}).get("name"),
            "raw_json": json.dumps(a)
        })
    df = pd.DataFrame(rows)
    # Ensure columns exist even if rows == 0
    return df.reindex(columns=SCHEMA)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="config/company_map.csv", help="CSV with columns: ticker,company")
    ap.add_argument("--out", default="data/raw/news/newsapi.parquet")
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit("Missing NEWSAPI_KEY in .env (set NEWSAPI_KEY=your_key)")

    ensure_dir("data/raw/news")
    m = pd.read_csv(args.mapping)
    frm, to = daterange()

    dfs = []
    for _, row in m.iterrows():
        try:
            items = search_company(row["company"], frm, to)
        except requests.HTTPError as e:
            print(f"[warn] NewsAPI error for {row['company']}: {e}")
            continue
        df = normalize(items, row["ticker"], row["company"])
        if not df.empty:
            dfs.append(df)

    if not dfs:
        # Write an empty file with the right schema so downstream steps won't fail
        pd.DataFrame(columns=SCHEMA).to_parquet(args.out, index=False)
        print("No NewsAPI results; wrote empty file with schema:", args.out)
        return

    out = pd.concat(dfs, ignore_index=True)
    if "published_utc" in out.columns:
        out = out.sort_values("published_utc")
    out = out.drop_duplicates(subset=["url"])
    out.to_parquet(args.out, index=False)
    print("Saved:", args.out, "rows=", len(out))

if __name__ == "__main__":
    main()
