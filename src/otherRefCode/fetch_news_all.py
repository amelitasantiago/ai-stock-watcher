# src/fetch_news_all.py
import os, json, time, argparse, requests, pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

SCHEMA = ["provider","ticker","company_query","published_utc","title","snippet","url","source","raw_json"]

def ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True); return p

def daterange(days: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return start.date().isoformat(), end.date().isoformat()

def _norm_df(rows):
    df = pd.DataFrame(rows).reindex(columns=SCHEMA)
    if df.empty: return df
    # robust sort by timestamp if present
    ts = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df.assign(_ts=ts).sort_values("_ts").drop(columns="_ts")
    # de-dupe by URL (primary) then (title, published_utc) as fallback
    df = df.drop_duplicates(subset=["url"]).drop_duplicates(subset=["title","published_utc"])
    return df

# ---------- Providers ----------
def fetch_newsapi(company, ticker, frm, to, key, page_size=100, max_pages=3):
    if not key: return []
    base = "https://newsapi.org/v2/everything"
    hdrs = {"X-Api-Key": key}
    q = f"\"{company}\" OR {ticker}"
    out = []
    for page in range(1, max_pages+1):
        params = {"q": q, "language":"en", "from": frm, "to": to,
                  "sortBy":"publishedAt", "pageSize": page_size, "page": page}
        #r = requests.get(base, headers=hdrs, params=params, timeout=30)
        #r.raise_for_status()
        try:
            r = requests.get(base, headers=hdrs, params=params, timeout=30)
            r.raise_for_status()
        except requests.HTTPError as e:
            # NewsAPI Developer plan often returns 426 on page>1 or heavier usage
            if getattr(e.response, "status_code", None) == 426:
                print(f"[newsapi] {ticker}: Developer plan limit hit → stopping at page {page}.")
                break
            raise
        arts = r.json().get("articles", [])
        for a in arts:
            out.append({
                "provider": "newsapi",
                "ticker": ticker,
                "company_query": company,
                "published_utc": a.get("publishedAt") or a.get("published_at"),
                "title": a.get("title"),
                "snippet": a.get("description"),
                "url": a.get("url"),
                "source": (a.get("source") or {}).get("name"),
                "raw_json": json.dumps(a)
            })
        if len(arts) < page_size: break
        time.sleep(1)
    return out

def fetch_finnhub(company, ticker, frm, to, key):
    if not key: return []
    url = "https://finnhub.io/api/v1/company-news"
    r = requests.get(url, params={"symbol": ticker, "from": frm, "to": to, "token": key}, timeout=30)
    r.raise_for_status()
    rows = []
    for a in r.json():
        ts = a.get("datetime")
        #published = datetime.utcfromtimestamp(ts).isoformat()+"Z" if ts else None
        # timezone-aware UTC conversion (avoids deprecation warning)
        published = (datetime.fromtimestamp(ts, tz=timezone.utc)
                .isoformat().replace("+00:00","Z")) if ts else None
        rows.append({
            "provider": "finnhub",
            "ticker": ticker,
            "company_query": company,
            "published_utc": published,
            "title": a.get("headline"),
            "snippet": a.get("summary"),
            "url": a.get("url"),
            "source": a.get("source"),
            "raw_json": json.dumps(a)
        })
    return rows

def fetch_gnews(company, ticker, frm, to, key, max_items=100):
    if not key: return []
    url = "https://gnews.io/api/v4/search"
    # GNews supports ISO dates in from/to and returns ~30d history on free tier
    q = f"\"{company}\" OR {ticker}"
    r = requests.get(url, params={"q": q, "lang":"en", "from": frm, "to": to, "max": max_items, "token": key}, timeout=30)
    r.raise_for_status()
    rows = []
    for art in r.json().get("articles", []):
        rows.append({
            "provider": "gnews",
            "ticker": ticker,
            "company_query": company,
            "published_utc": art.get("publishedAt"),
            "title": art.get("title"),
            "snippet": art.get("description"),
            "url": art.get("url"),
            "source": (art.get("source") or {}).get("name"),
            "raw_json": json.dumps(art)
        })
    return rows

# ---------- Main ----------
def main():
    load_dotenv(find_dotenv())
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping", default="config/company_map.csv", help="CSV: ticker,company")
    ap.add_argument("--out", default="data/curated/news_last_30d.parquet")
    ap.add_argument("--days", type=int, default=15)
    ap.add_argument("--per_provider_limit", type=int, default=3, help="NewsAPI pages (x page_size); GNews max items ~100")
    args = ap.parse_args()

    newsapi_key = os.getenv("NEWSAPI_KEY")
    finnhub_key = os.getenv("FINNHUB_KEY")
    gnews_key = os.getenv("GNEWS_KEY")

    ensure_dir("data/curated")
    frm, to = daterange(args.days)
    m = pd.read_csv(args.mapping)

    all_rows = []
    provider_counts = {"newsapi":0, "finnhub":0, "gnews":0}

    for _, row in m.iterrows():
        ticker, company = str(row["ticker"]).strip(), str(row["company"]).strip()
        # Finnhub (symbol-based)
        try:
            rows = fetch_finnhub(company, ticker, frm, to, finnhub_key)
            provider_counts["finnhub"] += len(rows); all_rows.extend(rows)
        except requests.HTTPError as e:
            print(f"[finnhub] {ticker}: {e}")

        # NewsAPI (query-based)
        try:
            rows = fetch_newsapi(company, ticker, frm, to, newsapi_key, page_size=100, max_pages=args.per_provider_limit)
            provider_counts["newsapi"] += len(rows); all_rows.extend(rows)
        except requests.HTTPError as e:
            print(f"[newsapi] {ticker}: {e}")

        # GNews (query-based)
        try:
            rows = fetch_gnews(company, ticker, frm, to, gnews_key, max_items=100)
            provider_counts["gnews"] += len(rows); all_rows.extend(rows)
        except requests.HTTPError as e:
            print(f"[gnews] {ticker}: {e}")

    df = _norm_df(all_rows)
    # Always write a file, even if empty (stable schema)
    if df.empty:
        pd.DataFrame(columns=SCHEMA).to_parquet(args.out, index=False)
        print(f"Saved EMPTY (no results) → {args.out}")
    else:
        df.to_parquet(args.out, index=False)
        print(f"Saved → {args.out}  rows={len(df)}  by_provider={provider_counts}")

if __name__ == "__main__":
    main()
