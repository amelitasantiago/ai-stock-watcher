from __future__ import annotations
import datetime as dt
from typing import List, Dict, Any, Tuple
from pathlib import Path

import pandas as pd
import requests, feedparser, yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer

OUT_DIR = Path("data/sentiment")

def _to_date(ts: Any) -> dt.date | None:
    try:
        if isinstance(ts, (int, float)) and ts > 0:
            return dt.date.fromtimestamp(int(ts))
        if isinstance(ts, str) and ts.isdigit():
            return dt.date.fromtimestamp(int(ts))
    except Exception:
        pass
    return None

def _dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set(); out = []
    for r in rows:
        k = (r.get("date"), (r.get("title") or "").strip())
        if not k[1] or k in seen: continue
        seen.add(k); out.append(r)
    return out

def _yf_news_rows(ticker: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        for n in (yf.Ticker(ticker).news or []):
            title = n.get("title") or n.get("headline") or ""
            ts = n.get("providerPublishTime") or n.get("providerPublishDate") or n.get("published")
            d = _to_date(ts)
            if title and d: rows.append({"date": d, "title": title})
    except Exception:
        pass
    return rows

def _yahoo_rss_rows(ticker: str, max_items: int = 300) -> List[Dict[str, Any]]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    rows: List[Dict[str, Any]] = []
    try:
        feed = feedparser.parse(url)
        for e in feed.entries[:max_items]:
            title = e.get("title", "")
            pp = e.get("published_parsed")
            if not title or not pp: continue
            d = dt.date(pp.tm_year, pp.tm_mon, pp.tm_mday)
            rows.append({"date": d, "title": title})
    except Exception:
        pass
    return rows

def _gdelt_rows(ticker: str, timespan_days: int = 120, max_records: int = 250) -> List[Dict[str, Any]]:
    """
    GDELT 2.0 Doc API (no key). Returns English news mentioning the ticker.
    Docs: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
    """
    rows: List[Dict[str, Any]] = []
    try:
        params = {
            "query": ticker,               # simple query; you can add quotes if you like: f'"{ticker}"'
            "mode": "ArtList",
            "maxrecords": str(max_records),
            "timespan": f"{int(timespan_days)}d",
            "format": "json",
        }
        r = requests.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        for art in js.get("articles", []):
            title = art.get("title") or ""
            # seendate is like "20251020T235000Z" -> parse safely
            sd = art.get("seendate")
            date = None
            if isinstance(sd, str) and len(sd) >= 8:
                try:
                    date = dt.datetime.strptime(sd[:8], "%Y%m%d").date()
                except Exception:
                    pass
            if title and date:
                rows.append({"date": date, "title": title})
    except Exception:
        pass
    return rows

def _score_daily(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date","pos","neu","neg","compound","model","n_items"])
    df = pd.DataFrame(rows)
    sid = SentimentIntensityAnalyzer()
    sc = df["title"].astype(str).map(sid.polarity_scores).apply(pd.Series)
    df = pd.concat([df[["date"]], sc], axis=1)
    daily = df.groupby("date")[["pos","neu","neg","compound"]].mean().sort_index()
    daily["model"] = "VADER"
    daily["n_items"] = df.groupby("date").size().reindex(daily.index).astype(int).values
    return daily.reset_index()

def collect_and_score(
    ticker: str,
    window_days: int = 120,
    max_items_rss: int = 300,
    max_records_gdelt: int = 250,
    sources: Tuple[str, ...] = ("yfinance", "yahoo_rss", "gdelt"),
) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ticker = ticker.upper()
    cutoff = dt.date.today() - dt.timedelta(days=int(window_days))

    rows: List[Dict[str, Any]] = []
    if "yfinance" in sources:
        rows += _yf_news_rows(ticker)
    if "yahoo_rss" in sources:
        rows += _yahoo_rss_rows(ticker, max_items=max_items_rss)
    if "gdelt" in sources:
        rows += _gdelt_rows(ticker, timespan_days=window_days, max_records=max_records_gdelt)

    rows = [r for r in rows if r.get("date") and r["date"] >= cutoff and r.get("title")]
    rows = _dedupe(rows)

    out = _score_daily(rows)
    out.to_csv(OUT_DIR / f"{ticker}_daily.csv", index=False)
    return out
