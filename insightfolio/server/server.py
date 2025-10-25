import json
from pathlib import Path
from typing import Optional
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import HTTPException
from functools import lru_cache
from pydantic import BaseModel
from datetime import datetime



# Resolve paths:
# ROOT/insightfolio/server/server.py
HERE    = Path(__file__).resolve().parent              # .../insightfolio/server
APPDIR  = HERE.parent                                  # .../insightfolio
ROOT    = APPDIR.parent                                # project root
WEB     = APPDIR / "web"                               # .../insightfolio/web
DATA    = ROOT / "data"
PRICE_DIR = DATA / "my_stock_data"
BT_DIR    = DATA / "backtests"
SIG_DIR   = DATA / "signals"

STRAT_DIR = Path("data/strategies")
STRAT_DIR.mkdir(parents=True, exist_ok=True)
STRAT_FILE = STRAT_DIR / "trade_plans.jsonl"

class TradePlan(BaseModel):
    ticker: str
    action: str            # attack | defend | retreat
    note: str = ""
    price: float | None = None
    horizon: int | None = None
    confidence: float | None = None
    ts: str | None = None

def _read_plans():
    items=[]
    if STRAT_FILE.exists():
        with STRAT_FILE.open("r", encoding="utf-8") as f:
            for ln in f:
                ln=ln.strip()
                if ln:
                    try: items.append(json.loads(ln))
                    except: pass
    return items

def _append_plan(obj: dict):
    with STRAT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _write_plan(obj: dict):
    with STRAT_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ===== FastAPI server =====
app = FastAPI(title="Nightwatch API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/tickers")
def api_tickers():
    tickers = set()
    tickers |= {p.name.split("_")[0] for p in SIG_DIR.glob("*_signals.csv")}
    tickers |= {p.name.split("_")[0] for p in BT_DIR.glob("*_equity.csv")}
    tickers |= {p.name.split("_")[0] for p in PRICE_DIR.glob("*_1d.csv")}
    return {"tickers": sorted(tickers)}

@app.get("/api/holdings")
def api_holdings(
    tickers: str = Query(""),
    horizon: int = Query(5),
    sparklen: int = Query(60),
):
    req = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    holdings = []

    for t in req:
        prices = load_prices(t)
        if prices is None or prices.empty:
            continue

        close = pick_close(prices).dropna().astype(float)
        if len(close) < 2:
            continue

        price = float(close.iloc[-1])
        prev = float(close.iloc[-2])
        change_abs = price - prev
        change_pct = (price / prev - 1.0) if prev else 0.0
        spark = [float(x) for x in close.tail(sparklen).tolist()]

        sig = last_signal(t) or {}
        holdings.append(
            {
                "ticker": t,
                "company": t,  # (optional: enrich with name later)
                "price": price,
                "change_abs": change_abs,
                "change_pct": change_pct,
                "sparkline": spark,
                "sentiment": sig.get("signal", "HOLD"),
                "horizon_days": int(sig.get("horizon_days", horizon)),
                "updated": str(close.index[-1].date()),
            }
        )

    return {"holdings": holdings}

# --- endpoints ---
@app.get("/api/strategies")
def list_strategies(limit: int = 100):
    items = list(reversed(_read_plans()))
    return {"items": items[:limit], "count": len(items)}

@app.post("/api/trade-plan")
def save_trade_plan(plan: TradePlan):
    obj = plan.dict()
    obj["ts"] = obj.get("ts") or datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _write_plan(obj)
    return {"ok": True, "saved": obj}

@app.delete("/api/trade-plan")
def clear_trade_plans():
    # simple bulk-clear helper (optional)
    if STRAT_FILE.exists():
        STRAT_FILE.unlink()
    return {"ok": True}


def _load_backtest(ticker: str):
    eq_path = BT_DIR / f"{ticker}_equity.csv"
    mt_path = BT_DIR / f"{ticker}_metrics.csv"
    js_path = BT_DIR / f"{ticker}_pnl_summary.json"
    if not eq_path.exists():
        return None, None, None

    # ------------- read equity file (robust to different headers) -------------
    df = pd.read_csv(eq_path)

    # Pick a date column: prefer 'Date'/'Timestamp'; else first column (often unnamed)
    colmap = {c.lower().strip(): c for c in df.columns}
    date_col = None
    for key in ("date", "timestamp", "time", "unnamed: 0"):
        if key in colmap:
            date_col = colmap[key]
            break
    if date_col is None:
        date_col = df.columns[0]  # fallback: first column

    # Parse dates and index
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).rename(columns={date_col: "Date"}).set_index("Date").sort_index()

    # Choose a returns/equity column and build equity curve
    ret_col = next((c for c in ["net_ret", "strategy_ret", "ret", "return"] if c in df.columns), None)
    if ret_col:
        r = pd.to_numeric(df[ret_col], errors="coerce").fillna(0.0)
        equity = (1.0 + r).cumprod()
    else:
        # some exports already store an equity curve or cumulative return
        eq_col = next((c for c in ["equity", "equity_curve", "cumret", "cum_ret"] if c in df.columns), None)
        if eq_col:
            equity = pd.to_numeric(df[eq_col], errors="coerce").ffill().fillna(1.0)
        else:
            # graceful empty response (UI will show “No metrics.”)
            equity = pd.Series([], dtype=float)

    # ------------- read metrics + summary (if present) -------------
    metrics = []
    if mt_path.exists():
        try:
            metrics = pd.read_csv(mt_path).to_dict(orient="records")
        except Exception:
            metrics = []

    pnl_summary = {}
    if js_path.exists():
        try:
            pnl_summary = json.loads(js_path.read_text(encoding="utf-8"))
        except Exception:
            pnl_summary = {}

    return equity, metrics, pnl_summary


@app.get("/api/backtest/{ticker}/equity")
def api_backtest_equity(ticker: str):
    equity, _, _ = _load_backtest(ticker.upper())
    if equity is None: return {"ticker": ticker.upper(), "dates": [], "equity": []}
    dates = [d.date().isoformat() for d in equity.index]
    return {"ticker": ticker.upper(), "dates": dates, "equity": equity.tolist()}

@app.get("/api/backtest/{ticker}/metrics")
def api_backtest_metrics(ticker: str):
    _, metrics, pnl_summary = _load_backtest(ticker.upper())
    return {"ticker": ticker.upper(), "pnl_summary": pnl_summary or {}, "metrics": metrics or []}

@lru_cache(maxsize=256)
def _read_sentiment_csv_cached(path: str, mtime: float):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["date"]).sort_values("date")
    return {
        "dates": [d.date().isoformat() for d in df["date"]],
        "compound": pd.to_numeric(df["compound"], errors="coerce").fillna(0).round(4).tolist(),
        "n_items": df.get("n_items", pd.Series([None]*len(df))).tolist(),
    }

@app.get("/api/sentiment/{ticker}")
def api_sentiment(ticker: str):
    p = DATA / "sentiment" / f"{ticker.upper()}_daily.csv"
    if not p.exists():
        return {"ticker": ticker.upper(), "dates": [], "compound": [], "n_items": []}
    data = _read_sentiment_csv_cached(str(p), p.stat().st_mtime)  # mtime busts the cache when file changes
    data["ticker"] = ticker.upper()
    return data

@app.get("/api/ping")
def ping(): return {"ok": True}

@app.get("/api/strategies")
def api_strategies(limit: int = 100):
    items = list(reversed(_read_plans()))
    return {"items": items[:limit], "count": len(items)}

@app.post("/api/trade-plan")
def api_trade_plan(plan: TradePlan):
    obj = plan.dict()
    obj["ts"] = obj.get("ts") or datetime.utcnow().isoformat(timespec="seconds") + "Z"
    _append_plan(obj)
    return {"ok": True, "saved": obj}

# ---- News API (headlines + quick VADER score) ----
import feedparser, yfinance as yf
from nltk.sentiment.vader import SentimentIntensityAnalyzer

_sid = SentimentIntensityAnalyzer()

def _sent_label(x: float) -> str:
    return "Positive" if x > 0.05 else "Negative" if x < -0.05 else "Neutral"

@app.get("/api/news/{ticker}")
def api_news(ticker: str, limit: int = 6):
    t = ticker.upper()
    items = []

    # 1) yfinance news (if available)
    try:
        for n in (yf.Ticker(t).news or []):
            title = (n.get("title") or "").strip()
            if not title:
                continue
            comp = float(_sid.polarity_scores(title)["compound"])
            items.append({
                "title": title,
                "source": n.get("publisher") or "Yahoo",
                "compound": round(comp, 3),
                "label": _sent_label(comp),
            })
    except Exception:
        pass

    # 2) Yahoo RSS fallback
    if len(items) < limit:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
        try:
            feed = feedparser.parse(url)
            for e in feed.entries:
                title = (e.get("title") or "").strip()
                if not title:
                    continue
                comp = float(_sid.polarity_scores(title)["compound"])
                items.append({
                    "title": title,
                    "source": (getattr(e, "source", None) or "Yahoo"),
                    "compound": round(comp, 3),
                    "label": _sent_label(comp),
                })
                if len(items) >= limit:
                    break
        except Exception:
            pass

    return {"ticker": t, "items": items[:limit]}


# ===== Last signal endpoint =====
from pathlib import Path
import pandas as pd

@app.get("/api/signal/{ticker}")
def api_signal(ticker: str):
    t = ticker.upper()
    p = DATA / "signals" / f"{t}_signals.csv"
    if not p.exists():
        return {"ticker": t, "date": None, "signal": None, "pred": None, "uncertainty": None}

    df = pd.read_csv(p)
    if df.empty:
        return {"ticker": t, "date": None, "signal": None, "pred": None, "uncertainty": None}

    row = df.iloc[-1]
    # best-effort column detection
    date_col = next((c for c in df.columns if c.lower() in ("date","timestamp","time")), None)
    sig_col  = next((c for c in df.columns if "signal" in c.lower() or "decision" in c.lower()), None)

    date = str(row[date_col]) if date_col else None
    signal = (str(row[sig_col]).upper() if sig_col and pd.notna(row[sig_col]) else None)
    pred = None
    for k in ("pred_ens","pred","score","prob"):
        if k in df.columns and pd.notna(row.get(k)):
            pred = float(row[k]); break
    uncertainty = None
    for k in ("uncertainty","sigma","std"):
        if k in df.columns and pd.notna(row.get(k)):
            uncertainty = float(row[k]); break

    return {"ticker": t, "date": date, "signal": signal, "pred": pred, "uncertainty": uncertainty}


# --- News items (flexible loader) ---
import pandas as pd
from pathlib import Path

def _read_news_csv(t):
    t = t.upper()
    cands = [
        DATA / "news" / f"{t}_news.csv",
        DATA / "sentiment" / f"{t}_news.csv",
        DATA / "sentiment" / f"{t}_daily_news.csv",
    ]
    for p in cands:
        if p.exists():
            df = pd.read_csv(p)
            return df
    # fallback: no news file
    return pd.DataFrame(columns=["date","title","source","url","label","score"])

@app.get("/api/news/{ticker}")
def api_news(ticker: str):
    df = _read_news_csv(ticker)
    cols = {c.lower(): c for c in df.columns}
    out = []
    for _, r in df.sort_values(cols.get("date", df.columns[0])).tail(200).iterrows():
        out.append({
            "date": str(r.get(cols.get("date"), "")),
            "title": str(r.get(cols.get("title"), "")),
            "source": str(r.get(cols.get("source"), "")),
            "url": str(r.get(cols.get("url"), "")),
            "label": str(r.get(cols.get("label"), "")) or ("positive" if float(r.get(cols.get("score"), 0))>0 else "negative" if float(r.get(cols.get("score"), 0))<0 else "neutral"),
            "score": float(r.get(cols.get("score"), 0) or 0),
        })
    return {"ticker": ticker.upper(), "items": out}


CAL_DIR = Path("data/models/calibration")

@app.get("/api/calibration/{ticker}")
def get_calibration(ticker: str):
    p = CAL_DIR / f"{ticker.upper()}.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Calibration not found")
    return json.loads(p.read_text())


# Serve the static Nightwatch UI (your index.html) at /
# Any relative assets (app.js, css, images) are served automatically.
app.mount("/", StaticFiles(directory=WEB, html=True), name="web")

def pick_close(prices: pd.DataFrame) -> pd.Series:
    for c in ["Close", "Adj Close", "close", "adj_close"]:
        if c in prices.columns:
            s = pd.to_numeric(prices[c], errors="coerce")
            if s.notna().sum() > 0:
                return s
    num = prices.select_dtypes(include="number")
    if not num.empty:
        return num.iloc[:, 0]
    raise ValueError(f"No numeric price column: {list(prices.columns)}")

def load_prices(ticker: str) -> Optional[pd.DataFrame]:
    path = PRICE_DIR / f"{ticker}_1d.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    return df

def last_signal(ticker: str) -> Optional[dict]:
    path = SIG_DIR / f"{ticker}_signals.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[-1].to_dict()
    return {
        "signal": str(row.get("signal", "HOLD")).upper(),
        "horizon_days": int(row.get("horizon_days", 5)),
        "pred_ensemble": float(row.get("pred_ensemble", 0.0)),
    }


