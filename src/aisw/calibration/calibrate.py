from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

CAL_DIR = Path("data/models/calibration")
SIG_DIR = Path("data/signals")
PRICE_DIR = Path("data/my_stock_data")
CAL_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class CalResult:
    ticker: str
    method: str
    n: int
    bins: list[float]
    probs: list[float]

# ---------- utilities ----------
def _pick(df: pd.DataFrame, names: list[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

def _best_numeric_col(df: pd.DataFrame, candidates: list[str], min_ratio: float = 0.6) -> Optional[str]:
    """Return first candidate that is mostly numeric and has variance."""
    for cand in candidates:
        col = _pick(df, [cand])
        if not col:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        ratio = vals.notna().mean()
        if ratio >= min_ratio and (vals.var(skipna=True) > 1e-12):
            df[col] = vals
            return col
    return None

def _load_prices(ticker: str) -> pd.Series:
    p = PRICE_DIR / f"{ticker.upper()}_1d.csv"
    if not p.exists():
        raise FileNotFoundError(f"Price file not found: {p}")
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    for c in ["Close", "Adj Close", "close", "adj_close"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").dropna().astype(float)
            if len(s):
                return s
    num = df.select_dtypes(include="number")
    if num.shape[1] == 0:
        raise ValueError(f"No numeric price column in {p}")
    return pd.to_numeric(num.iloc[:, 0], errors="coerce").dropna().astype(float)

def _asof_join_y(sig_dates: pd.Series, close: pd.Series, horizon: int) -> pd.Series:
    """Compute forward return y using asof alignment (handles weekends/holidays)."""
    # daily calendar with ffill so every day has a value
    cd = close.asfreq("D").ffill()
    y_daily = (cd.shift(-horizon) / cd) - 1.0
    y_df = y_daily.dropna().rename("y").to_frame()
    y_df = y_df.reset_index().rename(columns={"index": "date"})
    # ensure datetime & sorted
    y_df["date"] = pd.to_datetime(y_df["date"])
    y_df = y_df.sort_values("date")
    sig_df = pd.DataFrame({"date": pd.to_datetime(sig_dates, errors="coerce")}).dropna().sort_values("date")
    if sig_df.empty:
        return pd.Series(dtype=float)
    # merge_asof (backward) with 7D tolerance
    out = pd.merge_asof(sig_df, y_df, on="date", direction="backward", tolerance=pd.Timedelta("7D"))
    return out["y"]

def _ensure_y(df_sig: pd.DataFrame, ticker: str, horizon: int) -> Tuple[pd.DataFrame, str]:
    """
    Ensure forward return 'y' exists.

    Order of attempts:
      1) Use an existing numeric y/ret_fwd column (if mostly numeric & with variance).
      2) If a usable date column exists, compute y via as-of alignment with prices.
      3) Fallback (no valid dates): length-match the tail of the price series to the signals rows.
    """
    # 1) existing numeric y
    y_col = _best_numeric_col(df_sig, ["y", "ret_fwd", "forward_return", "ret_forward", "fwd_ret"])
    if y_col:
        return df_sig, y_col

    # 2) try as-of alignment using a date-like column
    date_col = _pick(df_sig, ["date", "timestamp", "ts", "Date", "Timestamp"])
    if date_col:
        dates = pd.to_datetime(df_sig[date_col], errors="coerce")
        # require at least a few valid dates
        if dates.notna().sum() >= max(5, int(0.1 * len(df_sig))):
            close = _load_prices(ticker)
            y_series = _asof_join_y(dates, close, horizon)  # may contain NaNs if very sparse
            df = df_sig.copy()
            # If merge_asof returned fewer rows (shouldn't), align by position
            if len(y_series) != len(df):
                y_series = pd.Series(np.asarray(y_series)[:len(df)], index=df.index)
            df["y"] = y_series.values
            return df, "y"

    # 3) fallback: no usable dates — length-match to price tail
    close = _load_prices(ticker)
    y_all = (close.shift(-horizon) / close) - 1.0
    y_all = y_all.dropna()
    df = df_sig.copy()
    n = len(df)
    if len(y_all) >= n and n > 0:
        df["y"] = y_all.iloc[-n:].values
    elif n > 0:
        # pad with last available y (or zeros if nothing)
        last = float(y_all.iloc[-1]) if len(y_all) else 0.0
        df["y"] = np.full(n, last, dtype=float)
    else:
        df["y"] = np.array([], dtype=float)
    return df, "y"


def _ensure_score(df_sig: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    score_col = _best_numeric_col(
        df_sig,
        ["pred", "score", "margin", "logit", "yhat", "proba", "prob_up", "probability"],
        min_ratio=0.6
    )
    if score_col:
        return df_sig, score_col

    # derive from signal (+/-1/0) * optional confidence/prob
    sig_col = _pick(df_sig, ["signal"])
    conf_col = _best_numeric_col(df_sig, ["confidence", "conf", "prob", "proba", "prob_up"], min_ratio=0.5)
    df = df_sig.copy()
    if sig_col:
        sign_map = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}
        sgn = df[sig_col].astype(str).str.upper().map(sign_map).fillna(0.0).astype(float)
        mag = (df[conf_col].clip(0, 1) if conf_col else 1.0)
        df["pred"] = (sgn * mag).astype(float)
    else:
        df["pred"] = 0.0
    return df, "pred"

# ---------- main loader ----------
def _load_signals(ticker: str, horizon: int) -> pd.DataFrame:
    p = SIG_DIR / f"{ticker.upper()}_signals.csv"
    if not p.exists():
        raise FileNotFoundError(f"Signals not found: {p}")
    df = pd.read_csv(p)

    # ensure outcome
    df, y_col = _ensure_y(df, ticker, horizon)

    # ensure numeric score
    df, score_col = _ensure_score(df)

    # numeric selection and cleanup
    out = df[[score_col, y_col]].copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")
    out[y_col]     = pd.to_numeric(out[y_col], errors="coerce")
    out = out.dropna()
    if out.empty:
        # fallback: if y exists but score is NA, set small jitter so variance>0
        if y_col in df and df[y_col].notna().any():
            tmp = df[[y_col]].copy()
            tmp["score"] = 1e-6  # tiny nonzero
            tmp = tmp.dropna()
            if not tmp.empty:
                out = tmp.rename(columns={y_col: "y"})
        if out.empty:
            raise ValueError("No valid numeric rows for calibration after cleaning.")

    s = out[score_col if score_col in out.columns else "score"].astype(float).values
    y = out[y_col if y_col in out.columns else "y"].astype(float).values

    hit = (np.sign(s) == np.sign(y)) & (np.abs(y) > 0)
    pos = (y > 0).astype(int)

    return pd.DataFrame({
        "score": s,
        "abs_score": np.abs(s),
        "hit": hit.astype(int),
        "pos": pos
    })

# ---------- model fitting ----------
def _fit_isotonic(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(np.abs(x))
    xs, ys = np.abs(x)[order], y[order]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(xs, ys)
    grid = np.linspace(xs.min(), xs.max(), 11)
    pg = iso.predict(grid)
    return grid, pg

def _fit_logistic(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x.reshape(-1, 1), y)
    grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 25)
    pg = lr.predict_proba(grid.reshape(-1, 1))[:, 1]
    return grid, pg

def calibrate_ticker(ticker: str, horizon: int) -> CalResult:
    df = _load_signals(ticker, horizon)
    n = len(df)
    if n < 50:
        return CalResult(ticker.upper(), "flat", n, [0, 1], [0.5, 0.5])

    # prefer isotonic on correctness vs |score|; fallback logistic on direction vs signed score
    try:
        g, p = _fit_isotonic(df["score"].values, df["hit"].values)
        method = "isotonic"
    except Exception:
        g, p = _fit_logistic(df["score"].values, df["pos"].values)
        method = "logistic"

    probs = [float(min(0.99, max(0.01, v))) for v in p.tolist()]
    return CalResult(ticker.upper(), method, n, g.tolist(), probs)

def save_calibration(res: CalResult) -> None:
    out = CAL_DIR / f"{res.ticker}.json"
    payload = {
        "ticker": res.ticker,
        "method": res.method,
        "n": res.n,
        "bins": res.bins,
        "probs": res.probs,
        "version": 1,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

def run(tickers: List[str], horizon: int = 5) -> None:
    for t in tickers:
        try:
            res = calibrate_ticker(t, horizon=horizon)
            save_calibration(res)
            print(f"{t} calibrated with {res.method} (n={res.n}) → data/models/calibration/{t.upper()}.json")
        except Exception as e:
            print(f"{t} skipped: {e}")
