from __future__ import annotations
import pandas as pd, numpy as np
from typing import Dict, Any

def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    d = series.diff(); g = d.clip(lower=0); l = -d.clip(upper=0)
    ag = g.ewm(alpha=1/window, adjust=False).mean(); al = l.ewm(alpha=1/window, adjust=False).mean()
    rs = ag / (al + 1e-9); return 100 - (100 / (1 + rs))

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _macd(series: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ef = _ema(series, fast); es = _ema(series, slow); m = ef - es; s = _ema(m, signal); h = m - s
    return pd.DataFrame({"macd": m, "macd_signal": s, "macd_hist": h})

def build_feature_frame(close: pd.Series, cfg: Dict[str, Any]) -> pd.DataFrame:
    #close = close.astype(float); df = pd.DataFrame({"close": close}); df["ret"] = close.pct_change() # runtime error:astype
    close = pd.to_numeric(close, errors="coerce") # fixed: astype
    close = close.dropna().astype(float)
    df = pd.DataFrame({"close": close})
    df["ret"] = close.pct_change()
    for L in cfg.get("features", {}).get("r_lags", [1,2,3,5]): df[f"ret_lag{L}"] = df["ret"].shift(L)
    for W in cfg.get("features", {}).get("sma_windows", [5,10,20,50]): df[f"sma_{W}"] = close.rolling(W).mean() / close - 1.0
    for W in cfg.get("features", {}).get("rsi_windows", [14]): df[f"rsi_{W}"] = _rsi(close, W)
    m = cfg.get("features", {}).get("macd", {"fast":12,"slow":26,"signal":9})
    df = pd.concat([df, _macd(close, m.get("fast",12), m.get("slow",26), m.get("signal",9))], axis=1)
    for W in cfg.get("features", {}).get("vol_windows", [10,20]): df[f"vol_{W}"] = df["ret"].rolling(W).std() * np.sqrt(252)
    df.dropna(inplace=True); return df
