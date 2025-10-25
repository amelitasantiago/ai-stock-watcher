import os
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Nightwatch – AI Stock Watcher", layout="wide")

DATA_DIR = Path("data")
BT_DIR = DATA_DIR / "backtests"
SIG_DIR = DATA_DIR / "signals"

def _list_tickers():
    tickers = set()
    if BT_DIR.exists():
        for p in BT_DIR.glob("*_equity.csv"):
            tickers.add(p.name.split("_")[0])
    if SIG_DIR.exists():
        for p in SIG_DIR.glob("*_signals.csv"):
            tickers.add(p.name.split("_")[0])
    return sorted(tickers) or ["AMZN","NVDA","TSLA","META","JPM","NFLX"]

def load_equity(ticker: str) -> pd.DataFrame:
    p = BT_DIR / f"{ticker}_equity.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    if "net_ret" not in df.columns:
        return pd.DataFrame()
    df["equity"] = (1.0 + df["net_ret"]).cumprod()
    return df

def load_metrics(ticker: str) -> dict:
    j = BT_DIR / f"{ticker}_pnl_summary.json"
    if j.exists():
        try:
            return json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def load_model_metrics(ticker: str) -> pd.DataFrame:
    p = BT_DIR / f"{ticker}_metrics.csv"
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return pd.DataFrame()

def load_signals(ticker: str) -> pd.DataFrame:
    p = SIG_DIR / f"{ticker}_signals.csv"
    if p.exists():
        try:
            df = pd.read_csv(p)
            return df
        except Exception:
            pass
    return pd.DataFrame()

# Sidebar — controls
st.sidebar.title("Nightwatch")
st.sidebar.caption("AI Stock Watcher · single‑chart view")
tickers = _list_tickers()
sel = st.sidebar.selectbox("Ticker", tickers, index=0)
show_table = st.sidebar.checkbox("Show latest signal table", value=True)
st.sidebar.divider()
st.sidebar.write("Run: collect → featurize → train → backtest → signal")

# Header
st.markdown(f"# {sel} · Signal & PnL")
st.caption("One‑page Nightwatch view — equity curve (net), key risk/return metrics, and latest signal.")

# Load data
eq = load_equity(sel)
pnlsum = load_metrics(sel)
metrics = load_model_metrics(sel)
sigt = load_signals(sel)

# KPIs row
kpi_cols = st.columns(5)
def _fmt_pct(x): 
    try:
        return f"{x*100:,.2f}%"
    except Exception:
        return "–"

kpi_cols[0].metric("Total Return", _fmt_pct(pnlsum.get("total_return")))
kpi_cols[1].metric("CAGR", _fmt_pct(pnlsum.get("CAGR")))
kpi_cols[2].metric("Sharpe", f"{pnlsum.get('Sharpe', 0.0):.2f}")
kpi_cols[3].metric("Max Drawdown", _fmt_pct(pnlsum.get("MaxDD")))
if not metrics.empty:
    try:
        hr = float(metrics.loc[metrics["model"]=="ensemble","hit_rate"].values[0])
        kpi_cols[4].metric("Hit Rate (ensemble)", _fmt_pct(hr))
    except Exception:
        kpi_cols[4].metric("Hit Rate (ensemble)", "–")
else:
    kpi_cols[4].metric("Hit Rate (ensemble)", "–")

# Main single chart — equity curve
st.subheader("Equity curve (net of costs)")
if not eq.empty:
    fig = plt.figure()
    ax = plt.gca()
    eq["equity"].plot(ax=ax)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (× initial)")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)
else:
    st.info("No equity curve found yet. Run backtest to generate `data/backtests/*_equity.csv`.")

# Latest signal table
if show_table:
    st.subheader("Latest signal")
    if not sigt.empty:
        st.dataframe(sigt.tail(10), use_container_width=True)
    else:
        st.caption("No signals yet. Run `python -m aisw.cli signal --tickers ...`.")

# Footer
st.divider()
st.caption("Nightwatch · AI Stock Watcher — clean, single‑chart format · © 2025 Amelita Talavera Santiago")
