from __future__ import annotations
import argparse, json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from . import config as _cfg
from .data.market_data import fetch_daily_bars
from .features.technical import build_feature_frame
from .backtest.walkforward import walkforward_backtest
from .signals.rules import generate_signals
from .models.registry import save_model
from .models.naive import NaiveReturn
from .models.arima import ARIMAWrapper
from .models.rf import RFLagged
from .ensemble.weighted import inverse_rmse_weights
from .calibration.calibrate import run as calibrate_run


def _load_cfg(cfg_path: str | None) -> _cfg.Cfg:
    return _cfg.load(cfg_path or "config/config.yaml")

def cmd_collect(args):
    cfg = _load_cfg(args.config); tickers = args.tickers or cfg.tickers
    for t in tqdm(tickers, desc="Collect"): fetch_daily_bars(t, cfg.price_cache_dir, lookback=args.lookback or cfg.day_bar.get("lookback", 1000))
    print("Done.")

def cmd_featurize(args):
    cfg = _load_cfg(args.config); tickers = args.tickers or cfg.tickers; out_dir = Path(cfg.price_cache_dir)
    for t in tqdm(tickers, desc="Featurize"):
        prices = pd.read_csv(out_dir / f"{t}_1d.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        
        # pick a numeric close-like column
        close = None
        for candidate in ["Close", "Adj Close", "close", "adj_close"]:
            if candidate in prices.columns:
                close = prices[candidate]
                break
        if close is None and hasattr(prices, "columns") and prices.select_dtypes(include="number").shape[1] > 0:
            # fallback: first numeric column
            close = prices.select_dtypes(include="number").iloc[:, 0]
        if close is None:
            raise ValueError(f"No numeric price column found in {t}_1d.csv; columns={list(prices.columns)}")

        close = pd.to_numeric(close, errors="coerce").dropna().astype(float)

        feats = build_feature_frame(close, cfg.forecast)

        # feats = build_feature_frame(prices["Close"], cfg.forecast); feats.to_csv(out_dir / f"{t}_features.csv", index=True)
        feats.to_csv(out_dir / f"{t}_features.csv", index=True)
        print("Featurization complete.")

def cmd_backtest(args):
    cfg = _load_cfg(args.config); tickers = args.tickers or cfg.tickers; out_dir = Path(cfg.data_dir) / "backtests"; out_dir.mkdir(parents=True, exist_ok=True)
    for t in tqdm(tickers, desc="Backtest"):
        #prices = pd.read_csv(Path(cfg.price_cache_dir) / f"{t}_1d.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        #feats = pd.read_csv(Path(cfg.price_cache_dir) / f"{t}_features.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        #res = walkforward_backtest(t, prices["Close"], feats, horizon=args.horizon or cfg.forecast["horizon_days"], window=args.window or cfg.backtest["window_days"], step=args.step or cfg.backtest["step_days"])
        
        prices = pd.read_csv(Path(cfg.price_cache_dir) / f"{t}_1d.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        feats  = pd.read_csv(Path(cfg.price_cache_dir) / f"{t}_features.csv", parse_dates=["Date"]).set_index("Date").sort_index()

        # choose a numeric close series
        close = None
        for candidate in ["Close", "Adj Close", "close", "adj_close"]:
            if candidate in prices.columns:
                close = prices[candidate]; break
        if close is None and prices.select_dtypes(include="number").shape[1] > 0:
            close = prices.select_dtypes(include="number").iloc[:, 0]
        if close is None:
            raise ValueError(f"No numeric price column found in {t}_1d.csv; columns={list(prices.columns)}")

        close = pd.to_numeric(close, errors="coerce").dropna().astype(float)

        res = walkforward_backtest(
            t, close, feats,
            horizon=args.horizon or cfg.forecast["horizon_days"],
            window=args.window or cfg.backtest["window_days"],
            step=args.step or cfg.backtest["step_days"],
        )
        
        res["preds"].to_csv(out_dir / f"{t}_preds.csv"); res["metrics"].to_csv(out_dir / f"{t}_metrics.csv"); res["pnl"].to_csv(out_dir / f"{t}_equity.csv")
        with open(out_dir / f"{t}_pnl_summary.json", "w", encoding="utf-8") as f: json.dump(res["pnl_summary"], f, indent=2)
    print(f"Backtests saved to {out_dir}")

def cmd_signal(args):
    cfg = _load_cfg(args.config)
    tickers = args.tickers or cfg.tickers
    out_dir = Path(cfg.data_dir) / "signals"
    out_dir.mkdir(parents=True, exist_ok=True)

    for t in tqdm(tickers, desc="Signals"):
        prices = pd.read_csv(Path(cfg.price_cache_dir) / f"{t}_1d.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        feats  = pd.read_csv(Path(cfg.price_cache_dir) / f"{t}_features.csv", parse_dates=["Date"]).set_index("Date").sort_index()

        # choose a numeric close series
        close = None
        for candidate in ["Close", "Adj Close", "close", "adj_close"]:
            if candidate in prices.columns:
                close = prices[candidate]; break
        if close is None and prices.select_dtypes(include="number").shape[1] > 0:
            close = prices.select_dtypes(include="number").iloc[:, 0]
        if close is None:
            raise ValueError(f"No numeric price column found in {t}_1d.csv; columns={list(prices.columns)}")

        close = pd.to_numeric(close, errors="coerce").dropna().astype(float)

        # generate + save
        sig_df = generate_signals(t, close, feats, horizon=args.horizon or cfg.forecast["horizon_days"])
        sig_df.to_csv(out_dir / f"{t}_signals.csv", index=False)

    print(f"Signals saved to {out_dir}")

def cmd_train(args):
    cfg = _load_cfg(args.config)
    tickers = args.tickers or cfg.tickers
    window = args.window or cfg.backtest.get("window_days", 252)
    horizon = args.horizon or cfg.forecast.get("horizon_days", 5)

    for t in tqdm(tickers, desc="Train"):
        prices = pd.read_csv(Path(cfg.price_cache_dir) / f"{t}_1d.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        feats   = pd.read_csv(Path(cfg.price_cache_dir) / f"{t}_features.csv", parse_dates=["Date"]).set_index("Date").sort_index()

        # choose a numeric close series
        close = None
        for candidate in ["Close", "Adj Close", "close", "adj_close"]:
            if candidate in prices.columns:
                close = prices[candidate]; break
        if close is None and prices.select_dtypes(include="number").shape[1] > 0:
            close = prices.select_dtypes(include="number").iloc[:, 0]
        if close is None:
            raise ValueError(f"No numeric price column found in {t}_1d.csv; columns={list(prices.columns)}")

        close = pd.to_numeric(close, errors="coerce").dropna().astype(float)

        # target + inner align by date (NO .loc on feats)
        y  = close.shift(-horizon) / close - 1.0
        df = feats.join(y.rename("y"), how="inner").dropna()

        tr  = df.iloc[-window:]
        Xtr = tr.drop(columns=["y"])
        ytr = tr["y"]

        m_naive = NaiveReturn().fit(Xtr, ytr)
        m_arima = ARIMAWrapper(seasonal=False, m=5).fit(Xtr, ytr)
        m_rf    = RFLagged().fit(Xtr, ytr)

        P_tr = pd.DataFrame({
            m_naive.name: m_naive.predict(Xtr),
            m_arima.name: m_arima.predict(Xtr),
            m_rf.name:    m_rf.predict(Xtr),
        }, index=Xtr.index)

        W    = inverse_rmse_weights(P_tr, ytr, window=min(60, len(tr)))
        cols = list(Xtr.columns)

        save_model(".", t, m_naive.name, horizon, m_naive, Xtr.index.min(), Xtr.index.max(), cols, window, cfg.backtest.get("step_days", 21))
        save_model(".", t, m_arima.name, horizon, m_arima, Xtr.index.min(), Xtr.index.max(), cols, window, cfg.backtest.get("step_days", 21))
        save_model(".", t, m_rf.name,    horizon, m_rf,    Xtr.index.min(), Xtr.index.max(), cols, window, cfg.backtest.get("step_days", 21), weights=W.to_dict())
    print("Model training complete.")

# handler (ensure it matches)
def cmd_sentiment(args):
    from .sentiment.collect import collect_and_score
    for t in args.tickers:
        df = collect_and_score(t, window_days=args.window_days)
        print(f"{t} saved {len(df)} rows")

def cmd_calibrate(args):
    calibrate_run(args.tickers, horizon=args.horizon)


def main(argv=None):
    p = argparse.ArgumentParser(prog="aisw")
    sub = p.add_subparsers(dest="cmd", required=True)

    # collect
    c = sub.add_parser("collect", help="fetch daily bars")
    c.add_argument("--tickers", nargs="*", default=None)
    c.add_argument("--lookback", type=int, default=None)
    c.add_argument("--config", default=None)
    c.set_defaults(func=cmd_collect)

    # featurize
    fz = sub.add_parser("featurize", help="build ML features")
    fz.add_argument("--tickers", nargs="*", default=None)
    fz.add_argument("--config", default=None)
    fz.set_defaults(func=cmd_featurize)

    # backtest
    bt = sub.add_parser("backtest", help="walk-forward backtest")
    bt.add_argument("--tickers", nargs="*", default=None)
    bt.add_argument("--horizon", type=int, default=None)
    bt.add_argument("--window", type=int, default=None)
    bt.add_argument("--step", type=int, default=None)
    bt.add_argument("--config", default=None)
    bt.set_defaults(func=cmd_backtest)

    # signal
    sg = sub.add_parser("signal", help="generate trading signals")
    sg.add_argument("--tickers", nargs="*", default=None)
    sg.add_argument("--horizon", type=int, default=None)
    sg.add_argument("--config", default=None)
    sg.set_defaults(func=cmd_signal)

    # train
    trn = sub.add_parser("train", help="train base models")
    trn.add_argument("--tickers", nargs="*", default=None)
    trn.add_argument("--horizon", type=int, default=None)
    trn.add_argument("--window", type=int, default=None)
    trn.add_argument("--config", default=None)
    trn.set_defaults(func=cmd_train)

    # sentiment
    sn = sub.add_parser("sentiment", help="collect daily news sentiment")
    sn.add_argument("--tickers", nargs="+", required=True)
    sn.add_argument("--window-days", type=int, default=30)
    sn.add_argument("--config", default=None)
    sn.set_defaults(func=cmd_sentiment)

    # calibrate this is the one you were missing
    cal = sub.add_parser("calibrate", help="fit probability calibration from signals")
    cal.add_argument("--tickers", nargs="+", required=True)
    cal.add_argument("--horizon", type=int, default=5)  # kept for symmetry
    cal.add_argument("--config", default=None)
    cal.set_defaults(func=cmd_calibrate)

    args = p.parse_args(argv)
    return args.func(args)

if __name__ == "__main__": raise SystemExit(main())
