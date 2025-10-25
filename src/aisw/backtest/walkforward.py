import pandas as pd, numpy as np
from ..models.naive import NaiveReturn
from ..models.arima import ARIMAWrapper
from ..models.rf import RFLagged
from ..ensemble.weighted import inverse_rmse_weights, blend
from .metrics import compute_positions, strategy_returns, summarize_pnl
def walkforward_backtest(ticker, close, feats, horizon=5, window=252, step=21):
    y = close.shift(-horizon)/close - 1.0; df = feats.join(y.rename("y")).dropna(); preds=[]
    for start in range(0, len(df)-window-horizon, step):
        tr = df.iloc[start:start+window]; te = df.iloc[start+window:start+window+step]
        if len(te)==0: break
        m1=NaiveReturn().fit(tr.drop(columns=["y"]), tr["y"]); m2=ARIMAWrapper(seasonal=False,m=5).fit(tr.drop(columns=["y"]), tr["y"]); m3=RFLagged().fit(tr.drop(columns=["y"]), tr["y"])
        P_tr = pd.DataFrame({m1.name:m1.predict(tr.drop(columns=["y"])), m2.name:m2.predict(tr.drop(columns=["y"])), m3.name:m3.predict(tr.drop(columns=["y"]))}, index=tr.index)
        W = inverse_rmse_weights(P_tr, tr["y"], window=min(60,len(tr)))
        P = pd.DataFrame({m1.name:m1.predict(te.drop(columns=["y"])), m2.name:m2.predict(te.drop(columns=["y"])), m3.name:m3.predict(te.drop(columns=["y"]))}, index=te.index)
        P["ensemble"] = blend(P[[m1.name,m2.name,m3.name]], W); P["y_true"]=te["y"]; preds.append(P)
    preds = pd.concat(preds).sort_index()
    def _rmse(a,b): import numpy as np; return float(np.sqrt(((a-b)**2).mean()))
    def _mae(a,b): return float((a-b).abs().mean())
    def _hit(a,b): return float(((a*b)>0).mean())
    rows=[]; 
    for col in ["naive_prev_ret","arima_auto","rf_lagged","ensemble"]:
        if col in preds.columns: rows.append({"ticker":ticker,"model":col,"rmse":_rmse(preds[col],preds["y_true"]),"mae":_mae(preds[col],preds["y_true"]),"hit_rate":_hit(preds[col],preds["y_true"]),"n":int(preds[col].notna().sum())})
    metrics = pd.DataFrame(rows).set_index(["ticker","model"])
    mu=0.01; cost_bps=5.0
    if "ensemble" in preds.columns:
        pos = compute_positions(preds["ensemble"], mu=mu); pnl_df = strategy_returns(preds["y_true"], pos, cost_bps=cost_bps, horizon_days=horizon); pnl_metrics = summarize_pnl(pnl_df["net_ret"], preds.index.min(), preds.index.max(), horizon_days=horizon)
    else:
        pnl_df = pd.DataFrame(index=preds.index, columns=["position","gross_ret","costs","net_ret"]).fillna(0.0); pnl_metrics={"total_return":0.0,"CAGR":0.0,"Sharpe":0.0,"MaxDD":0.0,"periods":0}
    return {"preds":preds,"metrics":metrics,"pnl":pnl_df,"pnl_summary":pnl_metrics}
