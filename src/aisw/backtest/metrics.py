import pandas as pd, numpy as np
def compute_positions(pred, mu: float = 0.01):
    import numpy as np, pandas as pd
    pos = np.where(pred>mu,1.0,np.where(pred<-mu,-1.0,0.0)); return pd.Series(pos, index=pred.index, dtype=float)
def apply_costs(positions, cost_bps: float = 5.0):
    chg = positions.diff().abs().fillna(abs(positions.iloc[0])); return (cost_bps/10000.0) * chg
def strategy_returns(y_true, positions, cost_bps: float, horizon_days: int):
    gross = positions * y_true; costs = apply_costs(positions, cost_bps=cost_bps); net = gross - costs
    return pd.DataFrame({"position":positions,"gross_ret":gross,"costs":costs,"net_ret":net})
def summarize_pnl(net_ret, start_date, end_date, horizon_days: int):
    eq = (1.0 + net_ret).cumprod(); total_ret = float(eq.iloc[-1]-1.0)
    elapsed_days = max((end_date - start_date).days, 1); cagr = (1.0 + total_ret) ** (365.0/elapsed_days) - 1.0
    mu = float(net_ret.mean()); sd = float(net_ret.std(ddof=1)+1e-12); ppy = 252.0/max(horizon_days,1)
    import numpy as np
    sharpe = (mu*ppy)/(sd*np.sqrt(ppy)); roll_max = eq.cummax(); dd=(eq/roll_max)-1.0; maxdd=float(dd.min())
    return {"total_return": total_ret, "CAGR": float(cagr), "Sharpe": float(sharpe), "MaxDD": float(maxdd), "periods": int(len(net_ret))}
