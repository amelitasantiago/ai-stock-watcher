import pandas as pd, numpy as np
def inverse_rmse_weights(pred_df: pd.DataFrame, y_true: pd.Series, window: int = 60) -> pd.Series:
    recent = pred_df.iloc[-window:]; yt = y_true.reindex(recent.index)
    rmses = ((recent.sub(yt, axis=0))**2).mean().pow(0.5)
    inv = 1.0 / (rmses + 1e-9); return inv / inv.sum()
def blend(pred_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weights = weights.reindex(pred_df.columns).fillna(0.0); return (pred_df * weights).sum(axis=1)
