
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pickle
import numpy as np

# Optional heavy deps guarded
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

def discover_artifacts(base_dir: Path) -> Dict[str, Dict[str, Path]]:
    out: Dict[str, Dict[str, Path]] = {}
    for pkl in base_dir.glob("hybrid_model_*.pkl"):
        ticker = pkl.stem.split("_", 2)[-1]
        pth = base_dir / f"{pkl.stem}_lstm_transformer.pth"
        out[ticker] = {"pkl": pkl}
        if pth.exists():
            out[ticker]["pth"] = pth
    return out

def load_bundle(pkl_path: Path) -> Dict[str, Any]:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj
    # Minimal wrapper
    out = {}
    for k in dir(obj):
        if k.startswith("_"):
            continue
        try:
            v = getattr(obj, k)
        except Exception:
            continue
        if k in ("fit","predict","__class__","__module__"):
            continue
        out[k] = v
    out["__wrapped_obj__"] = obj
    return out

def arima_forecast(bundle: Dict[str, Any], steps: int = 3) -> Tuple[Optional[List[float]], str]:
    # v2 usually stores a statsmodels or pmdarima model under keys like 'arima_model', 'arima'
    arima_obj = bundle.get("arima_model") or bundle.get("arima")
    if arima_obj is None:
        return None, "No ARIMA model found in artifact."
    try:
        if hasattr(arima_obj, "forecast"):
            yhat = arima_obj.forecast(steps=steps)
            return [float(x) for x in np.array(yhat).ravel()], "statsmodels.forecast"
        if hasattr(arima_obj, "predict"):  # pmdarima
            yhat = arima_obj.predict(n_periods=steps)
            return [float(x) for x in np.array(yhat).ravel()], "pmdarima.predict"
        return None, "ARIMA object does not expose forecast/predict."
    except Exception as e:
        return None, f"ARIMA forecast failed: {e}"

def ensemble_weights(bundle: Dict[str, Any]) -> Dict[str, float]:
    w = bundle.get("ensemble_weights") or {}
    if isinstance(w, dict) and w:
        # ensure floats and normalized
        total = sum(float(v) for v in w.values() if v)
        return {k: (float(v)/total if total else 0.0) for k, v in w.items()}
    return {}

def extract_meta(bundle: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["ticker","company_name","feature_columns","selected_features","sequence_length",
            "prediction_horizon","ensemble_weights","model_performance","metrics","report"]
    meta = {}
    for k in keys:
        if k in bundle:
            v = bundle[k]
            if isinstance(v, set):
                v = list(v)
            meta[k] = v
    # useful fallbacks
    if "ticker" not in meta:
        # infer from feature file names if present
        pass
    return meta
