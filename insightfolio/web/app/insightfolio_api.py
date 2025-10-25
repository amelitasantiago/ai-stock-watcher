
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from pathlib import Path

from insightfolio_model_utils import discover_artifacts, load_bundle, arima_forecast, ensemble_weights, extract_meta

BASE = Path("/mnt/data")
ARTIFACTS = discover_artifacts(BASE)

app = FastAPI(title="InsightFolio AI — API", version="0.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastResponse(BaseModel):
    ticker: str
    steps: int
    arima: List[float] | None = None
    method: str | None = None
    ensemble_weights: Dict[str, float] | None = None
    note: str | None = None

def _sentiment_from_forecast(preds: Optional[List[float]]) -> str:
    if not preds or len(preds) == 0:
        return "Neutral"
    s = preds[0]
    if s > 0.001:
        return "Bullish"
    if s < -0.001:
        return "Bearish"
    return "Neutral"

def _last_price_from_bundle(bundle: Dict[str, Any]) -> Optional[float]:
    # Best-effort: look for common keys a training script might have saved
    for k in ("last_close","latest_close","close","last_price","recent_close"):
        v = bundle.get(k)
        try:
            if v is not None:
                return float(v)
        except Exception:
            continue
    # If the ARIMA was trained on levels and exposes endog, try the last value
    try:
        arima_obj = bundle.get("arima_model") or bundle.get("arima")
        if hasattr(arima_obj, "data") and hasattr(arima_obj.data, "endog") and len(arima_obj.data.endog) > 0:
            return float(arima_obj.data.endog[-1])
    except Exception:
        pass
    return None

@app.get("/api/tickers")
def get_tickers():
    return sorted(list(ARTIFACTS.keys()))

@app.get("/api/meta/{ticker}")
def get_meta(ticker: str):
    t = ticker.upper()
    if t not in ARTIFACTS:
        raise HTTPException(status_code=404, detail=f"No artifacts for {t}")
    bundle = load_bundle(ARTIFACTS[t]["pkl"])
    meta = extract_meta(bundle)
    meta["has_lstm"] = bool("pth" in ARTIFACTS[t])
    meta["files"] = {k: str(v) for k, v in ARTIFACTS[t].items()}
    return meta

@app.get("/api/forecast/{ticker}", response_model=ForecastResponse)
def forecast(ticker: str, steps: int = 3):
    t = ticker.upper()
    if t not in ARTIFACTS:
        raise HTTPException(status_code=404, detail=f"No artifacts for {t}")
    bundle = load_bundle(ARTIFACTS[t]["pkl"])
    preds, method = arima_forecast(bundle, steps=steps)
    w = ensemble_weights(bundle)
    note = "ARIMA-only demo. Add predict_next(n) in bundle for full ensemble."
    return ForecastResponse(ticker=t, steps=steps, arima=preds, method=method, ensemble_weights=w or None, note=note)

@app.get("/api/dashboard")
def dashboard(steps: int = 7):
    """Aggregate meta + forecasts for all tickers for a portfolio-style view."""
    payload = {"holdings": [], "summary": {}}
    total_value = 0.0
    total_change = 0.0
    gainers = 0
    losers = 0

    for t, paths in ARTIFACTS.items():
        bundle = load_bundle(paths["pkl"])
        meta = extract_meta(bundle)
        preds, _ = arima_forecast(bundle, steps=steps)
        last_price = _last_price_from_bundle(bundle)
        label = _sentiment_from_forecast(preds)

        # Estimate change $ if we have both price and next-step change (assume preds are returns if within [-1,1])
        change_val = None
        if last_price is not None and preds and len(preds) > 0:
            p = preds[0]
            if abs(p) < 1.0:
                change_val = last_price * p
            else:
                change_val = p  # assume it's an absolute level delta

        holding = {
            "ticker": t,
            "company_name": meta.get("company_name", ""),
            "sequence_length": meta.get("sequence_length", None),
            "prediction_horizon": meta.get("prediction_horizon", None),
            "ensemble_weights": meta.get("ensemble_weights", {}),
            "last_price": last_price,
            "forecast": preds or [],
            "sentiment": label,
            "delta1": preds[0] if preds else None,
            "change_val": change_val,
        }
        payload["holdings"].append(holding)

        if last_price is not None:
            total_value += last_price
        if change_val is not None:
            total_change += change_val
            if change_val > 0:
                gainers += 1
            elif change_val < 0:
                losers += 1

    payload["summary"] = {
        "treasure": total_value,
        "fortune_change": total_change,
        "winds_pct": (total_change / total_value * 100.0) if total_value else 0.0,
        "count": len(payload["holdings"]),
        "gainers": gainers,
        "losers": losers,
    }
    return payload
