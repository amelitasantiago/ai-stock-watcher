from __future__ import annotations
import os, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class Cfg:
    data_dir: Path
    price_cache_dir: Path
    news_cache_dir: Path
    sentiment_cache_dir: Path
    tickers: list[str]
    day_bar: dict[str, Any]
    forecast: dict[str, Any]
    backtest: dict[str, Any]
    sentiment: dict[str, Any]

def load(path: str | Path) -> Cfg:
    p = Path(path)
    raw = yaml.safe_load(open(p, 'r', encoding='utf-8'))
    def _p(key, env): return Path(os.environ.get(env, raw.get(key)))
    return Cfg(
        data_dir=_p("data_dir","AISW_DATA_DIR"),
        price_cache_dir=_p("price_cache_dir","AISW_PRICE_DIR"),
        news_cache_dir=_p("news_cache_dir","AISW_NEWS_DIR"),
        sentiment_cache_dir=_p("sentiment_cache_dir","AISW_SENT_DIR"),
        tickers=raw.get("tickers", []),
        day_bar=raw.get("day_bar", {}),
        forecast=raw.get("forecast", {}),
        backtest=raw.get("backtest", {}),
        sentiment=raw.get("sentiment", {}),
    )
