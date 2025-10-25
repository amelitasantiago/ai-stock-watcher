from __future__ import annotations
import runpy
from pathlib import Path
LEGACY_DIR = Path(__file__).parent
def run_sentiment(): runpy.run_path(str(LEGACY_DIR / "financialSentimentAnalysis.py"), run_name="__main__")
def run_collect():   runpy.run_path(str(LEGACY_DIR / "YFinanceStockDataCollection.py"), run_name="__main__")
def run_train():     runpy.run_path(str(LEGACY_DIR / "multiModelStocksNightsWatch_v2.py"), run_name="__main__")
