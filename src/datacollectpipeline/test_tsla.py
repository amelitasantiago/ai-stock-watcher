# Save as test_tsla.py
import pickle, pandas as pd, sqlite3, numpy as np
conn = sqlite3.connect('data/my_stock_data/stock_data.db')
recent = pd.read_sql("SELECT * FROM stock_prices WHERE ticker='TSLA' ORDER BY date DESC LIMIT 70", conn)
with open('models/hybrid_model_TSLA.pkl', 'rb') as f:
    pred = pickle.load(f)
features = pred.prepare_features(recent)  # Adjust if method differs
forecasts = pred.ensemble_predict(features, horizon=5)  # Assume method
print(pd.DataFrame({
    'Day': [f'+{i}d' for i in range(1,6)],
    'Pred_Close': forecasts.flatten().round(2)
}))
# Ex: +1d: 245.67 (sentiment ~0.3, volatile TSLA)