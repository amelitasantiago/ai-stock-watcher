import sqlite3
import pandas as pd
conn = sqlite3.connect("C:/Users/ameli/ai_stock_watcher/src/datacollectpipeline/my_stock_data/stock_data.db")
print(pd.read_sql_query("SELECT DISTINCT ticker FROM stock_prices", conn))
df = pd.read_sql_query("SELECT * FROM stock_prices WHERE ticker='AAPL' LIMIT 5", conn)
print(df)
conn.close()