import pandas as pd
news = pd.read_parquet("data/curated/news_last_30d.parquet")
print(news.provider.value_counts())
print(news.ticker.value_counts())
print("Missing published_utc:", news['published_utc'].isna().mean())

sent = pd.read_parquet("data/curated/daily_sentiment_last_30d.parquet")
print(sent.head())
