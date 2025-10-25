import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import sqlite3
import json
import requests
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict
import concurrent.futures
from threading import Lock
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/comprehensive_news_collection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def load_config():
    """Load configuration from config.json"""
    #config_path = Path("src/config/config.json")
    config_path = Path("config/config.json")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open('r') as f:
        return json.load(f)

def load_api_keys():
    """Load API keys from api_keys.json"""
    #api_keys_path = Path("src/config/api_keys.json")
    api_keys_path = Path("config/api_keys.json")
    if not api_keys_path.exists():
        logging.error("API keys file not found: config/api_keys.json")
        return {}
    with api_keys_path.open('r') as f:
        return json.load(f)

@dataclass
class CollectionConfig:
    """Configuration for comprehensive news collection"""
    days_back: int
    max_articles_per_ticker: int = 200
    newsapi_rpm: int = 100
    gnews_rpm: int = 60
    finnhub_rpm: int = 60
    parallel_collection: bool = True
    retry_attempts: int = 3
    backoff_factor: float = 2.0

@dataclass
class Article:
    """Article class for news data"""
    article_id: str
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    author: str = ""

class NewsAPICollector:
    """Real NewsAPI collector with pagination"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.daily_quota = 1000
        self.calls_made_today = 0
        self.last_reset = datetime.now().date()

    def collect_stock_news(self, ticker: str, company_name: str, days_back: int) -> Tuple[List[Article], int]:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        query = f"{ticker} OR \"{urllib.parse.quote(company_name)}\" OR earnings OR stock"
        articles = []
        page = 1
        max_pages = 10
        api_calls = 0

        while page <= max_pages:
            url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&to={to_date}&sortBy=publishedAt&pageSize=100&page={page}&apiKey={self.api_key}"
            try:
                response = requests.get(url)
                api_calls += 1
                response.raise_for_status()
                data = response.json()
                if data['status'] != 'ok':
                    logging.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                    break
                for art in data.get('articles', []):
                    pub_at_str = art['publishedAt'].replace('Z', '+00:00')
                    pub_at = datetime.fromisoformat(pub_at_str)
                    article_id = hashlib.md5(art['url'].encode()).hexdigest()
                    desc = art.get('description', '') or ''
                    cont = art.get('content', '') or ''
                    content = (desc + ' ' + cont).strip() if (desc or cont) else 'No content available'
                    articles.append(Article(
                        article_id=article_id,
                        title=art['title'] or 'No title',
                        content=content,
                        url=art['url'],
                        source=art['source']['name'],
                        published_at=pub_at,
                        author=art.get('author', '')
                    ))
                total_results = data.get('totalResults', 0)
                if len(data.get('articles', [])) < 100 or page * 100 >= total_results:
                    break
                page += 1
            except Exception as e:
                logging.error(f"NewsAPI request failed: {e}")
                break

        logging.info(f"Collected {len(articles)} raw articles from NewsAPI for {ticker} ({api_calls} calls)")
        return articles, api_calls

class GNewsCollector:
    """Real GNews collector with pagination"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.daily_quota = 100
        self.calls_made_today = 0
        self.last_reset = datetime.now().date()

    def collect_stock_news(self, ticker: str, company_name: str, days_back: int) -> Tuple[List[Article], int]:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%dT%H:%M:%SZ')
        to_date = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        query = f"{ticker} OR \"{urllib.parse.quote(company_name)}\" OR earnings OR stock"
        articles = []
        page = 1
        max_pages = 5
        api_calls = 0

        while page <= max_pages:
            url = f"https://gnews.io/api/v4/search?q={query}&lang=en&country=us&max=100&page={page}&from={from_date}&to={to_date}&token={self.api_key}"
            try:
                response = requests.get(url)
                api_calls += 1
                response.raise_for_status()
                data = response.json()
                for art in data.get('articles', []):
                    pub_at_str = art['publishedAt'].replace('Z', '+00:00')
                    pub_at = datetime.fromisoformat(pub_at_str)
                    article_id = hashlib.md5(art['url'].encode()).hexdigest()
                    desc = art.get('description', '') or ''
                    cont = art.get('content', '') or ''
                    content = (desc + ' ' + cont).strip() if (desc or cont) else 'No content available'
                    articles.append(Article(
                        article_id=article_id,
                        title=art['title'] or 'No title',
                        content=content,
                        url=art['url'],
                        source=art['source']['name'],
                        published_at=pub_at,
                        author=''
                    ))
                if len(data.get('articles', [])) < 100:
                    break
                page += 1
            except Exception as e:
                logging.error(f"GNews request failed: {e}")
                break

        logging.info(f"Collected {len(articles)} raw articles from GNews for {ticker} ({api_calls} calls)")
        return articles, api_calls

class FinnhubCollector:
    """Real Finnhub collector"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.minute_quota = 60
        self.calls_made_this_minute = 0
        self.last_minute_reset = datetime.now().replace(second=0, microsecond=0)

    def collect_stock_news(self, ticker: str, company_name: str, days_back: int) -> Tuple[List[Article], int]:
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={to_date}&token={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            articles = []
            for art in data:
                pub_at = datetime.fromtimestamp(art['datetime'])
                article_id = str(art['id'])
                content = art['summary'] or 'No summary available'
                articles.append(Article(
                    article_id=article_id,
                    title=art['headline'] or 'No headline',
                    content=f"{ticker}: {content}",
                    url=art['url'],
                    source=art['source'],
                    published_at=pub_at,
                    author=''
                ))
            logging.info(f"Collected {len(articles)} raw articles from Finnhub news for {ticker}")
            return articles, 1
        except Exception as e:
            logging.error(f"Finnhub news request failed: {e}")
            return [], 1

    def collect_earnings_calendar(self, ticker: str, days: int) -> Tuple[List[Article], int]:
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&from={from_date}&to={to_date}&token={self.api_key}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            earnings = data.get('earningsCalendar', [])
            articles = []
            for earn in earnings:
                date = earn['date']
                pub_at = datetime.strptime(date, '%Y-%m-%d')
                content = f"{ticker} Earnings: EPS Actual: {earn.get('epsActual', 'N/A')} Estimate: {earn.get('epsEstimate', 'N/A')} | Revenue Actual: {earn.get('revenueActual', 'N/A')} Estimate: {earn.get('revenueEstimate', 'N/A')}"
                article_id = hashlib.md5(f"earnings_{ticker}_{date}".encode()).hexdigest()
                articles.append(Article(
                    article_id=article_id,
                    title=f"Earnings Report for {ticker} on {date}",
                    content=content,
                    url='',
                    source='Finnhub Earnings',
                    published_at=pub_at
                ))
            logging.info(f"Collected {len(articles)} earnings items from Finnhub for {ticker}")
            return articles, 1
        except Exception as e:
            logging.error(f"Finnhub earnings request failed: {e}")
            return [], 1

class ComprehensiveTickerNewsCollector:
    """Enhanced collector for systematic multi-ticker news collection"""
    
    def __init__(self, config: CollectionConfig = None):
        config_data = load_config()
        # Map config.json keys to CollectionConfig fields
        config_params = {
            'days_back': config_data.get('days_back_news', 30),  # Map days_back_news to days_back
            'max_articles_per_ticker': config_data.get('max_articles_per_ticker', 200),
            'newsapi_rpm': config_data.get('newsapi_rpm', 100),
            'gnews_rpm': config_data.get('gnews_rpm', 60),
            'finnhub_rpm': config_data.get('finnhub_rpm', 60),
            'parallel_collection': config_data.get('parallel_collection', True),
            'retry_attempts': config_data.get('retry_attempts', 3),
            'backoff_factor': config_data.get('backoff_factor', 2.0)
        }
        self.config = config or CollectionConfig(**config_params)
        self.data_dir = Path(config_data['news_db_path']).parent
        self.data_dir.mkdir(exist_ok=True)
        self.db_lock = Lock()
        self.db_path = Path(config_data['news_db_path'])
        self.tickers = config_data['tickers']
        self.company_names = config_data['company_names']
        self.setup_enhanced_database()
        self.collection_stats = {}
        self.collectors = {}
    
    def setup_enhanced_database(self):
        """Setup enhanced database schema for comprehensive collection"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id TEXT UNIQUE,
                ticker TEXT NOT NULL,
                title TEXT,
                content TEXT,
                url TEXT,
                source TEXT,
                published_at TIMESTAMP,
                collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sentiment_score REAL,
                relevance_score REAL,
                author TEXT,
                content_length INTEGER
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_ticker ON articles(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source)')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                source TEXT,
                collection_date DATE,
                articles_collected INTEGER,
                articles_filtered INTEGER,
                success_rate REAL,
                avg_relevance_score REAL,
                date_range_start DATE,
                date_range_end DATE,
                api_calls_made INTEGER,
                errors_encountered INTEGER,
                collection_duration_seconds REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info(f"Database initialized at {self.db_path}")
    
    def get_all_tickers_from_stock_data(self) -> List[Dict]:
        """Get tickers from stock database"""
        config = load_config()
        stock_db_path = Path(config['stock_db_path'])
        try:
            conn = sqlite3.connect(stock_db_path)
            query = "SELECT DISTINCT ticker FROM stock_prices"
            df = pd.read_sql_query(query, conn)
            conn.close()
            tickers = [
                {'ticker': row['ticker'], 'company_name': self.company_names.get(row['ticker'], row['ticker'])}
                for _, row in df.iterrows()
            ]
            logging.info(f"Found {len(tickers)} tickers in stock database")
            return tickers
        except Exception as e:
            logging.warning(f"Failed to load tickers from stock DB: {e}. Using config tickers.")
            return [
                {'ticker': ticker, 'company_name': self.company_names.get(ticker, ticker)}
                for ticker in self.tickers
            ]
    
    def save_ticker_metadata(self, tickers_info: List[Dict]):
        """Save ticker metadata"""
        pass
    
    def setup_collectors_with_quotas(self, api_keys: Dict[str, str]):
        """Setup collectors with API keys"""
        if 'newsapi' in api_keys:
            self.collectors['newsapi'] = NewsAPICollector(api_keys['newsapi'])
        if 'gnews' in api_keys:
            self.collectors['gnews'] = GNewsCollector(api_keys['gnews'])
        if 'finnhub' in api_keys:
            self.collectors['finnhub'] = FinnhubCollector(api_keys['finnhub'])
    
    def _check_quota(self, collector, source: str) -> bool:
        """Check if collector has quota remaining"""
        now = datetime.now()
        if source in ['newsapi', 'gnews']:
            if collector.last_reset != now.date():
                collector.calls_made_today = 0
                collector.last_reset = now.date()
            return collector.calls_made_today < collector.daily_quota
        elif source == 'finnhub':
            current_minute = now.replace(second=0, microsecond=0)
            if collector.last_minute_reset != current_minute:
                collector.calls_made_this_minute = 0
                self.last_minute_reset = current_minute
            return collector.calls_made_this_minute < collector.minute_quota
        return True
    
    def _filter_and_enhance_articles(self, articles: List[Article], ticker: str, company_name: str) -> List[Article]:
        """Filter articles for quality and enhance with relevance scores"""
        enhanced_articles = []
        for article in articles:
            if len(article.content) < 20:
                continue
            if not article.title or len(article.title) < 10:
                continue
            relevance_score = self._calculate_relevance_score(article, ticker, company_name)
            article.relevance_score = relevance_score
            if relevance_score >= 0.3:
                enhanced_articles.append(article)
        enhanced_articles.sort(key=lambda x: x.relevance_score or 0, reverse=True)
        return enhanced_articles[:self.config.max_articles_per_ticker]
    
    def _calculate_relevance_score(self, article: Article, ticker: str, company_name: str) -> float:
        """Calculate relevance score for an article"""
        score = 0.0
        text = (article.title + ' ' + article.content).lower()
        ticker_lower = ticker.lower()
        company_lower = company_name.lower()
        mentions = text.count(ticker_lower) + text.upper().count(ticker)
        company_words = company_lower.split()
        for word in company_words:
            if len(word) > 3:
                mentions += text.count(word)
        score += min(mentions * 0.1, 0.4)
        financial_keywords = [
            'earnings', 'revenue', 'profit', 'loss', 'stock', 'shares', 'trading',
            'market', 'analyst', 'upgrade', 'downgrade', 'price target', 'dividend',
            'acquisition', 'merger', 'partnership', 'contract', 'guidance', 'outlook'
        ]
        financial_score = sum(1 for keyword in financial_keywords if keyword in text)
        score += min(financial_score * 0.1, 0.5)
        if 'bloomberg' in article.source.lower() or 'reuters' in article.source.lower():
            score += 0.1
        return min(score, 1.0)
    
    def _save_articles_batch(self, articles: List[Article], ticker: str, source: str) -> int:
        """Save a batch of articles to database"""
        if not articles:
            return 0
        saved_count = 0
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for article in articles:
                try:
                    cursor.execute('''
                        INSERT OR REPLACE INTO articles
                        (article_id, ticker, title, content, url, source, published_at,
                         sentiment_score, relevance_score, author, content_length)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        article.article_id,
                        ticker,
                        article.title,
                        article.content,
                        article.url,
                        article.source,
                        article.published_at.isoformat(),
                        article.sentiment_score,
                        article.relevance_score,
                        article.author,
                        len(article.content)
                    ))
                    saved_count += 1
                except Exception as e:
                    logging.error(f"Error saving article {article.article_id}: {e}")
            conn.commit()
            conn.close()
        return saved_count
    
    def _log_collection_progress(self, ticker: str, source: str, collected: int, 
                               saved: int, success_rate: float, avg_relevance: float, 
                               duration: float, date_range_start, date_range_end,
                               api_calls_made: int, errors_encountered: int):
        """Log collection progress to database"""
        with self.db_lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO collection_progress (ticker, source, collection_date, articles_collected, articles_filtered,
                 success_rate, avg_relevance_score, date_range_start, date_range_end,
                 api_calls_made, errors_encountered, collection_duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker, source, datetime.now().date(), collected, saved,
                success_rate, avg_relevance, date_range_start, date_range_end,
                api_calls_made, errors_encountered, duration
            ))
            conn.commit()
            conn.close()
    
    def collect_ticker_news_comprehensive(self, ticker_info: Dict, source_name: str) -> Dict:
        """Collect news for a single ticker from a source"""
        ticker = ticker_info['ticker']
        company_name = ticker_info['company_name']
        collector = self.collectors.get(source_name)
        if not collector:
            return {'success': False, 'error': 'Collector not initialized'}
        
        start_time = time.time()
        articles, api_calls = [], 0
        errors_encountered = 0
        
        for attempt in range(self.config.retry_attempts):
            try:
                if source_name == 'newsapi':
                    articles, api_calls = collector.collect_stock_news(ticker, company_name, self.config.days_back)
                elif source_name == 'gnews':
                    articles, api_calls = collector.collect_stock_news(ticker, company_name, self.config.days_back)
                elif source_name == 'finnhub':
                    news_articles, news_calls = collector.collect_stock_news(ticker, company_name, self.config.days_back)
                    earnings_articles, earnings_calls = collector.collect_earnings_calendar(ticker, self.config.days_back)
                    articles = news_articles + earnings_articles
                    api_calls = news_calls + earnings_calls
                break
            except Exception as e:
                errors_encountered += 1
                backoff_time = self.config.backoff_factor ** attempt
                logging.warning(f"Attempt {attempt+1} failed for {source_name}: {e}. Retrying in {backoff_time}s")
                time.sleep(backoff_time)
        
        collected = len(articles)
        enhanced = self._filter_and_enhance_articles(articles, ticker, company_name)
        saved = self._save_articles_batch(enhanced, ticker, source_name)
        success_rate = saved / max(collected, 1)
        avg_relevance = np.mean([a.relevance_score for a in enhanced]) if enhanced else 0.0
        date_range_start = min([a.published_at.date() for a in enhanced]) if enhanced else None
        date_range_end = max([a.published_at.date() for a in enhanced]) if enhanced else None
        duration = time.time() - start_time
        
        self._log_collection_progress(
            ticker, source_name, collected, saved, success_rate, avg_relevance,
            duration, date_range_start, date_range_end, api_calls, errors_encountered
        )
        
        return {
            'success': saved > 0,
            'articles_collected': collected,
            'articles_saved': saved,
            'success_rate': success_rate,
            'avg_relevance': avg_relevance,
            'api_calls_made': api_calls,
            'errors_encountered': errors_encountered,
            'collection_time': duration
        }
    
    def collect_all_tickers_systematically(self, api_keys: Dict[str, str]) -> Dict:
        """Systematically collect news for all tickers"""
        self.setup_collectors_with_quotas(api_keys)
        tickers_info = self.get_all_tickers_from_stock_data()
        self.save_ticker_metadata(tickers_info)
        
        total_stats = {
            'tickers_processed': 0,
            'total_articles': 0,
            'successful_collections': 0,
            'failed_collections': 0,
            'sources_used': list(self.collectors.keys()),
            'total_api_calls': 0
        }
        
        logging.info(f"Starting systematic collection for {len(tickers_info)} tickers")
        logging.info(f"Using sources: {list(self.collectors.keys())}")
        
        def process_ticker(ticker_info):
            ticker = ticker_info['ticker']
            ticker_stats = {'ticker': ticker, 'sources': {}}
            for source_name in self.collectors.keys():
                logging.info(f"  Collecting from {source_name} for {ticker}...")
                result = self.collect_ticker_news_comprehensive(ticker_info, source_name)
                if result['success']:
                    total_stats['successful_collections'] += 1
                    total_stats['total_articles'] += result['articles_saved']
                    logging.info(f"  Collected from {source_name}: {result['articles_saved']} articles")
                else:
                    total_stats['failed_collections'] += 1
                    logging.warning(f"  Failed {source_name}: {result.get('error', 'Unknown error')}")
                ticker_stats['sources'][source_name] = result
                total_stats['total_api_calls'] += result.get('api_calls_made', 0)
                time.sleep(1.0)
            total_stats['tickers_processed'] += 1
            self.collection_stats[ticker] = ticker_stats
            return ticker_stats
        
        if self.config.parallel_collection:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_ticker = {executor.submit(process_ticker, ticker_info): ticker_info['ticker'] for ticker_info in tickers_info}
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        future.result()
                        logging.info(f"Progress: {total_stats['tickers_processed']}/{len(tickers_info)} tickers processed")
                    except Exception as e:
                        logging.error(f"Error processing {ticker}: {e}")
        else:
            for i, ticker_info in enumerate(tickers_info, 1):
                process_ticker(ticker_info)
                logging.info(f"Progress: {i}/{len(tickers_info)} tickers processed")
        
        self.generate_collection_summary(total_stats)
        return total_stats
    
    def generate_collection_summary(self, stats: Dict):
        """Generate comprehensive collection summary"""
        logging.info("=== COLLECTION SUMMARY ===")
        logging.info(f"Tickers processed: {stats['tickers_processed']}")
        logging.info(f"Total articles collected: {stats['total_articles']}")
        logging.info(f"Successful collections: {stats['successful_collections']}")
        logging.info(f"Failed collections: {stats['failed_collections']}")
        logging.info(f"Total API calls: {stats['total_api_calls']}")
        logging.info(f"Sources used: {', '.join(stats['sources_used'])}")
        
        summary_data = []
        for ticker, ticker_stats in self.collection_stats.items():
            for source, source_stats in ticker_stats['sources'].items():
                summary_data.append({
                    'ticker': ticker,
                    'source': source,
                    'success': source_stats['success'],
                    'articles_collected': source_stats.get('articles_collected', 0),
                    'articles_saved': source_stats.get('articles_saved', 0),
                    'success_rate': source_stats.get('success_rate', 0),
                    'collection_time': source_stats.get('collection_time', 0),
                    'error': source_stats.get('error', '')
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.data_dir / "comprehensive_collection_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        ticker_summary = summary_df.groupby('ticker').agg({
            'articles_saved': 'sum',
            'success_rate': 'mean',
            'collection_time': 'sum'
        }).round(3)
        ticker_summary_path = self.data_dir / "ticker_summary.csv"
        ticker_summary.to_csv(ticker_summary_path)
        logging.info(f"Detailed summaries saved to {self.data_dir}")

def main():
    """Main execution for news collection"""
    config_data = load_config()
    api_keys = load_api_keys()
    if not api_keys:
        print("⚠️ No valid API keys provided. Please update config/api_keys.json")
        return
    
    collector = ComprehensiveTickerNewsCollector()
    start_time = time.time()
    results = collector.collect_all_tickers_systematically(api_keys)
    end_time = time.time()
    
    print(f"\n🎉 Collection completed in {(end_time - start_time)/60:.1f} minutes")
    print(f"📈 Total articles collected: {results['total_articles']}")
    print(f"💾 Data saved to: {collector.data_dir}")

if __name__ == "__main__":
    main()