import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import sqlite3
import json
import re
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import hashlib
from dataclasses import dataclass
from urllib.parse import urlencode, quote
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_data_collection.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class NewsArticle:
    """Data class for news articles"""
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    ticker_mentions: List[str]
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    article_id: Optional[str] = None
    author: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'source': self.source,
            'published_at': self.published_at.isoformat(),
            'ticker_mentions': ','.join(self.ticker_mentions),
            'sentiment_score': self.sentiment_score,
            'relevance_score': self.relevance_score,
            'article_id': self.article_id,
            'author': self.author
        }

class NewsCollector:
    """Base class for news collection from various sources"""
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 1.0):
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # Setup robust session with retries
        self.session = requests.Session()
        try:
            # Try new parameter name first (urllib3 >= 1.26.0)
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1
            )
        except TypeError:
            # Fall back to old parameter name (urllib3 < 1.26.0)
            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                method_whitelist=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1
            )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
    def _rate_limit_sleep(self):
        """Implement rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, headers: Dict = None, params: Dict = None) -> Optional[Dict]:
        """Make HTTP request with error handling"""
        try:
            self._rate_limit_sleep()
            
            if headers is None:
                headers = {'User-Agent': 'FinancialNewsBot/1.0'}
            
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {url}: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error for {url}: {e}")
            return None

class NewsAPICollector(NewsCollector):
    """Collect news from NewsAPI.org"""
    
    def __init__(self, api_key: str):
        super().__init__(api_key, rate_limit=0.5)  # 2 requests per second
        self.base_url = "https://newsapi.org/v2"
    
    def collect_stock_news(self, tickers: List[str], days_back: int = 30) -> List[NewsArticle]:
        """Collect news articles mentioning specific stock tickers"""
        articles = []
        
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        for ticker in tickers:
            logging.info(f"Collecting NewsAPI articles for {ticker}")
            
            # Search for ticker in business/financial sources
            query = f'"{ticker}" OR "{ticker.lower()}" stock OR finance'
            
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.api_key,
                'pageSize': 100,
                'sources': 'bloomberg,financial-times,reuters,the-wall-street-journal,cnbc,marketwatch'
            }
            
            data = self._make_request(f"{self.base_url}/everything", params=params)
            
            if data and 'articles' in data:
                for article_data in data['articles']:
                    try:
                        article = self._parse_newsapi_article(article_data, [ticker])
                        if article and self._is_relevant_article(article, ticker):
                            articles.append(article)
                    except Exception as e:
                        logging.error(f"Error parsing NewsAPI article: {e}")
                        continue
            
            time.sleep(0.1)  # Small additional delay between tickers
        
        logging.info(f"Collected {len(articles)} articles from NewsAPI")
        return articles
    
    def _parse_newsapi_article(self, data: Dict, tickers: List[str]) -> Optional[NewsArticle]:
        """Parse NewsAPI article data"""
        try:
            published_at = datetime.fromisoformat(data['publishedAt'].replace('Z', '+00:00'))
            
            return NewsArticle(
                title=data.get('title', ''),
                content=data.get('description', '') + ' ' + (data.get('content', '') or ''),
                url=data.get('url', ''),
                source=f"newsapi_{data.get('source', {}).get('name', 'unknown')}",
                published_at=published_at,
                ticker_mentions=tickers,
                article_id=hashlib.md5(data.get('url', '').encode()).hexdigest(),
                author=data.get('author')
            )
        except Exception as e:
            logging.error(f"Error parsing NewsAPI article: {e}")
            return None
    
    def _is_relevant_article(self, article: NewsArticle, ticker: str) -> bool:
        """Check if article is relevant to the ticker"""
        text = (article.title + ' ' + article.content).lower()
        ticker_lower = ticker.lower()
        
        # Must contain ticker and financial keywords
        has_ticker = ticker_lower in text or ticker in text.upper()
        financial_keywords = ['stock', 'share', 'trading', 'market', 'investor', 'financial', 'earnings', 'revenue', 'profit']
        has_financial = any(keyword in text for keyword in financial_keywords)
        
        return has_ticker and has_financial

class GNewsCollector(NewsCollector):
    """Collect news from Google News via GNews API"""
    
    def __init__(self):
        super().__init__(rate_limit=1.0)  # 1 request per second for free tier
        self.base_url = "https://gnews.io/api/v4"
    
    def collect_stock_news(self, tickers: List[str], days_back: int = 30, api_key: str = None) -> List[NewsArticle]:
        """Collect news from Google News"""
        if not api_key:
            logging.warning("GNews API key not provided, skipping GNews collection")
            return []
        
        articles = []
        from_date = datetime.now() - timedelta(days=days_back)
        
        for ticker in tickers:
            logging.info(f"Collecting GNews articles for {ticker}")
            
            params = {
                'q': f'{ticker} stock market finance',
                'token': api_key,
                'lang': 'en',
                'country': 'us',
                'max': 100,
                'from': from_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            
            data = self._make_request(f"{self.base_url}/search", params=params)
            
            if data and 'articles' in data:
                for article_data in data['articles']:
                    try:
                        article = self._parse_gnews_article(article_data, [ticker])
                        if article:
                            articles.append(article)
                    except Exception as e:
                        logging.error(f"Error parsing GNews article: {e}")
                        continue
        
        logging.info(f"Collected {len(articles)} articles from GNews")
        return articles
    
    def _parse_gnews_article(self, data: Dict, tickers: List[str]) -> Optional[NewsArticle]:
        """Parse GNews article data"""
        try:
            published_at = datetime.fromisoformat(data['publishedAt'].replace('Z', '+00:00'))
            
            return NewsArticle(
                title=data.get('title', ''),
                content=data.get('description', ''),
                url=data.get('url', ''),
                source=f"gnews_{data.get('source', {}).get('name', 'unknown')}",
                published_at=published_at,
                ticker_mentions=tickers,
                article_id=hashlib.md5(data.get('url', '').encode()).hexdigest()
            )
        except Exception as e:
            logging.error(f"Error parsing GNews article: {e}")
            return None

class RedditCollector(NewsCollector):
    """Collect financial discussions from Reddit"""
    
    def __init__(self, client_id: str = None, client_secret: str = None):
        super().__init__(rate_limit=2.0)  # 30 requests per minute limit
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expires = 0
        
        if client_id and client_secret:
            self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Reddit API"""
        try:
            auth = requests.auth.HTTPBasicAuth(self.client_id, self.client_secret)
            data = {
                'grant_type': 'client_credentials'
            }
            headers = {'User-Agent': 'FinancialNewsBot/1.0'}
            
            response = requests.post('https://www.reddit.com/api/v1/access_token',
                                   auth=auth, data=data, headers=headers)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            self.token_expires = time.time() + token_data['expires_in'] - 60  # 1 min buffer
            
            logging.info("Successfully authenticated with Reddit API")
            
        except Exception as e:
            logging.error(f"Reddit authentication failed: {e}")
            self.access_token = None
    
    def collect_stock_discussions(self, tickers: List[str], days_back: int = 7) -> List[NewsArticle]:
        """Collect Reddit discussions about stocks"""
        if not self.access_token:
            logging.warning("Reddit API not authenticated, skipping Reddit collection")
            return []
        
        articles = []
        subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting', 'wallstreetbets']
        
        for ticker in tickers:
            for subreddit in subreddits:
                logging.info(f"Collecting Reddit posts for {ticker} in r/{subreddit}")
                
                try:
                    posts = self._search_subreddit(subreddit, ticker, days_back)
                    for post in posts:
                        article = self._parse_reddit_post(post, [ticker])
                        if article:
                            articles.append(article)
                except Exception as e:
                    logging.error(f"Error collecting from r/{subreddit}: {e}")
                    continue
                    
                time.sleep(0.5)  # Rate limiting between subreddits
        
        logging.info(f"Collected {len(articles)} posts from Reddit")
        return articles
    
    def _search_subreddit(self, subreddit: str, ticker: str, days_back: int) -> List[Dict]:
        """Search for posts in a subreddit"""
        if time.time() > self.token_expires:
            self._authenticate()
        
        headers = {
            'Authorization': f'bearer {self.access_token}',
            'User-Agent': 'FinancialNewsBot/1.0'
        }
        
        params = {
            'q': ticker,
            'restrict_sr': 'on',
            'sort': 'new',
            'limit': 25,
            't': 'week' if days_back <= 7 else 'month'
        }
        
        url = f'https://oauth.reddit.com/r/{subreddit}/search'
        data = self._make_request(url, headers=headers, params=params)
        
        if data and 'data' in data:
            return data['data']['children']
        return []
    
    def _parse_reddit_post(self, post_data: Dict, tickers: List[str]) -> Optional[NewsArticle]:
        """Parse Reddit post data"""
        try:
            post = post_data['data']
            created_utc = datetime.fromtimestamp(post['created_utc'])
            
            # Filter out very short posts
            content = post.get('selftext', '') or post.get('title', '')
            if len(content) < 50:
                return None
            
            return NewsArticle(
                title=post.get('title', ''),
                content=content,
                url=f"https://reddit.com{post.get('permalink', '')}",
                source=f"reddit_{post.get('subreddit', 'unknown')}",
                published_at=created_utc,
                ticker_mentions=tickers,
                article_id=post.get('id'),
                author=post.get('author')
            )
        except Exception as e:
            logging.error(f"Error parsing Reddit post: {e}")
            return None

class FinnhubCollector(NewsCollector):
    """Collect financial news from Finnhub.io - Comprehensive financial data platform"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key, rate_limit=0.2)  # 5 requests per second for free tier
        self.base_url = "https://finnhub.io/api/v1"
    
    def collect_stock_news(self, tickers: List[str], days_back: int = 30) -> List[NewsArticle]:
        """Collect company news from Finnhub"""
        if not self.api_key:
            logging.warning("Finnhub API key not provided, skipping Finnhub collection")
            return []
        
        articles = []
        
        # Calculate date range
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days_back)
        
        for ticker in tickers:
            logging.info(f"Collecting Finnhub news for {ticker}")
            
            try:
                # Company news endpoint
                company_articles = self._collect_company_news(ticker, from_date, to_date)
                articles.extend(company_articles)
                
                # Market news endpoint (general market news that might affect the stock)
                if ticker in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:  # Major stocks
                    market_articles = self._collect_market_news(ticker, from_date, to_date)
                    articles.extend(market_articles)
                
                time.sleep(0.3)  # Additional rate limiting between tickers
                
            except Exception as e:
                logging.error(f"Error collecting Finnhub news for {ticker}: {e}")
                continue
        
        logging.info(f"Collected {len(articles)} articles from Finnhub")
        return articles
    
    def _collect_company_news(self, ticker: str, from_date: datetime, to_date: datetime) -> List[NewsArticle]:
        """Collect company-specific news"""
        params = {
            'symbol': ticker,
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': self.api_key
        }
        
        data = self._make_request(f"{self.base_url}/company-news", params=params)
        articles = []
        
        if data and isinstance(data, list):
            for news_item in data:
                try:
                    article = self._parse_finnhub_article(news_item, [ticker], source_type='company')
                    if article and self._is_relevant_finnhub_article(article, ticker):
                        articles.append(article)
                except Exception as e:
                    logging.error(f"Error parsing Finnhub company article: {e}")
                    continue
        
        return articles
    
    def _collect_market_news(self, ticker: str, from_date: datetime, to_date: datetime) -> List[NewsArticle]:
        """Collect general market news"""
        params = {
            'category': 'general',  # General market news
            'token': self.api_key
        }
        
        data = self._make_request(f"{self.base_url}/news", params=params)
        articles = []
        
        if data and isinstance(data, list):
            for news_item in data[:20]:  # Limit to recent 20 articles
                try:
                    # Check if the article mentions the ticker
                    article_text = (news_item.get('headline', '') + ' ' + 
                                  news_item.get('summary', '')).lower()
                    
                    if ticker.lower() in article_text or ticker in article_text.upper():
                        article = self._parse_finnhub_article(news_item, [ticker], source_type='market')
                        if article:
                            articles.append(article)
                except Exception as e:
                    logging.error(f"Error parsing Finnhub market article: {e}")
                    continue
        
        return articles
    
    def _parse_finnhub_article(self, data: Dict, tickers: List[str], source_type: str = 'company') -> Optional[NewsArticle]:
        """Parse Finnhub article data"""
        try:
            # Finnhub returns timestamp in Unix format
            published_at = datetime.fromtimestamp(data.get('datetime', 0))
            
            # Skip very old articles (data quality check)
            if published_at < datetime.now() - timedelta(days=90):
                return None
            
            headline = data.get('headline', '')
            summary = data.get('summary', '')
            
            # Combine headline and summary for content
            content = f"{headline}. {summary}" if summary else headline
            
            # Skip articles that are too short or don't have meaningful content
            if len(content) < 30:
                return None
            
            return NewsArticle(
                title=headline,
                content=content,
                url=data.get('url', ''),
                source=f"finnhub_{source_type}",
                published_at=published_at,
                ticker_mentions=tickers,
                article_id=hashlib.md5((data.get('url', '') + str(data.get('datetime', ''))).encode()).hexdigest(),
                author=data.get('source', 'Finnhub')
            )
            
        except Exception as e:
            logging.error(f"Error parsing Finnhub article: {e}")
            return None
    
    def _is_relevant_finnhub_article(self, article: NewsArticle, ticker: str) -> bool:
        """Check if Finnhub article is relevant to the ticker"""
        text = (article.title + ' ' + article.content).lower()
        ticker_lower = ticker.lower()
        
        # Finnhub company news should already be relevant, but double-check
        has_ticker = ticker_lower in text or ticker in text.upper()
        
        # Check for financial relevance
        financial_keywords = [
            'stock', 'share', 'trading', 'market', 'investor', 'earnings', 'revenue', 
            'profit', 'loss', 'financial', 'quarter', 'guidance', 'outlook', 'analyst',
            'upgrade', 'downgrade', 'price target', 'dividend', 'acquisition', 'merger'
        ]
        has_financial = any(keyword in text for keyword in financial_keywords)
        
        return has_ticker and (has_financial or len(article.content) > 100)
    
    def collect_earnings_calendar(self, tickers: List[str], days_ahead: int = 30) -> List[NewsArticle]:
        """Collect earnings calendar information"""
        if not self.api_key:
            return []
        
        articles = []
        to_date = datetime.now() + timedelta(days=days_ahead)
        from_date = datetime.now()
        
        params = {
            'from': from_date.strftime('%Y-%m-%d'),
            'to': to_date.strftime('%Y-%m-%d'),
            'token': self.api_key
        }
        
        data = self._make_request(f"{self.base_url}/calendar/earnings", params=params)
        
        if data and 'earningsCalendar' in data:
            for earning in data['earningsCalendar']:
                if earning.get('symbol') in tickers:
                    try:
                        # Create a news article from earnings data
                        ticker = earning.get('symbol')
                        date_str = earning.get('date', '')
                        
                        title = f"Earnings Report: {ticker} scheduled for {date_str}"
                        content = f"Upcoming earnings report for {ticker} on {date_str}. "
                        
                        if earning.get('epsEstimate'):
                            content += f"EPS Estimate: ${earning.get('epsEstimate')}. "
                        if earning.get('revenueEstimate'):
                            content += f"Revenue Estimate: ${earning.get('revenueEstimate')}. "
                        
                        article = NewsArticle(
                            title=title,
                            content=content,
                            url=f"https://finnhub.io/calendar/earnings?symbol={ticker}",
                            source='finnhub_earnings',
                            published_at=datetime.strptime(date_str, '%Y-%m-%d') if date_str else datetime.now(),
                            ticker_mentions=[ticker],
                            article_id=f"earnings_{ticker}_{date_str}",
                            relevance_score=1.0  # Earnings are highly relevant
                        )
                        articles.append(article)
                        
                    except Exception as e:
                        logging.error(f"Error parsing earnings data: {e}")
                        continue
        
        return articles

class ComprehensiveNewsCollector:
    """Main class that coordinates collection from all sources"""
    
    def __init__(self, data_dir: str = "news_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = self.data_dir / "news_data.db"
        self.setup_database()
        
        # Initialize collectors
        self.collectors = {}
    
    def setup_database(self):
        """Initialize SQLite database for news data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                article_id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                url TEXT,
                source TEXT,
                published_at TIMESTAMP,
                ticker_mentions TEXT,
                sentiment_score REAL,
                relevance_score REAL,
                author TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                ticker TEXT,
                articles_collected INTEGER,
                date_range_start TIMESTAMP,
                date_range_end TIMESTAMP,
                collection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info(f"News database initialized at {self.db_path}")
    
    def setup_collectors(self, api_keys: Dict[str, str]):
        """Setup news collectors with API keys"""
        # NewsAPI
        if 'newsapi' in api_keys:
            self.collectors['newsapi'] = NewsAPICollector(api_keys['newsapi'])
        
        # GNews
        if 'gnews' in api_keys:
            self.collectors['gnews'] = GNewsCollector()
            self.collectors['gnews'].api_key = api_keys['gnews']
        
        # Reddit
        if 'reddit_client_id' in api_keys and 'reddit_client_secret' in api_keys:
            self.collectors['reddit'] = RedditCollector(
                api_keys['reddit_client_id'],
                api_keys['reddit_client_secret']
            )
        
        # Finnhub
        if 'finnhub' in api_keys:
            self.collectors['finnhub'] = FinnhubCollector(api_keys['finnhub'])
        
        # Twitter (now replaced by Finnhub, but keeping for future reference)
        # if 'twitter_bearer' in api_keys:
        #     self.collectors['twitter'] = TwitterCollector(api_keys['twitter_bearer'])
        
        logging.info(f"Initialized {len(self.collectors)} news collectors")
    
    def collect_all_news(self, tickers: List[str], days_back: int = 30) -> Dict[str, List[NewsArticle]]:
        """Collect news from all available sources"""
        all_articles = {}
        
        for source_name, collector in self.collectors.items():
            logging.info(f"Starting collection from {source_name}")
            
            try:
                if source_name == 'newsapi':
                    articles = collector.collect_stock_news(tickers, days_back)
                elif source_name == 'gnews':
                    articles = collector.collect_stock_news(tickers, days_back, collector.api_key)
                #elif source_name == 'reddit':
                #    articles = collector.collect_stock_discussions(tickers, min(days_back, 7))
                elif source_name == 'finnhub':
                    articles = collector.collect_stock_news(tickers, days_back)
                    # Also collect earnings calendar for forward-looking sentiment
                    earnings_articles = collector.collect_earnings_calendar(tickers, days_ahead=30)
                    articles.extend(earnings_articles)
                # elif source_name == 'twitter':
                #     articles = collector.collect_stock_tweets(tickers, min(days_back, 7))
                else:
                    continue
                
                all_articles[source_name] = articles
                
                # Save to database
                self._save_articles_to_db(articles)
                
                # Save to CSV
                if articles:
                    self._save_articles_to_csv(articles, f"{source_name}_articles.csv")
                
                logging.info(f"Completed {source_name}: {len(articles)} articles")
                
            except Exception as e:
                logging.error(f"Error collecting from {source_name}: {e}")
                all_articles[source_name] = []
        
        return all_articles
    
    def _save_articles_to_db(self, articles: List[NewsArticle]):
        """Save articles to database"""
        if not articles:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            try:
                data = article.to_dict()
                cursor.execute('''
                    INSERT OR REPLACE INTO news_articles 
                    (article_id, title, content, url, source, published_at, 
                     ticker_mentions, sentiment_score, relevance_score, author)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['article_id'], data['title'], data['content'], data['url'],
                    data['source'], data['published_at'], data['ticker_mentions'],
                    data['sentiment_score'], data['relevance_score'], data['author']
                ))
            except Exception as e:
                logging.error(f"Error saving article to database: {e}")
        
        conn.commit()
        conn.close()
    
    def _save_articles_to_csv(self, articles: List[NewsArticle], filename: str):
        """Save articles to CSV file"""
        if not articles:
            return
        
        df = pd.DataFrame([article.to_dict() for article in articles])
        csv_path = self.data_dir / filename
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved {len(articles)} articles to {csv_path}")
    
    def create_news_summary(self, all_articles: Dict[str, List[NewsArticle]]) -> pd.DataFrame:
        """Create summary of collected news data"""
        summary_data = []
        
        for source, articles in all_articles.items():
            if articles:
                dates = [article.published_at for article in articles]
                
                summary_data.append({
                    'source': source,
                    'articles_count': len(articles),
                    'date_range_start': min(dates),
                    'date_range_end': max(dates),
                    'unique_tickers': len(set().union(*[article.ticker_mentions for article in articles])),
                    'avg_content_length': np.mean([len(article.content) for article in articles])
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.data_dir / "news_collection_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        return summary_df

# Configuration and usage example
def main():
    """Example usage of the comprehensive news collector"""
    
    # API Keys configuration (replace with your actual keys)
    api_keys = {
        'newsapi': '2454405473734322a1c58919ea0c8530',  # Get from https://newsapi.org/
        'gnews': 'b6b22a5feda3e931d79429a18d2fd462',      # Get from https://gnews.io/
        #'reddit_client_id': 'your_reddit_client_id',
        #'reddit_client_secret': 'your_reddit_secret',
        'finnhub': 'd33ene9r01qib1p0mo80d33ene9r01qib1p0mo8g'   # Get from https://finnhub.io/ (FREE!)
    }
    
    # Remove keys that are not available (use empty dict to skip all)
    available_keys = {k: v for k, v in api_keys.items() if v != f'your_{k.split("_")[0]}_key_here' and v != f'your_{k}_here'}
    
    if not available_keys:
        print("⚠️  No API keys provided. The collector will not work without at least one API key.")
        print("\nTo use this pipeline, you need to obtain API keys from:")
        print("- NewsAPI: https://newsapi.org/ (Free: 500 requests/day)")
        print("- GNews: https://gnews.io/ (Free: 100 requests/day)")
        print("- Reddit API: https://www.reddit.com/prefs/apps (Free)")
        print("- Finnhub: https://finnhub.io/ (Free: 60 requests/minute) ⭐ RECOMMENDED")
        print("\nFinnhub is particularly valuable as it provides:")
        print("  • High-quality company-specific news")
        print("  • Earnings calendar data") 
        print("  • Professional financial data with good free tier")
        return
    
    # Initialize collector
    collector = ComprehensiveNewsCollector("financial_news_data")
    collector.setup_collectors(available_keys)
    
    # Define tickers (matching your stock data)
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print(f"Starting news collection for {len(tickers)} tickers from {len(collector.collectors)} sources...")
    print(f"Active sources: {list(collector.collectors.keys())}")
    
    # Collect news
    all_articles = collector.collect_all_news(tickers, days_back=14)
    
    # Create summary
    summary = collector.create_news_summary(all_articles)
    print("\nNews Collection Summary:")
    print(summary.to_string(index=False))
    
    # Calculate totals
    total_articles = sum(len(articles) for articles in all_articles.values())
    print(f"\n📰 Total articles collected: {total_articles}")
    print(f"💾 Data saved to: {collector.data_dir}")
    print(f"🗄️  Database location: {collector.db_path}")

if __name__ == "__main__":
    main()