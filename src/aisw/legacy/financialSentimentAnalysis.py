import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import logging
import re
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from dataclasses import dataclass
import traceback
import warnings
import torch
warnings.filterwarnings('ignore', category=UserWarning)

# Sentiment libraries
try:
    from transformers import pipeline
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    FINBERT_AVAILABLE = True
    VADER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some sentiment libraries not available: {e}")
    FINBERT_AVAILABLE = False
    VADER_AVAILABLE = False

# Configure logging
log_filename = f"logs/sentiment_analysis_{datetime.now().strftime('%Y-%m-%d')}.log"
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        file_handler,
        logging.StreamHandler()
    ]
)

def load_config():
    """Load configuration from config.json"""
    #config_path = Path("src/config/config.json")
    config_path = Path("config/config.json")
    with config_path.open('r') as f:
        return json.load(f)

@dataclass
class SentimentResult:
    """Data class for sentiment analysis results"""
    article_id: str
    ticker: str
    finbert_label: Optional[str] = None
    finbert_score: Optional[float] = None
    finbert_confidence: Optional[float] = None
    vader_compound: Optional[float] = None
    vader_positive: Optional[float] = None
    vader_neutral: Optional[float] = None
    vader_negative: Optional[float] = None
    combined_sentiment: Optional[float] = None
    sentiment_magnitude: Optional[float] = None
    processing_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert result to dictionary, handling None values"""
        return {
            'article_id': self.article_id or '',
            'ticker': self.ticker or '',
            'finbert_label': self.finbert_label,
            'finbert_score': self.finbert_score,
            'finbert_confidence': self.finbert_confidence,
            'vader_compound': self.vader_compound,
            'vader_positive': self.vader_positive,
            'vader_neutral': self.vader_neutral,
            'vader_negative': self.vader_negative,
            'combined_sentiment': self.combined_sentiment,
            'sentiment_magnitude': self.sentiment_magnitude,
            'processing_timestamp': self.processing_timestamp.isoformat() if self.processing_timestamp else None
        }

class FinancialSentimentAnalyzer:
    """Advanced sentiment analysis for financial texts"""
    def __init__(self, model_cache_dir: str = "data/sentiment_models"):
        config = load_config()
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        self.finbert_analyzer = None
        self.vader_analyzer = None
        self.setup_models()
        self.financial_lexicon = self._load_financial_lexicon()
        self.processing_stats = {
            'articles_processed': 0,
            'finbert_successful': 0,
            'vader_successful': 0,
            'processing_errors': 0,
            'start_time': None,
            'per_ticker_time': {}
        }

    def setup_models(self):
        """Initialize sentiment analysis models"""
        logging.info("Initializing sentiment analysis models...")
        batch_size = 32 if torch.cuda.is_available() else 16
        if FINBERT_AVAILABLE:
            try:
                model_name = "ProsusAI/finbert"
                logging.info(f"Loading FinBERT model: {model_name}")
                device = 0 if torch.cuda.is_available() else -1
                logging.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
                logging.info(f"Using FinBERT batch size: {batch_size}")
                self.finbert_analyzer = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    framework="pt",
                    device=device,
                    return_all_scores=True
                )
                logging.info("FinBERT model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load FinBERT: {traceback.format_exc()}")
                self.finbert_analyzer = None
        if VADER_AVAILABLE:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                logging.info("VADER analyzer initialized successfully")
            except Exception as e:
                logging.error(f"Failed to initialize VADER: {traceback.format_exc()}")
                self.vader_analyzer = None
        if not self.finbert_analyzer and not self.vader_analyzer:
            logging.warning("No sentiment analyzers available. Using dummy values.")
    
    def _load_financial_lexicon(self) -> Dict[str, float]:
        """Load enhanced financial sentiment lexicon"""
        financial_lexicon = {
            'earnings beat': 0.8, 'revenue growth': 0.7, 'profit surge': 0.8,
            'strong performance': 0.6, 'bullish': 0.7, 'outperform': 0.6,
            'upgrade': 0.5, 'buy rating': 0.6, 'dividend increase': 0.5,
            'market share': 0.4, 'innovation': 0.4, 'partnership': 0.3,
            'acquisition': 0.3, 'expansion': 0.4, 'growth': 0.4,
            'earnings miss': -0.8, 'revenue decline': -0.7, 'loss': -0.6,
            'weak performance': -0.6, 'bearish': -0.7, 'underperform': -0.6,
            'downgrade': -0.5, 'sell rating': -0.6, 'dividend cut': -0.7,
            'market loss': -0.4, 'lawsuit': -0.5, 'investigation': -0.5,
            'bankruptcy': -0.9, 'restructuring': -0.4, 'layoffs': -0.5,
            'guidance': 0.0, 'forecast': 0.0, 'analyst': 0.0,
            'conference call': 0.0, 'sec filing': 0.0
        }
        logging.info(f"Loaded financial lexicon with {len(financial_lexicon)} terms")
        return financial_lexicon

    def preprocess_text(self, text: str) -> str:
        """Minimal preprocessing: remove URLs and extra spaces"""
        if not text:
            return ""
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def analyze_with_finbert(self, texts: List[str], batch_size: int) -> List[Dict]:
        """Batch-analyze sentiment using FinBERT"""
        if not self.finbert_analyzer or not texts:
            return [{} for _ in texts]
        try:
            results = self.finbert_analyzer(texts, batch_size=batch_size, truncation=True, max_length=512)
            processed_results = []
            for result in results:
                if result:
                    all_scores = {r['label']: r['score'] for r in result}
                    best = max(result, key=lambda x: x['score'])
                    processed_results.append({
                        'label': best['label'],
                        'score': best['score'],
                        'confidence': best['score'],
                        'all_scores': all_scores
                    })
                else:
                    processed_results.append({})
            return processed_results
        except Exception as e:
            logging.error(f"FinBERT batch analysis error: {traceback.format_exc()}")
            return [{} for _ in texts]

    def analyze_with_vader(self, texts: List[str]) -> List[Dict]:
        """Vectorized VADER analysis for batch efficiency"""
        if not self.vader_analyzer or not texts:
            return [{} for _ in texts]
        try:
            results = [
                self.vader_analyzer.polarity_scores(text) if text else {}
                for text in texts
            ]
            return [
                {
                    'compound': r.get('compound', None),
                    'positive': r.get('pos', None),
                    'neutral': r.get('neu', None),
                    'negative': r.get('neg', None)
                }
                for r in results
            ]
        except Exception as e:
            logging.error(f"VADER batch analysis error: {traceback.format_exc()}")
            return [{} for _ in texts]

    def calculate_financial_lexicon_sentiment(self, text: str) -> float:
        """Calculate sentiment using financial lexicon"""
        if not text:
            return 0.0
        text_lower = text.lower()
        sentiment_score = 0.0
        matched_terms = 0
        for term, weight in self.financial_lexicon.items():
            if term in text_lower:
                frequency = text_lower.count(term)
                sentiment_score += weight * frequency
                matched_terms += frequency
        if matched_terms > 0:
            sentiment_score = sentiment_score / matched_terms
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
        return sentiment_score

    def combine_sentiment_scores(self, finbert_result: Dict, vader_result: Dict, lexicon_score: float) -> Tuple[float, float]:
        """Combine scores with improved FinBERT numeric score"""
        scores = []
        weights = []
        if 'all_scores' in finbert_result:
            all_scores = finbert_result['all_scores']
            finbert_numeric = all_scores.get('positive', 0.0) - all_scores.get('negative', 0.0)
            scores.append(finbert_numeric)
            weights.append(0.5)
        if 'compound' in vader_result:
            scores.append(vader_result['compound'])
            weights.append(0.3)
        if lexicon_score != 0.0:
            scores.append(lexicon_score)
            weights.append(0.2)
        if not scores:
            return 0.0, 0.0
        combined_score = np.average(scores, weights=weights)
        sentiment_magnitude = np.mean([abs(s) for s in scores])
        return combined_score, sentiment_magnitude

    def sample_sentiment_outputs(self, results: List[SentimentResult]):
        """Log sample outputs, handling missing analyzers gracefully."""
        if len(results) == 0:
            logging.info("No results to sample.")
            return
        
        logging.info("Sample sentiment outputs:")
        sample_size = min(5, len(results))
        dummy_mode = (not self.finbert_analyzer and not self.vader_analyzer)
        
        for i in range(sample_size):
            result = results[i]
            log_msg = f"Article {result.article_id} ({result.ticker}): "
            
            # FinBERT: Only if analyzer exists AND fields populated
            if self.finbert_analyzer and result.finbert_confidence is not None:
                log_msg += f"FinBERT={result.finbert_label} ({result.finbert_confidence:.2f}), "
            elif dummy_mode:
                log_msg += "FinBERT=dummy, "
            
            # VADER: Similar conditional
            if self.vader_analyzer and result.vader_compound is not None:
                log_msg += f"VADER={result.vader_compound:.2f}, "
            elif dummy_mode:
                log_msg += "VADER=dummy, "
            
            # Always include combined (safe default: 0.0)
            combined_str = f"{result.combined_sentiment:.2f}" if result.combined_sentiment is not None else "0.00"
            log_msg += f"Combined={combined_str}"
            
            # Flag dummy mode if active
            if dummy_mode:
                log_msg += " [DUMMY MODE - Install transformers/vaderSentiment for full analysis]"
            
            logging.info(log_msg)

    def analyze_articles_batch(self, batch_df: pd.DataFrame, batch_size: int) -> List[SentimentResult]:
        """Batch-process sentiment for efficiency"""
        batch_start_time = datetime.now()
        results = []
        texts = []
        for _, row in batch_df.iterrows():
            title = row['title'] or ''
            content = row['content'] or ''
            full_text = f"{title} {title} {content}"
            texts.append(self.preprocess_text(full_text))
        finbert_results = self.analyze_with_finbert(texts, batch_size=batch_size)
        vader_results = self.analyze_with_vader(texts)
        self.processing_stats['finbert_successful'] += len([r for r in finbert_results if r])
        self.processing_stats['vader_successful'] += len([r for r in vader_results if r])
        for idx, row in enumerate(batch_df.itertuples()):
            try:
                finbert = finbert_results[idx]
                vader = vader_results[idx]
                lexicon = self.calculate_financial_lexicon_sentiment(texts[idx])
                combined, magnitude = self.combine_sentiment_scores(finbert, vader, lexicon)
                results.append(SentimentResult(
                    article_id=row.article_id,
                    ticker=row.ticker,
                    finbert_label=finbert.get('label'),
                    finbert_score=finbert.get('score'),
                    finbert_confidence=finbert.get('confidence'),
                    vader_compound=vader.get('compound'),
                    vader_positive=vader.get('positive'),
                    vader_neutral=vader.get('neutral'),
                    vader_negative=vader.get('negative'),
                    combined_sentiment=combined,
                    sentiment_magnitude=magnitude,
                    processing_timestamp=datetime.now()
                ))
                self.processing_stats['articles_processed'] += 1
            except Exception as e:
                logging.error(f"Error analyzing article {row.article_id}: {traceback.format_exc()}")
                self.processing_stats['processing_errors'] += 1
        return results

class SentimentPipelineManager:
    """Manages the complete sentiment analysis pipeline"""
    def __init__(self):
        config = load_config()
        self.news_db_path = Path(config['news_db_path'])
        self.output_dir = Path(config['sentiment_db_path']).parent
        self.output_dir.mkdir(exist_ok=True)
        self.analyzer = FinancialSentimentAnalyzer()
        self.sentiment_db_path = Path(config['sentiment_db_path'])
        self.limit_articles = config['limit_articles']
        self.setup_sentiment_database()

    def setup_sentiment_database(self):
        """Setup database for sentiment results"""
        conn = sqlite3.connect(self.sentiment_db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id TEXT UNIQUE,
                ticker TEXT,
                finbert_label TEXT,
                finbert_score REAL,
                finbert_confidence REAL,
                vader_compound REAL,
                vader_positive REAL,
                vader_neutral REAL,
                vader_negative REAL,
                combined_sentiment REAL,
                sentiment_magnitude REAL,
                processing_timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_ticker ON sentiment_scores(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_timestamp ON sentiment_scores(processing_timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentiment_combined ON sentiment_scores(combined_sentiment)')
        conn.commit()
        conn.close()
        logging.info(f"Sentiment database initialized at {self.sentiment_db_path}")

    def load_articles_from_news_db(self, limit: int = None) -> pd.DataFrame:
        """Load articles, prioritizing high relevance"""
        if not self.news_db_path.exists():
            raise FileNotFoundError(f"News database not found: {self.news_db_path}")
        conn = sqlite3.connect(self.news_db_path)
        query = """
        SELECT article_id, ticker, title, content, source, published_at, relevance_score
        FROM articles
        WHERE title IS NOT NULL AND content IS NOT NULL
        ORDER BY relevance_score DESC, published_at DESC
        """
        if limit:
            query += f" LIMIT {limit}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        duplicates = df['article_id'].duplicated().sum()
        if duplicates > 0:
            logging.warning(f"Found {duplicates} duplicate article IDs; keeping first occurrence")
            df = df.drop_duplicates(subset='article_id', keep='first')
        logging.info(f"Loaded {len(df)} articles from news database")
        return df

    def process_articles_batch(self, articles_df: pd.DataFrame, analyzer_batch_size: int = 32, pipeline_batch_size: int = 50) -> List[SentimentResult]:
        """Process articles in batches"""
        total_articles = len(articles_df)
        results = []
        logging.info(f"Starting sentiment analysis for {total_articles} articles")
        self.analyzer.processing_stats['start_time'] = datetime.now()
        for i in range(0, total_articles, pipeline_batch_size):
            batch_end = min(i + pipeline_batch_size, total_articles)
            batch_df = articles_df.iloc[i:batch_end]
            logging.info(f"Processing batch {i//pipeline_batch_size + 1}: articles {i+1}-{batch_end}")
            batch_results = self.analyzer.analyze_articles_batch(batch_df, batch_size=analyzer_batch_size)
            results.extend(batch_results)
            logging.info(f"Processed {len(results)}/{total_articles} articles")
        self.analyzer.sample_sentiment_outputs(results)
        return results

    def save_sentiment_results(self, results: List[SentimentResult]):
        """Save sentiment results to database, CSV, and JSON"""
        if not results:
            logging.warning("No sentiment results to save")
            return
        conn = sqlite3.connect(self.sentiment_db_path)
        cursor = conn.cursor()
        for result in results:
            data = result.to_dict()
            cursor.execute('''
                INSERT OR REPLACE INTO sentiment_scores
                (article_id, ticker, finbert_label, finbert_score, finbert_confidence,
                 vader_compound, vader_positive, vader_neutral, vader_negative,
                 combined_sentiment, sentiment_magnitude, processing_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['article_id'], data['ticker'], data['finbert_label'],
                data['finbert_score'], data['finbert_confidence'],
                data['vader_compound'], data['vader_positive'],
                data['vader_neutral'], data['vader_negative'],
                data['combined_sentiment'], data['sentiment_magnitude'],
                data['processing_timestamp']
            ))
        conn.commit()
        conn.close()
        results_df = pd.DataFrame([r.to_dict() for r in results])
        csv_path = self.output_dir / "sentiment_analysis_results.csv"
        results_df.to_csv(csv_path, index=False)
        json_path = self.output_dir / "sentiment_analysis_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        logging.info(f"Saved {len(results)} sentiment results to DB, CSV, and JSON")

    def generate_sentiment_summary(self) -> pd.DataFrame:
        """Generate summary statistics by ticker"""
        conn = sqlite3.connect(self.sentiment_db_path)
        query = """
        SELECT 
            ticker,
            COUNT(*) as total_articles,
            AVG(combined_sentiment) as avg_sentiment,
            AVG(sentiment_magnitude) as avg_magnitude,
            AVG(finbert_confidence) as avg_finbert_confidence,
            AVG(vader_compound) as avg_vader_compound,
            SUM(CASE WHEN combined_sentiment > 0.1 THEN 1 ELSE 0 END) as positive_articles,
            SUM(CASE WHEN combined_sentiment < -0.1 THEN 1 ELSE 0 END) as negative_articles,
            SUM(CASE WHEN combined_sentiment BETWEEN -0.1 AND 0.1 THEN 1 ELSE 0 END) as neutral_articles
        FROM sentiment_scores
        GROUP BY ticker
        ORDER BY avg_sentiment DESC
        """
        summary_df = pd.read_sql_query(query, conn)
        conn.close()
        summary_df['positive_ratio'] = summary_df['positive_articles'] / summary_df['total_articles']
        summary_df['negative_ratio'] = summary_df['negative_articles'] / summary_df['total_articles']
        summary_df['neutral_ratio'] = summary_df['neutral_articles'] / summary_df['total_articles']
        numeric_columns = ['avg_sentiment', 'avg_magnitude', 'avg_finbert_confidence', 
                           'avg_vader_compound', 'positive_ratio', 'negative_ratio', 'neutral_ratio']
        summary_df[numeric_columns] = summary_df[numeric_columns].round(3)
        summary_path = self.output_dir / "sentiment_summary_by_ticker.csv"
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Saved sentiment summary to {summary_path}")
        return summary_df

    def run_complete_analysis(self):
        """Run the complete sentiment analysis pipeline"""
        try:
            logging.info("=== Starting Complete Sentiment Analysis Pipeline ===")
            articles_df = self.load_articles_from_news_db(limit=self.limit_articles)
            if articles_df.empty:
                logging.warning("No articles found in database")
                return None
            results = self.process_articles_batch(articles_df)
            if not results:
                logging.error("No sentiment results generated")
                return None
            self.save_sentiment_results(results)
            summary_df = self.generate_sentiment_summary()
            stats = self.analyzer.processing_stats
            processing_time = (datetime.now() - stats['start_time']).total_seconds()
            logging.info("=== SENTIMENT ANALYSIS COMPLETE ===")
            logging.info(f"Articles processed: {stats['articles_processed']}")
            logging.info(f"FinBERT successful: {stats['finbert_successful']}")
            logging.info(f"VADER successful: {stats['vader_successful']}")
            logging.info(f"Processing errors: {stats['processing_errors']}")
            logging.info(f"Total processing time: {processing_time:.1f} seconds")
            logging.info(f"Average time per article: {processing_time / max(stats['articles_processed'], 1):.2f} seconds")
            logging.info("Per-ticker processing times (seconds):")
            for ticker, time in stats['per_ticker_time'].items():
                logging.info(f"  {ticker}: {time:.2f}")
            print("\nSENTIMENT ANALYSIS SUMMARY BY TICKER:")
            print(summary_df.to_string(index=False))
            print(f"\nResults saved to: {self.output_dir}")
            print(f"Database: {self.sentiment_db_path}")
            print(f"Summary: {self.output_dir / 'sentiment_summary_by_ticker.csv'}")
            return summary_df
        except Exception as e:
            logging.error(f"Pipeline failed: {traceback.format_exc()}")
            raise e

def main():
    """Main execution function"""
    print("Initializing Financial Sentiment Analysis Pipeline...")
    print("This may take a few minutes to download models on first run")
    try:
        pipeline_manager = SentimentPipelineManager()
        summary = pipeline_manager.run_complete_analysis()
        print("\nSentiment analysis pipeline completed successfully!")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        logging.error(f"Main execution failed: {traceback.format_exc()}")

if __name__ == "__main__":
    main()