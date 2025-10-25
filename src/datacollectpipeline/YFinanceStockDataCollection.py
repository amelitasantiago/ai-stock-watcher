import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
import os
from typing import List, Dict, Optional, Tuple
import sqlite3
from pathlib import Path
import concurrent.futures
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stock_data_collection.log'),
        logging.StreamHandler()
    ]
)

def load_config():
    """Load configuration from config.json"""
    config_path = Path("src/config/config.json")
    with config_path.open('r') as f:
        return json.load(f)

class StockDataCollector:
    """Comprehensive stock data collection pipeline for Yahoo Finance"""
    
    def __init__(self):
        config = load_config()
        self.data_dir = Path(config['csv_dir'])
        self.data_dir.mkdir(exist_ok=True)
        self.db_path = Path(config['stock_db_path'])
        self.tickers = config['tickers']
        self.years = config['years']
        self.setup_database()
        
    def setup_database(self):
        """Initialize SQLite database for storing stock data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT,
                date DATE,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                start_date DATE,
                end_date DATE,
                records_collected INTEGER,
                status TEXT,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logging.info(f"Database initialized at {self.db_path}")
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate if ticker exists and has data"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return 'symbol' in info or 'shortName' in info
        except Exception as e:
            logging.warning(f"Ticker validation failed for {ticker}: {e}")
            return False
    
    def collect_single_ticker(self, ticker: str, years: int = 5, 
                            include_dividends: bool = True) -> Optional[pd.DataFrame]:
        """Collect data for a single ticker"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365 * years)
            
            logging.info(f"Collecting data for {ticker} from {start_date.date()} to {end_date.date()}")
            
            stock = yf.Ticker(ticker)
            data = stock.history(
                start=start_date,
                end=end_date,
                actions=include_dividends,
                auto_adjust=True,
                back_adjust=True
            )
            
            if data.empty:
                logging.warning(f"No data found for ticker {ticker}")
                return None
            
            data = self._clean_data(data, ticker)
            data['ticker'] = ticker
            data = data.reset_index()
            
            logging.info(f"Successfully collected {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logging.error(f"Error collecting data for {ticker}: {e}")
            return None
    
    def _clean_data(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Clean and validate stock data"""
        data = data.dropna(subset=['Close'])
        numeric_cols = ['Open', 'High', 'Low', 'Close']
        for col in numeric_cols:
            if col in data.columns:
                data = data[data[col] > 0]
        
        if all(col in data.columns for col in numeric_cols):
            valid_ohlc = (
                (data['High'] >= data['Open']) &
                (data['High'] >= data['Close']) &
                (data['High'] >= data['Low']) &
                (data['Low'] <= data['Open']) &
                (data['Low'] <= data['Close'])
            )
            invalid_count = (~valid_ohlc).sum()
            if invalid_count > 0:
                logging.warning(f"Removed {invalid_count} invalid OHLC records for {ticker}")
                data = data[valid_ohlc]
        
        data = data.sort_index()
        return data
    
    def collect_multiple_tickers(self, tickers: List[str], years: int = 5, 
                               delay_seconds: float = 0.3) -> Dict[str, pd.DataFrame]:
        """Collect data for multiple tickers with rate limiting"""
        results = {}
        successful_tickers = []
        failed_tickers = []
        
        logging.info(f"Starting collection for {len(tickers)} tickers")
        
        def process_ticker(ticker):
            ticker = ticker.upper().strip()
            logging.info(f"Processing ticker: {ticker}")
            if not self.validate_ticker(ticker):
                logging.warning(f"Invalid ticker: {ticker}")
                return ticker, None
            data = self.collect_single_ticker(ticker, years)
            if data is not None:
                self.save_to_database(data, ticker)
                csv_path = self.data_dir / f"{ticker}_data.csv"
                data.to_csv(csv_path, index=False)
                return ticker, data
            return ticker, None
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_ticker = {executor.submit(process_ticker, ticker): ticker for ticker in tickers}
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker, data = future.result()
                if data is not None:
                    results[ticker] = data
                    successful_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)
                time.sleep(delay_seconds)
        
        logging.info(f"Collection completed. Successful: {len(successful_tickers)}, Failed: {len(failed_tickers)}")
        if failed_tickers:
            logging.warning(f"Failed tickers: {failed_tickers}")
        
        return results
    
    def save_to_database(self, data: pd.DataFrame, ticker: str):
        """Save data to SQLite database, skipping duplicates."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            db_data = data.copy()
            if 'Date' in db_data.columns:
                db_data['date'] = pd.to_datetime(db_data['Date']).dt.date
            else:
                db_data['date'] = db_data.index.date
            
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in db_data.columns:
                    db_data[new_col] = db_data[old_col]
            
            required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            db_data = db_data[[col for col in required_cols if col in db_data.columns]]
            
            # Check for existing records
            cursor.execute("SELECT COUNT(*) FROM stock_prices WHERE ticker = ?", (ticker,))
            existing_count = cursor.fetchone()[0]
            if existing_count > 0:
                logging.info(f"Skipping database insert for {ticker}: {existing_count} records already exist")
                conn.close()
                return
            
            db_data.to_sql('stock_prices', conn, if_exists='append', index=False)
            
            cursor.execute('''
                INSERT INTO collection_log (ticker, start_date, end_date, records_collected, status)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                ticker,
                db_data['date'].min(),
                db_data['date'].max(),
                len(db_data),
                'SUCCESS'
            ))
            
            conn.commit()
            conn.close()
            logging.info(f"Inserted {len(db_data)} records for {ticker} into database")
    
        except Exception as e:
            logging.error(f"Error saving {ticker} to database: {e}")
            conn.close()
    
    def get_stock_info(self, ticker: str) -> Dict:
        """Get additional stock information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            stock_info = {
                'symbol': info.get('symbol', ticker),
                'company_name': info.get('longName', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD'),
                'exchange': info.get('exchange', 'Unknown')
            }
            return stock_info
        except Exception as e:
            logging.error(f"Error getting info for {ticker}: {e}")
            return {}
    
    def create_summary_report(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a summary report of collected data"""
        summary_data = []
        for ticker, data in data_dict.items():
            info = self.get_stock_info(ticker)
            summary_data.append({
                'ticker': ticker,
                'company_name': info.get('company_name', 'Unknown'),
                'sector': info.get('sector', 'Unknown'),
                'records_count': len(data),
                'date_range_start': data['Date'].min() if 'Date' in data.columns else data.index.min(),
                'date_range_end': data['Date'].max() if 'Date' in data.columns else data.index.max(),
                'avg_daily_volume': data['Volume'].mean() if 'Volume' in data.columns else 0,
                'price_volatility': data['Close'].pct_change().std() if 'Close' in data.columns else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = self.data_dir / "collection_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Summary report saved to {summary_path}")
        return summary_df

def main():
    """Main execution for stock data collection"""
    collector = StockDataCollector()
    results = collector.collect_multiple_tickers(
        tickers=collector.tickers,
        years=collector.years,
        delay_seconds=0.3
    )
    
    if results:
        summary = collector.create_summary_report(results)
        print("\nData Collection Summary:")
        print(summary.to_string(index=False))
        first_ticker = list(results.keys())[0]
        sample_data = results[first_ticker].head()
        print(f"\nSample data for {first_ticker}:")
        print(sample_data.to_string())
    
    print(f"\nData saved to: {collector.data_dir}")
    print(f"Database location: {collector.db_path}")

if __name__ == "__main__":
    main()