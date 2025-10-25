import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta
import logging
import pickle
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import traceback
import warnings
import json
warnings.filterwarnings('ignore')

# Statistical and ML libraries
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available for ARIMA")

# Auto-ARIMA
try:
    import pmdarima as pm
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False
    print("Warning: pmdarima not available - using fixed ARIMA parameters")

# Deep learning libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available for LSTM-Transformer")

# Technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not available, using basic indicators only")

from dataclasses import dataclass
import structlog
from multiprocessing import Pool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/hybrid_model_training.log'),
        logging.StreamHandler()
    ]
)

def load_config():
    """Load configuration from config.json"""
    config_path = Path("config/config.json")
    if not config_path.exists():
        # Create default config if missing
        default_config = {
            "stock_db_path": "data/my_stock_data/stock_data.db",
            "sentiment_db_path": "data/financial_sentiment_results/sentiment_analysis.db", 
            "news_db_path": "data/comprehensive_news_data/comprehensive_news.db",
            "csv_dir": "data/my_stock_data",
            "model_dir": "models",
            "enable_eval": True,
            "sentiment_backfill": True,
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "NFLX", "SPY", "QQQ", "IWM"],
            "company_names": {
                "AAPL": "Apple Inc.",
                "MSFT": "Microsoft Corporation", 
                "GOOGL": "Alphabet Inc.",
                "AMZN": "Amazon.com, Inc.",
                "META": "Meta Platforms, Inc.",
                "TSLA": "Tesla, Inc.",
                "NVDA": "NVIDIA Corporation",
                "NFLX": "Netflix, Inc.",
                "SPY": "SPDR S&P 500 ETF",
                "QQQ": "Invesco QQQ Trust",
                "IWM": "iShares Russell 2000 ETF"
            },
            "years": 5,
            "days_back_news": 30,
            "limit_articles": 500
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open('w') as f:
            json.dump(default_config, f, indent=2)
        logging.info(f"Created default config at {config_path}")
        
    with config_path.open('r') as f:
        return json.load(f)

@dataclass
class ModelConfig:
    """Configuration for the hybrid model"""
    sequence_length: int = 30  # Reduced for stability
    prediction_horizon: int = 3  # Reduced from 5 to 3 for better accuracy
    max_features: int = 8  # Reduced number of features
    test_size: float = 0.2
    validation_size: float = 0.1
    arima_order: tuple = (1, 1, 1)  # Will be overridden by auto-ARIMA
    lstm_hidden_size: int = 32  # Reduced complexity
    transformer_num_heads: int = 2  # Reduced
    transformer_num_layers: int = 1  # Reduced
    dropout_rate: float = 0.2  # Reduced
    learning_rate: float = 0.001  # Increased slightly
    batch_size: int = 32  # Increased
    epochs: int = 50  # Increased
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    arima_weight: float = 0.4
    bayesian_ridge_weight: float = 0.3
    lstm_transformer_weight: float = 0.3
    model_dir: str = "models"
    reports_dir: str = "reports"
    include_technical_indicators: bool = True
    include_sentiment: bool = True
    enable_eval: bool = True
    sentiment_backfill: bool = True
    max_features: int = 15  # Feature selection limit
    use_pca: bool = False  # PCA option
    n_components: float = 0.95  # PCA variance threshold

class DataProcessor:
    """Handles data loading, preprocessing, and feature engineering"""
    
    def __init__(self):
        config = load_config()
        self.stock_db_path = Path(config['stock_db_path']).resolve()
        self.sentiment_db_path = Path(config['sentiment_db_path']).resolve()
        self.csv_dir = Path(config['csv_dir']).resolve()
        self.scalers = {}
        
    def inspect_database(self) -> List[str]:
        """Inspect the SQLite database to verify tables and data."""
        try:
            if not self.stock_db_path.exists():
                logging.warning(f"Stock database not found: {self.stock_db_path}")
                return []
                
            conn = sqlite3.connect(self.stock_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            logging.info(f"Available tables: {tables}")
            cursor.execute("SELECT DISTINCT ticker FROM stock_prices")
            tickers = [t[0] for t in cursor.fetchall()]
            logging.info(f"Available tickers: {tickers}")
            conn.close()
            return tickers
        except sqlite3.Error as e:
            logging.error(f"Database inspection failed: {e}")
            return []
    
    def load_stock_data(self, ticker: str) -> pd.DataFrame:
        """Load stock data for a specific ticker."""
        if not self.stock_db_path.exists():
            logging.error(f"Stock database not found: {self.stock_db_path}")
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.stock_db_path)
            query = """
            SELECT date, open, high, low, close, adj_close, volume
            FROM stock_prices
            WHERE UPPER(ticker) = UPPER(?)
            ORDER BY date ASC
            """
            df = pd.read_sql_query(query, conn, params=(ticker,))
            conn.close()
            
            if df.empty:
                logging.warning(f"No stock data found for ticker: {ticker}")
                return pd.DataFrame()
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            logging.info(f"Loaded {len(df)} records for {ticker}")
            return df
        
        except sqlite3.Error as e:
            logging.error(f"Database error for ticker {ticker}: {e}")
            return pd.DataFrame()
    
    def load_sentiment_data(self, ticker: str) -> pd.DataFrame:
        """Load sentiment data with enhanced debugging and backfill."""
        if not self.sentiment_db_path.exists():
            logging.warning(f"Sentiment database not found: {self.sentiment_db_path}")
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.sentiment_db_path)
            
            # Enhanced query with debugging
            query = """
            SELECT DATE(processing_timestamp) as date,
                   AVG(combined_sentiment) as avg_sentiment,
                   AVG(sentiment_magnitude) as avg_magnitude,
                   COUNT(*) as article_count
            FROM sentiment_scores
            WHERE UPPER(ticker) = UPPER(?)
            GROUP BY DATE(processing_timestamp)
            ORDER BY date ASC
            """
            df = pd.read_sql_query(query, conn, params=(ticker,))
            
            # DEBUG: Log raw sentiment data
            logging.info(f"Raw sentiment query returned {len(df)} rows for {ticker}")
            if not df.empty:
                logging.info(f"Sample sentiment data for {ticker}:")
                logging.info(f"  First 3 rows:\n{df.head(3)}")
                logging.info(f"  Sentiment range: {df['avg_sentiment'].min():.3f} to {df['avg_sentiment'].max():.3f}")
                logging.info(f"  Non-zero sentiment days: {(df['avg_sentiment'] != 0).sum()}")
            
            # Enhanced sparse data handling with forward-fill
            if df.empty or len(df) < 50:  # Reduced threshold
                logging.warning(f"Sparse sentiment for {ticker}; using market sentiment proxy")
                
                # Get stock data date range
                stock_query_min = "SELECT MIN(date) as min_date FROM stock_prices WHERE UPPER(ticker) = UPPER(?)"
                stock_query_max = "SELECT MAX(date) as max_date FROM stock_prices WHERE UPPER(ticker) = UPPER(?)"
                
                stock_conn = sqlite3.connect(self.stock_db_path)
                min_date = pd.to_datetime(pd.read_sql_query(stock_query_min, stock_conn, params=(ticker,)).iloc[0, 0])
                max_date = pd.to_datetime(pd.read_sql_query(stock_query_max, stock_conn, params=(ticker,)).iloc[0, 0])
                stock_conn.close()
                
                # Create market-based sentiment proxy
                dates = pd.date_range(start=min_date, end=max_date, freq='B')
                
                # Use VIX-like volatility proxy for sentiment
                stock_conn = sqlite3.connect(self.stock_db_path)
                volatility_query = """
                SELECT date, close FROM stock_prices 
                WHERE UPPER(ticker) = UPPER(?) 
                ORDER BY date ASC
                """
                vol_df = pd.read_sql_query(volatility_query, stock_conn, params=(ticker,))
                stock_conn.close()
                
                if not vol_df.empty:
                    vol_df['date'] = pd.to_datetime(vol_df['date'])
                    vol_df.set_index('date', inplace=True)
                    vol_df['returns'] = vol_df['close'].pct_change()
                    vol_df['volatility'] = vol_df['returns'].rolling(5).std()
                    
                    # Create sentiment proxy: negative correlation with volatility
                    vol_df['sentiment_proxy'] = -vol_df['volatility'].fillna(0) * 2  # Scale appropriately
                    vol_df['sentiment_proxy'] = vol_df['sentiment_proxy'].clip(-0.5, 0.5)  # Reasonable bounds
                    
                    df = pd.DataFrame({
                        'avg_sentiment': vol_df['sentiment_proxy'].reindex(dates, method='ffill').fillna(0),
                        'avg_magnitude': 0.3,  # Conservative magnitude
                        'article_count': 1
                    }, index=dates)
                else:
                    df = pd.DataFrame({
                        'avg_sentiment': 0.0,
                        'avg_magnitude': 0.3,
                        'article_count': 1
                    }, index=dates)
                
                df.index.name = 'date'
                logging.info(f"Generated sentiment proxy for {len(df)} days for {ticker}")
            else:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                logging.info(f"Using real sentiment data for {len(df)} days for {ticker}")
            
            conn.close()
            
            # Apply enhanced smoothing with forward-fill instead of zero-fill
            df['avg_sentiment'] = df['avg_sentiment'].fillna(method='ffill').fillna(0)
            df['avg_magnitude'] = df['avg_magnitude'].fillna(method='ffill').fillna(0.3)
            df['article_count'] = df['article_count'].fillna(0)
            
            # Apply momentum-based exponential smoothing
            df['avg_sentiment'] = df['avg_sentiment'].ewm(span=3, adjust=False).mean()
            df['avg_magnitude'] = df['avg_magnitude'].ewm(span=3, adjust=False).mean()
            
            non_zero_ratio = (df['avg_sentiment'] != 0).sum() / len(df) * 100
            logging.info(f"Final sentiment for {ticker}: {len(df)} days; non-zero ratio: {non_zero_ratio:.2f}%")
            logging.info(f"Sentiment stats: mean={df['avg_sentiment'].mean():.3f}, std={df['avg_sentiment'].std():.3f}")
            
            return df   
     
        except sqlite3.Error as e:
            logging.error(f"Sentiment database error for ticker {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate essential technical indicators with proper lag."""
        df = df.copy()
        
        # Core price features (lagged to prevent leakage)
        df['returns'] = df['close'].pct_change().shift(1)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).shift(1)
        df['high_low_pct'] = ((df['high'] - df['low']) / df['low']).shift(1)
        df['volume_change'] = df['volume'].pct_change().shift(1)
        
        # Essential moving averages (reduced set)
        for window in [5, 10, 20]:
            df[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean().shift(1)
            df[f'price_vs_sma_{window}'] = (df['close'] / df[f'sma_{window}']).shift(1)
        
        # Volatility measures
        df['volatility_5'] = df['returns'].rolling(window=5, min_periods=1).std().shift(1)
        df['volatility_20'] = df['returns'].rolling(window=20, min_periods=1).std().shift(1)
        
        # Volume indicators
        df['volume_sma_10'] = df['volume'].rolling(window=10, min_periods=1).mean()
        df['volume_ratio'] = (df['volume'] / df['volume_sma_10']).shift(1)
        
        # Essential TA-Lib indicators if available
        if TALIB_AVAILABLE:
            try:
                # RSI (most important)
                df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
                df['rsi'] = df['rsi'].shift(1)
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
                df['macd_histogram'] = pd.Series(macd_hist, index=df.index).shift(1)
                
                # Bollinger Band position only
                upper, middle, lower = talib.BBANDS(df['close'].values)
                bb_position = (df['close'] - lower) / (upper - lower)
                df['bb_position'] = bb_position.shift(1)
                
            except Exception as e:
                logging.warning(f"Error calculating TA-Lib indicators: {e}")
        else:
            # Simple RSI approximation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-8)  # Avoid division by zero
            df['rsi'] = (100 - (100 / (1 + rs))).shift(1)
        
        # Fill NaN values with forward-fill, then backward-fill, finally zero
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove intermediate columns to reduce features
        cols_to_drop = ['volume_sma_10']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
        
        return df
    
    def merge_with_sentiment(self, stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Merge stock data with sentiment features using backward-looking approach."""
        if sentiment_df.empty:
            logging.info("No sentiment data available; using neutral values")
            stock_df['sentiment_t1'] = 0.0
            stock_df['sentiment_3d'] = 0.0
            stock_df['sentiment_trend'] = 0.0
            return stock_df
        
        # Merge data
        merged = stock_df.join(sentiment_df, how="left")
        
        # Create meaningful backward-looking sentiment features
        merged['sentiment_t1'] = merged['avg_sentiment'].shift(1)
        merged['sentiment_3d'] = merged['avg_sentiment'].rolling(3, min_periods=1).mean().shift(1)
        
        # Sentiment trend (change in sentiment)
        merged['sentiment_trend'] = (merged['avg_sentiment'] - merged['avg_sentiment'].shift(3)).shift(1)

        # Drop raw columns to prevent leakage
        merged = merged.drop(columns=["avg_sentiment", "avg_magnitude", "article_count"], errors='ignore')

        # Fill remaining NaN values with forward-fill
        sentiment_cols = ['sentiment_t1', 'sentiment_3d', 'sentiment_trend']
        for col in sentiment_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(method='ffill').fillna(0)

        sentiment_coverage = (merged["sentiment_t1"] != 0).sum()
        logging.info(f"Sentiment features: {sentiment_coverage} days with non-zero sentiment")
        return merged        
    
    def prepare_features(self, ticker: str, config: ModelConfig) -> pd.DataFrame:
        """Prepare all features for a ticker with enhanced feature selection."""
        stock_df = self.load_stock_data(ticker)
        if stock_df.empty:
            logging.warning(f"Skipping feature preparation for {ticker} due to empty stock data")
            return pd.DataFrame()
        
        logging.info(f"Initial stock data shape for {ticker}: {stock_df.shape}")
        
        # Calculate technical indicators
        if config.include_technical_indicators:
            stock_df = self.calculate_technical_indicators(stock_df)
            logging.info(f"After technical indicators: {stock_df.shape}")
        
        # Merge sentiment data
        if config.include_sentiment:
            sentiment_df = self.load_sentiment_data(ticker)
            stock_df = self.merge_with_sentiment(stock_df, sentiment_df)
            logging.info(f"After sentiment merge: {stock_df.shape}")
        
        # Create more stable return targets
        for horizon in range(1, config.prediction_horizon + 1):
            future_price = stock_df['close'].shift(-horizon)
            # Use simple returns instead of log returns for better stability
            stock_df[f'target_{horizon}d'] = (future_price - stock_df['close']) / stock_df['close']
            # Cap extreme values
            stock_df[f'target_{horizon}d'] = stock_df[f'target_{horizon}d'].clip(-0.2, 0.2)
        
        # Remove rows with NaN targets
        target_columns = [f'target_{i}d' for i in range(1, config.prediction_horizon + 1)]
        before_drop = len(stock_df)
        stock_df = stock_df.dropna(subset=target_columns)
        after_drop = len(stock_df)
        logging.info(f"Dropped {before_drop - after_drop} rows with NaN targets")
        
        # Remove columns with too many NaN values or zero variance
        feature_cols = [col for col in stock_df.columns if not col.startswith('target_')]
        for col in feature_cols:
            if stock_df[col].var() < 1e-8:  # Remove zero variance columns
                stock_df = stock_df.drop(columns=[col])
                logging.info(f"Removed zero variance column: {col}")
        
        # Fill remaining NaN values in features
        feature_cols = [col for col in stock_df.columns if not col.startswith('target_')]
        stock_df[feature_cols] = stock_df[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove infinite values
        stock_df = stock_df.replace([np.inf, -np.inf], 0)
        
        if stock_df.empty:
            logging.warning(f"Feature set for {ticker} is empty after processing")
        else:
            logging.info(f"Final feature set for {ticker}: {stock_df.shape}")
        
        return stock_df

class LSTMTransformer(nn.Module):
    """Simplified LSTM + Transformer hybrid"""
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1,
                 num_heads: int = 2, dropout: float = 0.2, prediction_horizon: int = 3):
        super(LSTMTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads, dropout=dropout, 
            batch_first=True, activation='gelu', dim_feedforward=hidden_size*2
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Output layers with regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, prediction_horizon)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for better training stability."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Transformer processing
        transformer_out = self.transformer_encoder(lstm_out)
        
        # Get final hidden state and apply regularization
        final_hidden = transformer_out[:, -1, :]
        final_hidden = self.layer_norm(final_hidden)
        final_hidden = self.dropout(final_hidden)
        
        # Two-layer output with activation
        out1 = torch.relu(self.fc1(final_hidden))
        out1 = self.dropout(out1)
        output = self.fc2(out1)
        return output

class HybridStockPredictor:
    """Enhanced Hybrid Stock Predictor with critical fixes"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_columns = []
        self.target_columns = [f'target_{i}d' for i in range(1, config.prediction_horizon + 1)]
        self.arima_model = None
        self.lstm_model = None
        self.bayesian_models = {}
        self.lstm_scalers = {}
        self.scalers = {}
        self.ensemble_weights = {}
        self.model_performance = {}  # Track model performance for dynamic weighting
        
    def create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        if len(X) < sequence_length:
            logging.warning(f"Insufficient data for sequences: {len(X)} < {sequence_length}")
            return np.array([]).reshape(0, sequence_length, X.shape[1]), np.array([]).reshape(0, y.shape[1])
        
        X_seqs, y_seqs = [], []
        for i in range(sequence_length, len(X)):
            X_seqs.append(X[i-sequence_length:i])
            y_seqs.append(y[i])
        
        return np.array(X_seqs), np.array(y_seqs)
    
    def train_arima(self, price_series: pd.Series) -> Tuple[bool, float]:
        """Train ARIMA model with auto-parameter selection."""
        if not STATSMODELS_AVAILABLE:
            logging.warning("ARIMA training skipped - statsmodels not available")
            return False, np.inf
        
        try:
            # Use log returns for stationarity
            returns = price_series.pct_change().dropna()
            if len(returns) < 100:
                logging.warning("Insufficient data for ARIMA")
                return False, np.inf
            
            # Auto-ARIMA if available, otherwise use fixed parameters
            if PMDARIMA_AVAILABLE:
                logging.info("Using Auto-ARIMA for parameter selection")
                auto_model = pm.auto_arima(
                    returns, 
                    seasonal=False, 
                    stepwise=True,
                    suppress_warnings=True, 
                    error_action="ignore",
                    max_p=3, max_q=3, max_d=2,  # Limit complexity
                    information_criterion='aic'
                )
                self.arima_model = auto_model
                aic_score = auto_model.aic()
                logging.info(f"Auto-ARIMA selected order: {auto_model.order}, AIC: {aic_score:.2f}")
            else:
                # Fallback to fixed parameters
                arima = ARIMA(returns, order=self.config.arima_order)
                self.arima_model = arima.fit()
                aic_score = self.arima_model.aic
                logging.info(f"Fixed ARIMA{self.config.arima_order}, AIC: {aic_score:.2f}")
            
            # Calculate validation error for weighting
            try:
                # Simple out-of-sample test
                split_point = int(len(returns) * 0.8)
                train_data = returns[:split_point]
                test_data = returns[split_point:]
                
                if PMDARIMA_AVAILABLE:
                    val_model = pm.auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True)
                    forecast = val_model.predict(n_periods=len(test_data))
                else:
                    val_model = ARIMA(train_data, order=self.config.arima_order).fit()
                    forecast = val_model.forecast(len(test_data))
                
                val_rmse = np.sqrt(mean_squared_error(test_data, forecast))
                self.model_performance['arima'] = val_rmse
                logging.info(f"ARIMA validation RMSE: {val_rmse:.4f}")
                
            except Exception as e:
                logging.warning(f"ARIMA validation failed: {e}")
                self.model_performance['arima'] = np.inf
            
            return True, aic_score
            
        except Exception as e:
            logging.error(f"ARIMA training failed: {e}")
            self.arima_model = None
            self.model_performance['arima'] = np.inf
            return False, np.inf
    
    def train_bayesian_ridge(self, X: np.ndarray, y_multi: np.ndarray) -> Tuple[Dict[int, object], List[float]]:
        """Train enhanced regression models with feature selection and regularization."""
        models = {}
        r2_scores = []
        
        if X.shape[0] < 50:
            logging.warning("Insufficient data for regression training")
            return models, r2_scores
        
        logging.info(f"Training regression with {X.shape[1]} features, {X.shape[0]} samples")
        
        # Enhanced feature selection and preprocessing
        try:
            # Remove highly correlated features
            corr_matrix = pd.DataFrame(X).corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]
            
            if high_corr_pairs:
                # Remove some highly correlated features
                keep_indices = [i for i in range(X.shape[1]) if i not in range(len(high_corr_pairs))]
                X = X[:, keep_indices]
                logging.info(f"Removed {len(high_corr_pairs)} highly correlated features")
            
            # Feature selection using univariate selection
            if X.shape[1] > self.config.max_features:
                selector = SelectKBest(score_func=f_regression, k=min(self.config.max_features, X.shape[1]))
                X_selected = selector.fit_transform(X, y_multi[:, 0])  # Use first horizon for selection
                selected_features = selector.get_support()
                logging.info(f"Selected {X_selected.shape[1]} out of {X.shape[1]} features using univariate selection")
                X = X_selected
            
            # Robust scaling to handle outliers
            feature_scaler = RobustScaler()
            X_scaled = feature_scaler.fit_transform(X)
            
            # Time-based split (crucial for time series)
            split_idx = int(len(X_scaled) * (1 - self.config.test_size))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            
            validation_rmses = []
            
            for h in range(1, min(self.config.prediction_horizon + 1, y_multi.shape[1] + 1)):
                try:
                    y_h = y_multi[:, h-1]
                    y_train, y_test = y_h[:split_idx], y_h[split_idx:]
                    
                    # Try multiple model types and select the best
                    model_candidates = [
                        ('BayesianRidge', BayesianRidge(
                            #alpha_1=1e-4, alpha_2=1e-4, lambda_1=1e-4, lambda_2=1e-4,
                            #fit_intercept=True, compute_score=True
                            alpha_1=1e-4, alpha_2=1e-4, lambda_1=1e-4, lambda_2=1e-4,
                            fit_intercept=True
                        )),
                        ('ElasticNet', ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000)),
                        ('Ridge', Ridge(alpha=1.0))
                    ]
                    
                    best_model = None
                    best_score = -np.inf
                    best_name = None
                    
                    for name, model in model_candidates:
                        try:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            
                            # Use R² as primary metric, but ensure it's reasonable
                            r2_h = r2_score(y_test, y_pred)
                            mae_h = mean_absolute_error(y_test, y_pred)
                            rmse_h = np.sqrt(mean_squared_error(y_test, y_pred))
                            
                            # Penalize if R² is negative (worse than predicting mean)
                            if r2_h < -0.5:
                                continue
                                
                            if r2_h > best_score:
                                best_score = r2_h
                                best_model = model
                                best_name = name
                            
                            logging.info(f"{name} horizon {h}: R²={r2_h:.3f}, MAE={mae_h:.4f}, RMSE={rmse_h:.4f}")
                            
                        except Exception as e:
                            logging.warning(f"{name} failed for horizon {h}: {e}")
                            continue
                    
                    if best_model is not None:
                        models[h] = best_model
                        r2_scores.append(best_score)
                        validation_rmses.append(np.sqrt(mean_squared_error(y_test, best_model.predict(X_test))))
                        logging.info(f"Selected {best_name} for horizon {h} with R²={best_score:.3f}")
                    else:
                        logging.warning(f"No suitable model found for horizon {h}")
                        
                except Exception as e:
                    logging.error(f"Model training failed for horizon {h}: {e}")
                    continue
            
            if models:
                self.scalers['ridge_features'] = feature_scaler
                if high_corr_pairs:
                    self.scalers['feature_mask'] = keep_indices
                if X.shape[1] != X_scaled.shape[1]:
                    self.scalers['feature_selector'] = selector
                
                avg_r2 = np.mean(r2_scores) if r2_scores else -1
                avg_rmse = np.mean(validation_rmses) if validation_rmses else np.inf
                self.model_performance['ridge'] = avg_rmse
                
                logging.info(f"Regression models: {len(models)} horizons trained, avg R²: {avg_r2:.3f}, avg RMSE: {avg_rmse:.4f}")
            
        except Exception as e:
            logging.error(f"Feature processing failed: {e}")
            return {}, []
        
        return models, r2_scores
    
    def train_lstm_transformer(self, X_seq: np.ndarray, y_seq: np.ndarray) -> Optional[Tuple]:
        """Train LSTM-Transformer with enhanced stability and learning rate scheduling."""
        if not TORCH_AVAILABLE:
            logging.warning("PyTorch unavailable; skipping LSTM-Transformer")
            return None
        
        try:
            n_samples, seq_len, n_features = X_seq.shape
            logging.info(f"Training LSTM on {n_samples} sequences, {seq_len} steps, {n_features} features")
            
            if n_samples < 100:
                logging.warning("Insufficient sequences for LSTM training")
                return None
            
            # Time-based splits
            n_train = int(n_samples * 0.7)
            n_val = int(n_samples * 0.15)
            
            X_train = X_seq[:n_train]
            X_val = X_seq[n_train:n_train+n_val]
            X_test = X_seq[n_train+n_val:]
            
            y_train = y_seq[:n_train]
            y_val = y_seq[n_train:n_train+n_val]
            y_test = y_seq[n_train+n_val:]
            
            # Robust scaling for better training
            scaler_X = RobustScaler()
            scaler_y = StandardScaler()
            
            # Fit scalers on training data only
            X_train_2d = X_train.reshape(-1, n_features)
            scaler_X.fit(X_train_2d)
            scaler_y.fit(y_train)
            
            # Transform all splits
            X_train = scaler_X.transform(X_train_2d).reshape(X_train.shape)
            X_val = scaler_X.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
            X_test = scaler_X.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
            
            y_train = scaler_y.transform(y_train)
            y_val = scaler_y.transform(y_val)
            y_test = scaler_y.transform(y_test)
            
            # Create data loaders with increased batch size for stability
            train_dataset = list(zip(torch.FloatTensor(X_train), torch.FloatTensor(y_train)))
            val_dataset = list(zip(torch.FloatTensor(X_val), torch.FloatTensor(y_val)))
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Initialize model with reduced complexity
            model = LSTMTransformer(
                input_size=n_features,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.transformer_num_layers,
                num_heads=self.config.transformer_num_heads,
                dropout=self.config.dropout_rate,
                prediction_horizon=self.config.prediction_horizon
            )
            
            # Enhanced training setup
            optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
            #scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                #optimizer, mode='min', patience=7, factor=0.7, verbose=True
            #)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=7, factor=0.7
            )
            criterion = nn.MSELoss()
            
            # Training loop with enhanced early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 15
            train_losses = []
            val_losses = []
            
            for epoch in range(self.config.epochs):
                # Training
                model.train()
                train_loss = 0.0
                batch_count = 0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    # Check for NaN loss
                    if torch.isnan(loss):
                        logging.warning("NaN loss detected, skipping batch")
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    batch_count += 1
                
                if batch_count == 0:
                    logging.error("No valid batches in training")
                    break
                    
                train_loss /= batch_count
                train_losses.append(train_loss)
                
                # Validation
                model.eval()
                val_loss = 0.0
                val_batch_count = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        if not torch.isnan(loss):
                            val_loss += loss.item()
                            val_batch_count += 1
                
                if val_batch_count == 0:
                    logging.warning("No valid validation batches")
                    val_loss = float('inf')
                else:
                    val_loss /= val_batch_count
                    
                val_losses.append(val_loss)
                scheduler.step(val_loss)
                
                # Enhanced logging
                if epoch % 10 == 0 or epoch < 5:
                    logging.info(f"[LSTM] Epoch {epoch+1}/{self.config.epochs}: "
                               f"Train={train_loss:.4f}, Val={val_loss:.4f}, "
                               f"LR={optimizer.param_groups[0]['lr']:.2e}")
                
                # Early stopping with improved logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(f"[LSTM] Early stopping at epoch {epoch+1}")
                        break
                
                # Stop if loss is not decreasing for too long
                if epoch > 20 and np.mean(train_losses[-10:]) > np.mean(train_losses[-20:-10]):
                    logging.info("[LSTM] Training loss stopped improving, early stop")
                    break
            
            # Restore best model
            if 'best_model_state' in locals():
                model.load_state_dict(best_model_state)
                logging.info("Restored best model state")
            
            # Store scalers
            self.lstm_scalers = {'X': scaler_X, 'y': scaler_y}
            
            # Enhanced evaluation on test set
            model.eval()
            test_predictions = []
            test_targets = []
            
            with torch.no_grad():
                test_tensor = torch.FloatTensor(X_test)
                test_outputs = model(test_tensor)
                
                # Inverse transform for evaluation
                test_pred_original = scaler_y.inverse_transform(test_outputs.numpy())
                test_true_original = scaler_y.inverse_transform(y_test)
                
                test_rmse = np.sqrt(mean_squared_error(test_true_original, test_pred_original))
                test_r2 = r2_score(test_true_original, test_pred_original)
                
                self.model_performance['lstm'] = test_rmse
            
            metrics = {
                'best_val_loss': best_val_loss,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'epochs_trained': epoch + 1,
                'final_lr': optimizer.param_groups[0]['lr']
            }
            
            logging.info(f"LSTM training complete: Test RMSE={test_rmse:.4f}, Test R²={test_r2:.3f}")
            return model, optimizer, metrics
            
        except Exception as e:
            logging.error(f"LSTM training failed: {e}")
            traceback.print_exc()
            self.model_performance['lstm'] = np.inf
            return None
    
    def calculate_dynamic_weights(self) -> Dict[str, float]:
        """Calculate ensemble weights based on model performance."""
        if not self.model_performance:
            # Fallback to equal weights if no performance data
            available_models = sum([
                1 if self.arima_model else 0,
                1 if self.bayesian_models else 0,
                1 if self.lstm_model else 0
            ])
            equal_weight = 1.0 / max(available_models, 1)
            return {
                'arima': equal_weight if self.arima_model else 0,
                'ridge': equal_weight if self.bayesian_models else 0,
                'lstm': equal_weight if self.lstm_model else 0
            }
        
        # Calculate weights based on inverse RMSE (lower RMSE = higher weight)
        weights = {}
        for model_name in ['arima', 'ridge', 'lstm']:
            if model_name in self.model_performance:
                rmse = self.model_performance[model_name]
                if rmse < np.inf and rmse > 0:
                    weights[model_name] = 1.0 / (rmse + 1e-8)  # Add small epsilon
                else:
                    weights[model_name] = 0.0
            else:
                weights[model_name] = 0.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # All models failed - fallback
            weights = {'arima': 0.33, 'ridge': 0.33, 'lstm': 0.34}
        
        # Zero out weights for unavailable models
        if not self.arima_model:
            weights['arima'] = 0
        if not self.bayesian_models:
            weights['ridge'] = 0
        if not self.lstm_model:
            weights['lstm'] = 0
            
        # Renormalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def train(self, ticker: str, data_processor: DataProcessor) -> bool:
        """Main training method for hybrid model with comprehensive improvements."""
        try:
            logging.info(f"Starting enhanced hybrid model training for {ticker}")
            
            # Load and prepare features
            df = data_processor.prepare_features(ticker, self.config)
            if df.empty:
                logging.warning(f"No data available for {ticker}")
                return False
            
            logging.info(f"Loaded {len(df)} records for {ticker}")
            
            # Prepare features and targets
            target_columns = [col for col in df.columns if col.startswith('target_')]
            feature_columns = [col for col in df.columns 
                             if not col.startswith('target_') and col not in ['date', 'ticker']]
            
            # Filter numeric columns and remove problematic ones
            numeric_features = []
            for col in feature_columns:
                if df[col].dtype in ['int64', 'float64']:
                    if df[col].var() > 1e-8:  # Remove zero variance
                        if not df[col].isnull().all():  # Remove all-null columns
                            numeric_features.append(col)
            
            self.feature_columns = numeric_features
            
            if len(self.feature_columns) == 0:
                logging.error(f"No valid numeric features available for {ticker}")
                return False
            
            logging.info(f"Using {len(self.feature_columns)} features for {ticker}")
            
            X_features = df[self.feature_columns].values
            y_targets = df[target_columns].values
            
            # Enhanced data validation
            if X_features.shape[0] < 200:  # Increased minimum
                logging.warning(f"Insufficient data for {ticker}: {X_features.shape[0]} samples")
                return False
            
            # Check for data quality issues
            nan_ratio = np.isnan(X_features).sum() / X_features.size
            if nan_ratio > 0.1:
                logging.warning(f"High NaN ratio in features: {nan_ratio:.2%}")
            
            inf_ratio = np.isinf(X_features).sum() / X_features.size
            if inf_ratio > 0:
                logging.warning(f"Infinite values in features: {inf_ratio:.2%}")
                X_features = np.nan_to_num(X_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Train models
            models_trained = 0
            
            # 1. Enhanced ARIMA Training
            if STATSMODELS_AVAILABLE and 'close' in df.columns:
                success, aic = self.train_arima(df['close'])
                if success:
                    models_trained += 1
                    logging.info(f"ARIMA trained successfully (AIC: {aic:.2f})")
            
            # 2. Enhanced Regression Training
            try:
                self.bayesian_models, r2_scores = self.train_bayesian_ridge(X_features, y_targets)
                if self.bayesian_models:
                    models_trained += 1
                    avg_r2 = np.mean(r2_scores) if r2_scores else 0
                    logging.info(f"Regression models trained (avg R²: {avg_r2:.3f})")
            except Exception as e:
                logging.error(f"Regression training failed: {e}")
                self.bayesian_models = {}
            
            # 3. Enhanced LSTM-Transformer Training
            if TORCH_AVAILABLE and len(X_features) > self.config.sequence_length * 3:
                try:
                    X_seq, y_seq = self.create_sequences(X_features, y_targets, self.config.sequence_length)
                    if X_seq.shape[0] > 100:  # Increased minimum sequences
                        lstm_results = self.train_lstm_transformer(X_seq, y_seq)
                        if lstm_results:
                            self.lstm_model = lstm_results[0]
                            models_trained += 1
                            logging.info("LSTM-Transformer trained successfully")
                except Exception as e:
                    logging.error(f"LSTM training failed: {e}")
                    self.lstm_model = None
            
            # Enhanced ensemble weighting
            if models_trained == 0:
                logging.error(f"No models successfully trained for {ticker}")
                return False
            
            # Use dynamic weighting based on performance
            self.ensemble_weights = self.calculate_dynamic_weights()
            
            logging.info(f"Dynamic ensemble weights: {self.ensemble_weights}")
            logging.info(f"Successfully trained {models_trained} out of 3 models for {ticker}")
            
            # Performance summary
            if self.model_performance:
                logging.info("Model Performance Summary:")
                for model_name, rmse in self.model_performance.items():
                    if rmse < np.inf:
                        logging.info(f"  {model_name}: RMSE = {rmse:.4f}")
            
            return True
            
        except Exception as e:
            logging.error(f"Training failed for {ticker}: {e}")
            traceback.print_exc()
            return False
    
    def predict(self, recent_data: np.ndarray, price_series: pd.Series = None) -> Dict[str, np.ndarray]:
        """Generate ensemble predictions with error handling."""
        predictions = {}
        
        # ARIMA predictions
        if self.arima_model is not None:
            try:
                if PMDARIMA_AVAILABLE:
                    arima_pred = self.arima_model.predict(n_periods=self.config.prediction_horizon)
                else:
                    arima_pred = self.arima_model.forecast(self.config.prediction_horizon)
                predictions['arima'] = np.array(arima_pred)
            except Exception as e:
                logging.error(f"ARIMA prediction failed: {e}")
                predictions['arima'] = np.zeros(self.config.prediction_horizon)
        else:
            predictions['arima'] = np.zeros(self.config.prediction_horizon)
        
        # Regression predictions
        if self.bayesian_models and 'ridge_features' in self.scalers:
            try:
                recent_flat = recent_data[-1:] if len(recent_data.shape) > 1 else recent_data.reshape(1, -1)
                
                # Apply feature transformations in the same order as training
                if 'feature_mask' in self.scalers:
                    recent_flat = recent_flat[:, self.scalers['feature_mask']]
                
                if 'feature_selector' in self.scalers:
                    recent_flat = self.scalers['feature_selector'].transform(recent_flat)
                
                recent_scaled = self.scalers['ridge_features'].transform(recent_flat)
                
                ridge_preds = []
                for h in range(1, self.config.prediction_horizon + 1):
                    if h in self.bayesian_models:
                        pred = self.bayesian_models[h].predict(recent_scaled)[0]
                        ridge_preds.append(pred)
                    else:
                        ridge_preds.append(0.0)
                
                predictions['ridge'] = np.array(ridge_preds)
            except Exception as e:
                logging.error(f"Ridge prediction failed: {e}")
                predictions['ridge'] = np.zeros(self.config.prediction_horizon)
        else:
            predictions['ridge'] = np.zeros(self.config.prediction_horizon)
        
        # LSTM predictions
        if self.lstm_model is not None and TORCH_AVAILABLE:
            try:
                self.lstm_model.eval()
                with torch.no_grad():
                    if len(recent_data.shape) == 3:
                        input_tensor = torch.FloatTensor(recent_data[-1:])
                    else:
                        seq_len = min(self.config.sequence_length, recent_data.shape[0])
                        input_seq = recent_data[-seq_len:].reshape(1, seq_len, -1)
                        
                        # Apply LSTM scaling
                        if 'X' in self.lstm_scalers:
                            input_scaled = self.lstm_scalers['X'].transform(
                                input_seq.reshape(-1, input_seq.shape[-1])
                            ).reshape(input_seq.shape)
                            input_tensor = torch.FloatTensor(input_scaled)
                        else:
                            input_tensor = torch.FloatTensor(input_seq)
                    
                    lstm_pred = self.lstm_model(input_tensor)
                    
                    # Inverse transform if scaler available
                    if 'y' in self.lstm_scalers:
                        lstm_pred_np = self.lstm_scalers['y'].inverse_transform(
                            lstm_pred.numpy()
                        )[0]
                    else:
                        lstm_pred_np = lstm_pred.numpy()[0]
                    
                    predictions['lstm'] = lstm_pred_np
            except Exception as e:
                logging.error(f"LSTM prediction failed: {e}")
                predictions['lstm'] = np.zeros(self.config.prediction_horizon)
        else:
            predictions['lstm'] = np.zeros(self.config.prediction_horizon)
        
        # Ensemble prediction using dynamic weights
        ensemble_pred = (
            predictions['arima'] * self.ensemble_weights.get('arima', 0) +
            predictions['ridge'] * self.ensemble_weights.get('ridge', 0) +
            predictions['lstm'] * self.ensemble_weights.get('lstm', 0)
        )
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def generate_evaluation_report(self, ticker: str) -> Dict:
        """Generate comprehensive evaluation report."""
        report = {
            'ticker': ticker,
            'models_trained': 0,
            'model_performance': self.model_performance.copy(),
            'ensemble_weights': self.ensemble_weights.copy(),
            'feature_count': len(self.feature_columns),
            'prediction_horizon': self.config.prediction_horizon
        }
        
        if self.arima_model:
            report['models_trained'] += 1
            report['arima_available'] = True
        
        if self.bayesian_models:
            report['models_trained'] += 1
            report['ridge_available'] = True
            report['ridge_horizons'] = list(self.bayesian_models.keys())
        
        if self.lstm_model:
            report['models_trained'] += 1
            report['lstm_available'] = True
        
        return report
    
    def save_model(self, filepath: str):
        """Save trained model components."""
        try:
            # Prepare model data
            model_data = {
                'config': self.config,
                'scalers': self.scalers,
                'lstm_scalers': self.lstm_scalers,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'arima_model': self.arima_model,
                'bayesian_models': self.bayesian_models,
                'ensemble_weights': self.ensemble_weights,
                'model_performance': self.model_performance
            }
            
            # Save main model
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save LSTM model separately
            if self.lstm_model is not None and TORCH_AVAILABLE:
                torch_path = f"{filepath}_lstm_transformer.pth"
                torch.save({
                    'model_state_dict': self.lstm_model.state_dict(),
                    'model_config': {
                        'input_size': len(self.feature_columns),
                        'hidden_size': self.config.lstm_hidden_size,
                        'num_layers': self.config.transformer_num_layers,
                        'num_heads': self.config.transformer_num_heads,
                        'dropout': self.config.dropout_rate,
                        'prediction_horizon': self.config.prediction_horizon
                    }
                }, torch_path)
                logging.info(f"LSTM model saved to {torch_path}")
            
            logging.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Model save failed: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load trained model components."""
        try:
            # Load main model data
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.config = model_data['config']
            self.scalers = model_data.get('scalers', {})
            self.lstm_scalers = model_data.get('lstm_scalers', {})
            self.feature_columns = model_data['feature_columns']
            self.target_columns = model_data.get('target_columns', self.target_columns)
            self.arima_model = model_data['arima_model']
            self.bayesian_models = model_data['bayesian_models']
            self.ensemble_weights = model_data.get('ensemble_weights', {})
            self.model_performance = model_data.get('model_performance', {})
            
            # Load LSTM model if exists
            torch_path = f"{filepath}_lstm_transformer.pth"
            if Path(torch_path).exists() and TORCH_AVAILABLE:
                checkpoint = torch.load(torch_path, map_location='cpu')
                model_config = checkpoint['model_config']
                
                self.lstm_model = LSTMTransformer(**model_config)
                self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f"LSTM model loaded from {torch_path}")
            else:
                self.lstm_model = None
            
            logging.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logging.error(f"Model load failed: {e}")
            raise

def populate_stock_data_from_csv(tickers: List[str], db_path: str, csv_dir: str) -> bool:
    """Populate database from CSV files."""
    try:
        Path(db_path).parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, date)
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON stock_prices(ticker)")
        
        records_inserted = 0
        for ticker in tickers:
            csv_path = Path(csv_dir) / f"{ticker}_data.csv"
            if not csv_path.exists():
                logging.warning(f"CSV file not found: {csv_path}")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                if df.empty:
                    continue
                
                # Standardize columns
                expected_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                if not all(col in df.columns for col in expected_cols):
                    logging.warning(f"Invalid CSV schema for {ticker}")
                    continue
                
                df['ticker'] = ticker
                df = df[['ticker'] + expected_cols]
                df.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                
                df.to_sql('stock_prices', conn, if_exists='append', index=False)
                records_inserted += len(df)
                logging.info(f"Inserted {len(df)} records for {ticker}")
                
            except Exception as e:
                logging.error(f"Error processing {ticker}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logging.info(f"Total records inserted: {records_inserted}")
        return records_inserted > 0
        
    except Exception as e:
        logging.error(f"CSV population failed: {e}")
        return False

def populate_stock_data_yfinance(tickers: List[str], db_path: str) -> bool:
    """Populate database using yfinance."""
    try:
        import yfinance as yf
        
        Path(db_path).parent.mkdir(exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adj_close REAL,
            volume INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (ticker, date)
        )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON stock_prices(ticker)")
        
        records_inserted = 0
        for ticker in tickers:
            try:
                logging.info(f"Fetching data for {ticker}...")
                stock = yf.Ticker(ticker)
                df = stock.history(start="2020-01-01", end="2025-01-01")
                
                if df.empty:
                    logging.warning(f"No data for {ticker}")
                    continue
                
                df = df.reset_index()
                df['ticker'] = ticker
                df = df[['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Close', 'Volume']]
                df.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                
                df.to_sql('stock_prices', conn, if_exists='append', index=False)
                records_inserted += len(df)
                logging.info(f"Inserted {len(df)} records for {ticker}")
                
            except Exception as e:
                logging.error(f"Error fetching {ticker}: {e}")
                continue
        
        conn.commit()
        conn.close()
        
        logging.info(f"Total records inserted via yfinance: {records_inserted}")
        return records_inserted > 0
        
    except ImportError:
        logging.error("yfinance not available")
        return False
    except Exception as e:
        logging.error(f"yfinance population failed: {e}")
        return False

def train_ticker(args):
    """Train model for a single ticker with enhanced reporting."""
    try:
        ticker, config, stock_db_path, sentiment_db_path = args
        logging.info(f"Starting enhanced training for {ticker}")
        
        # Initialize components
        data_processor = DataProcessor()
        predictor = HybridStockPredictor(config)
        
        # Train the model
        success = predictor.train(ticker, data_processor)
        
        if success:
            # Generate evaluation report
            report = predictor.generate_evaluation_report(ticker)
            
            # Save model
            model_dir = Path(config.model_dir)
            model_dir.mkdir(exist_ok=True, parents=True)
            
            model_path = model_dir / f"hybrid_model_{ticker}.pkl"
            predictor.save_model(str(model_path))
            
            # Save report
            reports_dir = Path(config.reports_dir)
            reports_dir.mkdir(exist_ok=True, parents=True)
            report_path = reports_dir / f"training_report_{ticker}.json"
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logging.info(f"Model and report saved for {ticker}")
            return f"SUCCESS: {ticker} - {report['models_trained']} models trained", report
        else:
            return f"FAILED: {ticker} - No models trained successfully", {}
    
    except Exception as e:
        logging.error(f"Training error for {ticker}: {e}")
        return f"ERROR: {ticker} - {str(e)}", {}

def main():
    """Enhanced main training function with comprehensive reporting."""
    try:
        # Load configuration
        config_data = load_config()
        config = ModelConfig()
        
        # Setup directories
        for directory in ['logs', config.model_dir, config.reports_dir]:
            Path(directory).mkdir(exist_ok=True)
        
        # Setup paths
        stock_db_path = Path(config_data['stock_db_path']).resolve()
        sentiment_db_path = Path(config_data['sentiment_db_path']).resolve()
        csv_dir = Path(config_data['csv_dir']).resolve()
        tickers = config_data['tickers']
        
        print(f"\n{'='*60}")
        print("ENHANCED HYBRID STOCK PREDICTOR")
        print(f"{'='*60}")
        print(f"Training Configuration:")
        print(f"  • Tickers: {len(tickers)} ({', '.join(tickers)})")
        print(f"  • Prediction Horizon: {config.prediction_horizon} days")
        print(f"  • Sequence Length: {config.sequence_length} days")
        print(f"  • Max Features: {config.max_features}")
        print(f"  • Models: ARIMA, Regression, LSTM-Transformer")
        print(f"{'='*60}")
        
        # Check database and populate if needed
        data_processor = DataProcessor()
        available_tickers = data_processor.inspect_database()
        
        missing_tickers = [t for t in tickers if t.upper() not in [at.upper() for at in available_tickers]]
        
        if missing_tickers:
            print(f"\nPopulating missing data for: {missing_tickers}")
            
            # Try CSV first, then yfinance
            if csv_dir.exists():
                print("Attempting CSV population...")
                if not populate_stock_data_from_csv(missing_tickers, str(stock_db_path), str(csv_dir)):
                    print("CSV failed, trying yfinance...")
                    populate_stock_data_yfinance(missing_tickers, str(stock_db_path))
            else:
                print("No CSV directory, trying yfinance...")
                populate_stock_data_yfinance(missing_tickers, str(stock_db_path))
        
        # Re-check available tickers
        available_tickers = data_processor.inspect_database()
        valid_tickers = [t for t in tickers if t.upper() in [at.upper() for at in available_tickers]]
        
        if not valid_tickers:
            print("\nERROR: No data available for training")
            return
        
        print(f"\nTraining {len(valid_tickers)} valid tickers...")
        
        # Train models with enhanced reporting
        successful_tickers = []
        failed_tickers = []
        training_reports = {}
        
        for i, ticker in enumerate(valid_tickers, 1):
            print(f"\n[{i}/{len(valid_tickers)}] Training {ticker}...")
            
            args = (ticker, config, str(stock_db_path), str(sentiment_db_path))
            result_message, report = train_ticker(args)
            
            if "SUCCESS" in result_message:
                successful_tickers.append(ticker)
                training_reports[ticker] = report
                print(f"  ✓ {ticker}: {report.get('models_trained', 0)} models trained")
                if report.get('model_performance'):
                    for model_name, rmse in report['model_performance'].items():
                        if rmse < np.inf:
                            print(f"    - {model_name}: RMSE = {rmse:.4f}")
            else:
                failed_tickers.append(ticker)
                print(f"  ✗ {ticker}: Training failed")
        
        # Generate comprehensive summary
        print(f"\n{'='*60}")
        print("TRAINING RESULTS SUMMARY")
        print(f"{'='*60}")
        
        if successful_tickers:
            print(f"✓ Successfully trained: {len(successful_tickers)} tickers")
            print(f"  Models directory: {config.model_dir}")
            print(f"  Reports directory: {config.reports_dir}")
            
            # Performance summary table
            print(f"\n{'Ticker':<8} {'Models':<8} {'ARIMA':<8} {'Ridge':<8} {'LSTM':<8}")
            print("-" * 50)
            
            for ticker in successful_tickers:
                report = training_reports.get(ticker, {})
                models_count = report.get('models_trained', 0)
                
                # Get performance indicators
                arima_perf = "✓" if report.get('arima_available') else "✗"
                ridge_perf = "✓" if report.get('ridge_available') else "✗"
                lstm_perf = "✓" if report.get('lstm_available') else "✗"
                
                print(f"{ticker:<8} {models_count:<8} {arima_perf:<8} {ridge_perf:<8} {lstm_perf:<8}")
        
        if failed_tickers:
            print(f"\n✗ Failed training: {len(failed_tickers)} tickers")
            print(f"  Failed: {', '.join(failed_tickers)}")
        
        # Save overall summary
        summary_report = {
            'timestamp': datetime.now().isoformat(),
            'total_tickers': len(valid_tickers),
            'successful_tickers': successful_tickers,
            'failed_tickers': failed_tickers,
            'training_reports': training_reports,
            'configuration': {
                'prediction_horizon': config.prediction_horizon,
                'sequence_length': config.sequence_length,
                'max_features': config.max_features
            }
        }
        
        summary_path = Path(config.reports_dir) / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print("KEY IMPROVEMENTS IMPLEMENTED:")
        print("• Enhanced sentiment data processing with volatility-based proxy")
        print("• Auto-ARIMA parameter selection (if pmdarima available)")
        print("• Multiple regression models (BayesianRidge, ElasticNet, Ridge)")
        print("• Feature selection and correlation removal")
        print("• Dynamic ensemble weighting based on validation performance") 
        print("• Robust scaling and enhanced data validation")
        print("• Comprehensive evaluation and reporting")
        print(f"• Training summary saved: {summary_path}")
        print(f"{'='*60}")
        
        if successful_tickers:
            print(f"\n🎯 PRESENTATION READY!")
            print(f"   {len(successful_tickers)} models trained with improved methodology")
            print(f"   Focus on: Dynamic weighting, feature selection, enhanced validation")
            
            # Quick install note for missing libraries
            missing_libs = []
            if not PMDARIMA_AVAILABLE:
                missing_libs.append("pmdarima")
            
            if missing_libs:
                print(f"\n📦 Optional Enhancement:")
                print(f"   pip install {' '.join(missing_libs)}")
                print(f"   (for Auto-ARIMA parameter selection)")
        else:
            print(f"\n⚠️  WARNING: No models trained successfully")
            print(f"   Check logs and data availability")
    
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logging.error(error_msg)
        traceback.print_exc()
        print(f"\nERROR: {error_msg}")

if __name__ == "__main__":
    main()