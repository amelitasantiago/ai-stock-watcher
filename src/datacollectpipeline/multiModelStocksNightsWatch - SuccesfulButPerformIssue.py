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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available for ARIMA")

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

from dataclasses import dataclass
import structlog
from multiprocessing import Pool

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
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
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open('r') as f:
        return json.load(f)

@dataclass
class ModelConfig:
    """Configuration for the hybrid model"""
    sequence_length: int = 60
    prediction_horizon: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    arima_order: tuple = (2, 1, 2)
    arima_seasonal_order: tuple = (1, 1, 1, 5)
    lstm_hidden_size: int = 128
    transformer_num_heads: int = 8
    transformer_num_layers: int = 4
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50
    alpha_1: float = 1e-6  # Ridge priors
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    arima_weight: float = 0.3  # Ensemble weights
    bayesian_ridge_weight: float = 0.2
    lstm_transformer_weight: float = 0.5
    model_dir: str = "models"
    reports_dir: str = "reports"
    include_technical_indicators: bool = True  # FIXED: Add this for TA toggle in prepare_features
    include_sentiment: bool = True  # FIXED: Add this for sentiment toggle in prepare_features

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
    
    def load_all_stock_data(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Load stock data for multiple tickers efficiently."""
        if not self.stock_db_path.exists():
            logging.error(f"Stock database not found: {self.stock_db_path}")
            return {}
        
        conn = sqlite3.connect(self.stock_db_path)
        query = """
        SELECT ticker, date, open, high, low, close, adj_close, volume
        FROM stock_prices
        WHERE UPPER(ticker) IN ({})
        ORDER BY ticker, date ASC
        """
        placeholders = ','.join(['?' for _ in tickers])
        query = query.format(placeholders)
        
        try:
            df = pd.read_sql_query(query, conn, params=[t.upper() for t in tickers])
            conn.close()
            
            if df.empty:
                logging.warning("No stock data found for any tickers")
                return {}
            
            result = {}
            for ticker in tickers:
                ticker_df = df[df['ticker'].str.upper() == ticker.upper()]
                if ticker_df.empty:
                    logging.warning(f"No stock data found for ticker: {ticker}")
                    continue
                ticker_df = ticker_df.copy()
                ticker_df['date'] = pd.to_datetime(ticker_df['date'])
                ticker_df.set_index('date', inplace=True)
                ticker_df.drop(columns=['ticker'], inplace=True)
                result[ticker] = ticker_df
                logging.info(f"Loaded {len(ticker_df)} records for {ticker}")
            return result
        
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            conn.close()
            return {}
    
    def load_sentiment_data(self, ticker: str) -> pd.DataFrame:
        """Load sentiment data for a specific ticker."""
        if not self.sentiment_db_path.exists():
            logging.warning(f"Sentiment database not found: {self.sentiment_db_path}")
            return pd.DataFrame()
        
        try:
            conn = sqlite3.connect(self.sentiment_db_path)
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
            conn.close()
            
            if df.empty:
                logging.warning(f"No sentiment data found for ticker: {ticker}")
                return pd.DataFrame()
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            logging.info(f"Loaded sentiment data for {len(df)} days for {ticker}")
            return df
        
        except sqlite3.Error as e:
            logging.error(f"Sentiment database error for ticker {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        df = df.copy()
        
        # Basic indicators
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        df['high_low_ratio'] = df['high'] / df['low']
        df['volume_change'] = df['volume'].pct_change()
        
        # Moving averages and volatility
        windows = [5, 10, 20, 50, 200]
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False, min_periods=1).mean()
        
        df['volatility_10'] = df['price_change'].rolling(window=10, min_periods=1).std()
        df['volatility_30'] = df['price_change'].rolling(window=30, min_periods=1).std()
        df['price_position_sma20'] = df['close'] / df['sma_20']
        df['price_position_sma50'] = df['close'] / df['sma_50']
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # TA-Lib indicators
        if TALIB_AVAILABLE:
            try:
                df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
                macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
                df['macd'] = macd
                df['macd_signal'] = macd_signal
                df['macd_histogram'] = macd_hist
                upper, middle, lower = talib.BBANDS(df['close'].values)
                df['bb_upper'] = upper
                df['bb_middle'] = middle
                df['bb_lower'] = lower
                df['bb_width'] = (upper - lower) / middle
                df['bb_position'] = (df['close'] - lower) / (upper - lower)
                slowk, slowd = talib.STOCH(df['high'].values, df['low'].values, df['close'].values)
                df['stoch_k'] = slowk
                df['stoch_d'] = slowd
            except Exception as e:
                logging.warning(f"Error calculating some technical indicators: {e}")
        
        # Fill NaNs for technical indicators
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df
    
    def merge_with_sentiment(self, stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Merge stock data with sentiment data."""
        if sentiment_df.empty:
            logging.info("No sentiment data available; using dummy values")
            stock_df['avg_sentiment'] = 0.0
            stock_df['avg_magnitude'] = 0.0
            stock_df['article_count'] = 0
            return stock_df
        
        merged_df = stock_df.merge(sentiment_df, left_index=True, right_index=True, how='left')
        sentiment_columns = ['avg_sentiment', 'avg_magnitude', 'article_count']
        merged_df[sentiment_columns] = merged_df[sentiment_columns].fillna(method='ffill').fillna(0)
        logging.info(f"Merged sentiment data: {merged_df['avg_sentiment'].notna().sum()} days with sentiment")
        return merged_df
    
    def prepare_features(self, ticker: str, config: ModelConfig) -> pd.DataFrame:
        """Prepare all features for a ticker."""
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
        sentiment_df = self.load_sentiment_data(ticker)
        if config.include_sentiment:
            stock_df = self.merge_with_sentiment(stock_df, sentiment_df)
            logging.info(f"After sentiment merge: {stock_df.shape}")
        
        # Create target columns
        for horizon in range(1, config.prediction_horizon + 1):
            stock_df[f'target_{horizon}d'] = stock_df['close'].shift(-horizon)
        
        # Selective NaN dropping (only on critical columns)
        critical_columns = ['close'] + [f'target_{i}d' for i in range(1, config.prediction_horizon + 1)]
        stock_df = stock_df.dropna(subset=critical_columns)
        logging.info(f"After dropping NaNs in critical columns: {stock_df.shape}")
        
        # Fill remaining NaNs
        stock_df = stock_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        if stock_df.empty:
            logging.warning(f"Feature set for {ticker} is empty after processing")
        else:
            logging.info(f"Final feature set for {ticker}: {stock_df.shape}")
            logging.info(f"Feature columns: {list(stock_df.columns)}")
        return stock_df

class LSTMTransformer(nn.Module):
    """LSTM + Transformer hybrid for multi-horizon forecasting"""
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 4,
                 num_heads: int = 8, dropout: float = 0.2, prediction_horizon: int = 5):
        super(LSTMTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.fc = nn.Linear(hidden_size, prediction_horizon)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, feat)
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden)
        transformer_out = self.transformer_encoder(lstm_out)
        last_hidden = self.dropout(transformer_out[:, -1, :])  # (batch, hidden)
        out = self.fc(last_hidden)  # (batch, horizon)
        return out

class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series sequences with multi-horizon targets"""
    def __init__(self, X: torch.Tensor, y: torch.Tensor, sequence_length: int = 60):
        # FIXED: Accept sequence_length as 4th positional (self + X + y + seq_len)
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch required for TimeSeriesDataset")
        self.X = X  # (n_samples, n_features) or (n, seq, feat)—flatten if needed
        self.y = y  # (n_samples, horizon)
        self.sequence_length = sequence_length
        if len(self.X.shape) == 2:  # Reshape to (n, 1, feat) if flat
            self.X = self.X.unsqueeze(1)
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(f"X samples ({self.X.shape[0]}) != y samples ({self.y.shape[0]})")
        self.n_samples = self.X.shape[0] - sequence_length + 1  # Valid windows
        if self.n_samples <= 0:
            raise ValueError(f"Insufficient samples ({self.X.shape[0]}) for seq_len {sequence_length}")
        logging.debug(f"TimeSeriesDataset: {self.n_samples} windows from {self.X.shape[0]} samples, seq={sequence_length}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Sliding window: X[idx:idx+seq], y[idx+seq-1] (align to end for causal)
        x_seq = self.X[idx:idx + self.sequence_length].permute(1, 0, 2) if len(self.X.shape) == 3 else self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length - 1]  # Last in window for next-horizon
        return x_seq, y_target

class ARIMAModel:
    """ARIMA model wrapper with automatic parameter selection."""
    
    def __init__(self, order=(2, 1, 2)):
        self.order = order
        self.model = None
        self.model_fit = None
    
    def fit(self, y: pd.Series):
        """Fit ARIMA model."""
        try:
            adf_result = adfuller(y.dropna())
            if adf_result[1] > 0.05:
                logging.info(f"Series may not be stationary (p-value: {adf_result[1]:.4f})")
            
            self.model = ARIMA(y, order=self.order)
            self.model_fit = self.model.fit()
            logging.info(f"ARIMA{self.order} fitted successfully")
            logging.info(f"AIC: {self.model_fit.aic:.2f}")
        except Exception as e:
            logging.error(f"ARIMA fitting failed: {e}")
            self.model = None
            self.model_fit = None
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate predictions."""
        if self.model_fit is None:
            raise ValueError("Model not fitted")
        forecast = self.model_fit.forecast(steps=steps)
        return np.array(forecast)
    
    def get_model_summary(self):
        """Get model summary."""
        if self.model_fit is None:
            return "Model not fitted"
        return str(self.model_fit.summary())

class HybridStockPredictor:
    """Hybrid predictor combining ARIMA, LSTM-Transformer, and Bayesian Ridge"""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_columns = []
        self.target_columns = ['target_1d', 'target_2d', 'target_3d', 'target_4d', 'target_5d']
        self.arima_model = None
        self.lstm_model = None
        self.bayesian_models = {}  # Dict for per-horizon Ridge models
        self.lstm_scalers = {}  # LSTM-specific scalers
        self.scalers = {}  # General scalers
        self.ensemble_weights = {}
        logging.debug(f"HybridStockPredictor initialized")
    
    def train(self, ticker: str, data_processor: DataProcessor) -> bool:
        """Train hybrid model for a ticker."""
        try:
            logging.info(f"Starting hybrid model training for {ticker}")
            
            # FIXED: Use correct method calls from DataProcessor
            df = data_processor.prepare_features(ticker, self.config)
            if df.empty:
                logging.warning(f"No data for {ticker}")
                return False
            
            logging.info(f"Loaded {len(df)} records for {ticker}")
            logging.info(f"Initial stock data shape for {ticker}: {df.shape}")
            
            # Prepare features and targets
            target_columns = [col for col in df.columns if col.startswith('target_')]
            feature_columns = [col for col in df.columns if not col.startswith('target_') and col != 'date']
            
            # Filter numeric columns only
            numeric_columns = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = numeric_columns
            
            logging.info(f"Final feature set for {ticker}: ({len(df)}, {len(self.feature_columns)})")
            logging.info(f"Feature columns: {self.feature_columns}")
            logging.info(f"Using {len(self.feature_columns)} features: {self.feature_columns[:10]}...")
            
            if len(self.feature_columns) == 0:
                logging.error(f"No numeric features available for {ticker}")
                return False
            
            X_features = df[self.feature_columns].values
            y_targets = df[target_columns].values
            
            if X_features.shape[0] == 0 or y_targets.shape[0] == 0:
                logging.error(f"Empty feature or target arrays for {ticker}")
                return False
            
            # ARIMA Training
            if STATSMODELS_AVAILABLE:
                try:
                    price_series = df['close'].dropna()
                    if len(price_series) > 0:
                        # Check stationarity
                        adf_result = adfuller(price_series)
                        if adf_result[1] > 0.05:
                            logging.info(f"Series may not be stationary (p-value: {adf_result[1]:.4f})")
                        
                        # Fit ARIMA
                        self.arima_model = ARIMA(price_series, order=self.config.arima_order)
                        arima_fitted = self.arima_model.fit()
                        logging.info(f"ARIMA{self.config.arima_order} fitted successfully")
                        logging.info(f"AIC: {arima_fitted.aic:.2f}")
                        logging.info("ARIMA model trained successfully")
                        self.arima_model = arima_fitted  # Store the fitted model
                    else:
                        logging.warning("Empty price series for ARIMA")
                        self.arima_model = None
                except Exception as e:
                    logging.error(f"ARIMA training failed: {e}")
                    self.arima_model = None
            else:
                logging.warning("ARIMA skipped - statsmodels not available")
                self.arima_model = None
            
            # Bayesian Ridge Training - FIXED: Handle return values correctly
            try:
                ridge_results = self.train_bayesian_ridge(X_features, y_targets)
                if ridge_results and len(ridge_results) == 2:
                    self.bayesian_models, r2_scores = ridge_results
                    if r2_scores:
                        avg_r2 = np.mean(r2_scores)
                        logging.info(f"Bayesian Ridge trained (avg R²: {avg_r2:.3f})")
                    else:
                        logging.warning("Bayesian Ridge trained but no R² scores available")
                else:
                    logging.warning("Bayesian Ridge training returned unexpected results")
                    self.bayesian_models = {}
            except Exception as e:
                logging.error(f"Bayesian Ridge training failed: {e}")
                traceback.print_exc()
                self.bayesian_models = {}
            
            # LSTM-Transformer Training - FIXED: Handle return values correctly
            if TORCH_AVAILABLE:
                try:
                    # Create sequences for LSTM
                    X_seq, y_seq = self.create_sequences(X_features, y_targets, self.config.sequence_length)
                    if X_seq.shape[0] > 0:
                        lstm_results = self.train_lstm_transformer(X_seq, y_seq)
                        if lstm_results and len(lstm_results) >= 3:
                            self.lstm_model, optimizer, metrics = lstm_results
                            logging.info(f"LSTM trained (loss: {metrics.get('loss', 'N/A'):.4f})")
                        else:
                            logging.warning("LSTM training returned unexpected results")
                            self.lstm_model = None
                    else:
                        logging.warning("Insufficient data for LSTM sequences")
                        self.lstm_model = None
                except Exception as e:
                    logging.error(f"LSTM training failed: {e}")
                    traceback.print_exc()
                    self.lstm_model = None
            else:
                logging.warning("LSTM skipped - PyTorch not available")
                self.lstm_model = None
            
            # Configure ensemble weights
            active_models = 0
            if self.arima_model is not None:
                active_models += 1
            if self.bayesian_models:
                active_models += 1
            if self.lstm_model is not None:
                active_models += 1
            
            if active_models == 0:
                logging.error(f"No models were successfully trained for {ticker}")
                return False
            
            # Normalize weights based on active models
            total_weight = 0
            if self.arima_model is not None:
                total_weight += self.config.arima_weight
            if self.bayesian_models:
                total_weight += self.config.bayesian_ridge_weight
            if self.lstm_model is not None:
                total_weight += self.config.lstm_transformer_weight
            
            if total_weight > 0:
                self.ensemble_weights = {
                    'arima': self.config.arima_weight / total_weight if self.arima_model else 0,
                    'ridge': self.config.bayesian_ridge_weight / total_weight if self.bayesian_models else 0,
                    'lstm': self.config.lstm_transformer_weight / total_weight if self.lstm_model else 0
                }
            else:
                # Equal weights fallback
                weight_per_model = 1.0 / active_models
                self.ensemble_weights = {
                    'arima': weight_per_model if self.arima_model else 0,
                    'ridge': weight_per_model if self.bayesian_models else 0,
                    'lstm': weight_per_model if self.lstm_model else 0
                }
            
            logging.info(f"Hybrid ensemble configured: {self.ensemble_weights}")
            logging.info(f"Successfully trained {active_models} out of 3 models for {ticker}")
            
            return True
            
        except Exception as e:
            logging.error(f"Training failed for {ticker}: {e}")
            traceback.print_exc()
            return False
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build sequences for LSTM training"""
        if len(X) < sequence_length:
            logging.warning(f"Insufficient data for sequences: {len(X)} < {sequence_length}")
            return np.array([]).reshape(0, sequence_length, X.shape[1]), np.array([]).reshape(0, y.shape[1])
        
        X_seqs, y_vecs = [], []
        for i in range(sequence_length - 1, len(X)):
            X_seqs.append(X[i - sequence_length + 1 : i + 1, :])
            y_vecs.append(y[i, :])
        
        X_seqs = np.array(X_seqs)
        y_vecs = np.array(y_vecs)
        
        logging.info(f"Created {len(X_seqs)} sequences: X{X_seqs.shape}, y{y_vecs.shape}")
        return X_seqs, y_vecs
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for training."""
        target_columns = [col for col in df.columns if col.startswith('target_')]
        feature_columns = [col for col in df.columns if not col.startswith('target_')]
        numeric_columns = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        self.feature_columns = numeric_columns
        logging.info(f"Using {len(self.feature_columns)} features: {self.feature_columns[:10]}...")
        X = df[self.feature_columns].values
        y = df[target_columns].values
        self.scalers['features'] = StandardScaler()
        X_scaled = self.scalers['features'].fit_transform(X)
        self.scalers['targets'] = StandardScaler()
        y_scaled = self.scalers['targets'].fit_transform(y)
        return X_scaled, y_scaled, target_columns
    
    def train_arima(self, price_series: pd.Series):
        """Train ARIMA model."""
        if not STATSMODELS_AVAILABLE:
            logging.warning("ARIMA training skipped - statsmodels not available")
            return
        try:
            self.arima_model = ARIMAModel(order=self.config.arima_order)
            self.arima_model.fit(price_series)
            logging.info("ARIMA model trained successfully")
        except Exception as e:
            logging.error(f"ARIMA training failed: {e}")
            self.arima_model = None
    
    def train_lstm_transformer(self, X: np.ndarray, y: np.ndarray) -> Optional[Tuple[nn.Module, optim.Optimizer, Dict]]:
        """Train LSTM-Transformer with proper validation"""
        if not TORCH_AVAILABLE:
            logging.warning("PyTorch unavailable; skipping LSTM-Transformer")
            return None

        try:
            if len(X.shape) != 3 or X.shape[0] == 0:
                logging.error(f"Invalid LSTM input shape: {X.shape}")
                return None
            
            logging.info(f"LSTM training: X={X.shape}, y={y.shape}")
            
            # Scale data
            n_samples, seq_len, n_features = X.shape
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = scaler_X.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)
            y_scaled = scaler_y.fit_transform(y)
            
            # Create dataset with pre-sequenced data
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y_scaled)
            
            # Simple dataset without additional windowing
            dataset = list(zip(X_tensor, y_tensor))
            dataloader = DataLoader(
                dataset, 
                batch_size=min(self.config.batch_size, len(dataset)), 
                shuffle=False
            )
            
            # Initialize model
            model = LSTMTransformer(
                input_size=n_features,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.transformer_num_layers,
                num_heads=self.config.transformer_num_heads,
                dropout=self.config.dropout_rate,
                prediction_horizon=self.config.prediction_horizon
            )
            
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()
            
            # Training
            model.train()
            final_loss = 0
            
            for epoch in range(self.config.epochs):
                total_loss = 0
                batch_count = 0
                
                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                if batch_count > 0:
                    final_loss = total_loss / batch_count
                    if epoch % 10 == 0 or epoch == self.config.epochs - 1:
                        logging.info(f"LSTM Epoch {epoch+1}: Loss={final_loss:.4f}")
            
            self.lstm_scalers = {'X': scaler_X, 'y': scaler_y}
            return model, optimizer, {'loss': final_loss}

        except Exception as e:
            logging.error(f"LSTM training error: {e}")
            traceback.print_exc()
            return None
    
    def train_bayesian_ridge(self, X: np.ndarray, y_multi: np.ndarray) -> Tuple[Dict[int, BayesianRidge], List[float]]:
        """Train per-horizon Ridge models with robust error handling."""
        bayesian_models = {}
        r2_scores = []
        
        if X.shape[0] == 0 or y_multi.shape[0] == 0:
            logging.warning("Empty input data for Bayesian Ridge")
            return bayesian_models, r2_scores
        
        # Pre-scale features once for all horizons
        feature_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(X)
        
        for h in range(1, self.config.prediction_horizon + 1):
            try:
                if h > y_multi.shape[1]:
                    logging.warning(f"Horizon {h} exceeds target columns {y_multi.shape[1]}")
                    continue
                
                y_h = y_multi[:, h-1]  # 0-based indexing
                
                # Skip if insufficient data
                min_samples = max(10, int(X.shape[1] * 1.5))  # At least 10 or 1.5x features
                if X.shape[0] < min_samples:
                    logging.warning(f"Insufficient samples ({X.shape[0]}) for horizon {h}")
                    continue
                
                # Train/test split on scaled data
                test_size = max(0.1, min(self.config.test_size, 0.3))  # Between 10-30%
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_h, test_size=test_size, shuffle=False, random_state=42
                )
                
                if len(X_train) == 0 or len(X_test) == 0:
                    logging.warning(f"Empty train/test split for horizon {h}")
                    continue
                
                # Train model with sklearn-compatible parameters only
                model_h = BayesianRidge(
                    alpha_1=self.config.alpha_1, 
                    alpha_2=self.config.alpha_2,
                    lambda_1=self.config.lambda_1, 
                    lambda_2=self.config.lambda_2,
                    fit_intercept=True,
                    copy_X=True,
                    compute_score=False
                )
                
                model_h.fit(X_train, y_train)
                y_pred = model_h.predict(X_test)
                r2_h = r2_score(y_test, y_pred)
                
                bayesian_models[h] = model_h
                r2_scores.append(r2_h)
                
                logging.info(f"Ridge horizon {h}: R²={r2_h:.3f}, train={len(X_train)}, test={len(X_test)}")
                
            except Exception as e:
                logging.error(f"Bayesian Ridge training failed for horizon {h}: {e}")
                continue
        
        # Store the feature scaler for later use in predictions
        if bayesian_models:
            self.scalers['ridge_features'] = feature_scaler
            avg_r2 = np.mean(r2_scores)
            logging.info(f"Bayesian Ridge: {len(bayesian_models)} horizons trained, avg R²: {avg_r2:.3f}")
        else:
            logging.warning("No Bayesian Ridge models successfully trained")
        
        return bayesian_models, r2_scores
    
    def train(self, ticker: str, data_processor: DataProcessor) -> bool:
        """Main training method - single entry point"""
        try:
            logging.info(f"Starting hybrid model training for {ticker}")
            
            # Load and prepare features
            df = data_processor.prepare_features(ticker, self.config)
            if df.empty:
                logging.warning(f"No data available for {ticker}")
                return False
            
            logging.info(f"Loaded {len(df)} records for {ticker}")
            logging.info(f"Initial stock data shape for {ticker}: {df.shape}")
            
            # Identify feature and target columns
            target_columns = [col for col in df.columns if col.startswith('target_')]
            feature_columns = [col for col in df.columns 
                             if not col.startswith('target_') and col != 'date' and col != 'ticker']
            
            # Filter to numeric columns only
            numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = numeric_features
            
            logging.info(f"Final feature set for {ticker}: ({len(df)}, {len(self.feature_columns)})")
            logging.info(f"Feature columns: {self.feature_columns}")
            logging.info(f"Using {len(self.feature_columns)} features: {self.feature_columns[:10]}...")
            
            if len(self.feature_columns) == 0:
                logging.error(f"No numeric features available for {ticker}")
                return False
            
            X_features = df[self.feature_columns].values
            y_targets = df[target_columns].values
            
            if X_features.shape[0] == 0 or y_targets.shape[0] == 0:
                logging.error(f"Empty arrays: X{X_features.shape}, y{y_targets.shape}")
                return False
            
            # Train ARIMA
            if STATSMODELS_AVAILABLE:
                try:
                    price_series = df['close'].dropna()
                    if len(price_series) > 10:
                        adf_result = adfuller(price_series)
                        if adf_result[1] > 0.05:
                            logging.info(f"Series may not be stationary (p-value: {adf_result[1]:.4f})")
                        
                        arima_model = ARIMA(price_series, order=self.config.arima_order)
                        self.arima_model = arima_model.fit()
                        logging.info(f"ARIMA{self.config.arima_order} fitted successfully")
                        logging.info(f"AIC: {self.arima_model.aic:.2f}")
                        logging.info("ARIMA model trained successfully")
                    else:
                        logging.warning("Insufficient data for ARIMA")
                        self.arima_model = None
                except Exception as e:
                    logging.error(f"ARIMA training failed: {e}")
                    self.arima_model = None
            
            # Train Bayesian Ridge
            try:
                self.bayesian_models, r2_scores = self.train_bayesian_ridge(X_features, y_targets)
                if self.bayesian_models:
                    avg_r2 = np.mean(r2_scores) if r2_scores else 0
                    logging.info(f"Bayesian Ridge trained (avg R²: {avg_r2:.3f})")
            except Exception as e:
                logging.error(f"Bayesian Ridge failed: {e}")
                self.bayesian_models = {}
            
            # Train LSTM-Transformer
            if TORCH_AVAILABLE and len(X_features) > self.config.sequence_length:
                try:
                    X_seq, y_seq = self.create_sequences(X_features, y_targets, self.config.sequence_length)
                    if X_seq.shape[0] > 0:
                        lstm_results = self.train_lstm_transformer(X_seq, y_seq)
                        if lstm_results:
                            self.lstm_model = lstm_results[0]
                            logging.info("LSTM-Transformer trained successfully")
                        else:
                            self.lstm_model = None
                    else:
                        logging.warning("No sequences created for LSTM")
                        self.lstm_model = None
                except Exception as e:
                    logging.error(f"LSTM training failed: {e}")
                    self.lstm_model = None
            
            # Configure ensemble
            active_models = sum([
                1 if self.arima_model else 0,
                1 if self.bayesian_models else 0, 
                1 if self.lstm_model else 0
            ])
            
            if active_models == 0:
                logging.error(f"No models successfully trained for {ticker}")
                return False
            
            # Set ensemble weights
            total_weight = (
                (self.config.arima_weight if self.arima_model else 0) +
                (self.config.bayesian_ridge_weight if self.bayesian_models else 0) +
                (self.config.lstm_transformer_weight if self.lstm_model else 0)
            )
            
            if total_weight > 0:
                self.ensemble_weights = {
                    'arima': self.config.arima_weight / total_weight if self.arima_model else 0,
                    'ridge': self.config.bayesian_ridge_weight / total_weight if self.bayesian_models else 0,
                    'lstm': self.config.lstm_transformer_weight / total_weight if self.lstm_model else 0
                }
            else:
                equal_weight = 1.0 / active_models
                self.ensemble_weights = {
                    'arima': equal_weight if self.arima_model else 0,
                    'ridge': equal_weight if self.bayesian_models else 0,
                    'lstm': equal_weight if self.lstm_model else 0
                }
            
            logging.info(f"Hybrid ensemble configured: {self.ensemble_weights}")
            logging.info(f"Successfully trained {active_models} out of 3 models for {ticker}")
            
            return True
            
        except Exception as e:
            logging.error(f"Training failed for {ticker}: {e}")
            traceback.print_exc()
            return False
    
    def predict(self, recent_data: np.ndarray, price_series: pd.Series = None) -> Dict[str, np.ndarray]:
        """Generate predictions from all models."""
        predictions = {}
        
        # ARIMA predictions
        if self.arima_model is not None:
            try:
                arima_pred = self.arima_model.forecast(self.config.prediction_horizon)
                predictions['arima'] = np.array(arima_pred)
            except Exception as e:
                logging.error(f"ARIMA prediction failed: {e}")
                predictions['arima'] = np.zeros(self.config.prediction_horizon)
        else:
            predictions['arima'] = np.zeros(self.config.prediction_horizon)
        
        # LSTM-Transformer predictions
        if self.lstm_model is not None and TORCH_AVAILABLE:
            try:
                self.lstm_model.eval()
                with torch.no_grad():
                    # Use the last sequence from recent_data
                    if len(recent_data.shape) == 3:
                        input_tensor = torch.FloatTensor(recent_data[-1:])  # Last sequence
                    else:
                        # Create sequence from recent data
                        seq_len = min(self.config.sequence_length, recent_data.shape[0])
                        input_seq = recent_data[-seq_len:].reshape(1, seq_len, -1)
                        input_tensor = torch.FloatTensor(input_seq)
                    
                    lstm_pred = self.lstm_model(input_tensor)
                    predictions['lstm'] = lstm_pred.numpy()[0]
            except Exception as e:
                logging.error(f"LSTM prediction failed: {e}")
                predictions['lstm'] = np.zeros(self.config.prediction_horizon)
        else:
            predictions['lstm'] = np.zeros(self.config.prediction_horizon)
        
        # Bayesian Ridge predictions
        if self.bayesian_models:
            try:
                br_predictions = []
                recent_flat = recent_data[-1:] if len(recent_data.shape) > 2 else recent_data[-1:].reshape(1, -1)
                
                # Apply the same scaling used during training
                if 'ridge_features' in self.scalers:
                    recent_scaled = self.scalers['ridge_features'].transform(recent_flat)
                else:
                    recent_scaled = recent_flat
                    logging.warning("No Ridge feature scaler found, using unscaled data")
                
                for h in range(1, self.config.prediction_horizon + 1):
                    if h in self.bayesian_models:
                        pred = self.bayesian_models[h].predict(recent_scaled)[0]
                        br_predictions.append(pred)
                    else:
                        br_predictions.append(0.0)
                
                predictions['ridge'] = np.array(br_predictions)
            except Exception as e:
                logging.error(f"Bayesian Ridge prediction failed: {e}")
                predictions['ridge'] = np.zeros(self.config.prediction_horizon)
        else:
            predictions['ridge'] = np.zeros(self.config.prediction_horizon)
        
        # Ensemble prediction
        ensemble_pred = (
            predictions['arima'] * self.ensemble_weights.get('arima', 0) +
            predictions['lstm'] * self.ensemble_weights.get('lstm', 0) +
            predictions['ridge'] * self.ensemble_weights.get('ridge', 0)
        )
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        predictions = self.predict(X, None)['ensemble']
        metrics = {}
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        metrics = {'mse': mse, 'mae': mae, 'r2': r2}
        logging.info(f"Ensemble - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained model."""
        try:
            # Prepare model data (exclude PyTorch model)
            model_data = {
                'config': self.config,
                'scalers': self.scalers,
                'lstm_scalers': self.lstm_scalers,
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'arima_model': self.arima_model,
                'bayesian_models': self.bayesian_models,
                'ensemble_weights': self.ensemble_weights
            }
            
            # Save main model data
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save PyTorch model separately if it exists
            if self.lstm_model is not None and TORCH_AVAILABLE:
                torch_path = f"{filepath}_lstm_transformer.pth"
                torch.save(self.lstm_model.state_dict(), torch_path)
                logging.info(f"LSTM model saved to {torch_path}")
            
            logging.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logging.error(f"Model save failed: {e}")
            raise
    
    def load_model(self, filepath: str):
        """Load trained model."""
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
            
            # Load PyTorch model if it exists
            pytorch_path = f"{filepath}_lstm_transformer.pth"
            if Path(pytorch_path).exists() and TORCH_AVAILABLE:
                try:
                    input_size = len(self.feature_columns)
                    self.lstm_model = LSTMTransformer(
                        input_size=input_size,
                        hidden_size=self.config.lstm_hidden_size,
                        num_heads=self.config.transformer_num_heads,
                        num_layers=self.config.transformer_num_layers,
                        dropout=self.config.dropout_rate,
                        prediction_horizon=self.config.prediction_horizon
                    )
                    self.lstm_model.load_state_dict(torch.load(pytorch_path))
                    logging.info(f"LSTM model loaded from {pytorch_path}")
                except Exception as e:
                    logging.error(f"LSTM model load failed: {e}")
                    self.lstm_model = None
            else:
                self.lstm_model = None
            
            logging.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logging.error(f"Model load failed: {e}")
            raise

def populate_stock_data_from_csv(tickers: List[str], db_path: str, csv_dir: str) -> bool:
    """Populate stock database from CSV files."""
    try:
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
        
        for ticker in tickers:
            csv_path = Path(csv_dir) / f"{ticker}_data.csv"
            if not csv_path.exists():
                logging.warning(f"CSV file not found for {ticker}: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                logging.warning(f"Empty CSV file for {ticker}: {csv_path}")
                continue
            expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            if not all(col in df.columns for col in expected_columns):
                logging.warning(f"Invalid schema in {csv_path}. Expected: {expected_columns}")
                continue
            df['ticker'] = ticker
            df = df[['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            df.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            df.to_sql('stock_prices', conn, if_exists='append', index=False)
            logging.info(f"Inserted {len(df)} records from {csv_path} for {ticker}")
        
        conn.commit()
        conn.close()
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stock_prices")
        count = cursor.fetchone()[0]
        conn.close()
        if count == 0:
            logging.error("No data inserted into stock_prices table")
            return False
        logging.info(f"Populated database with {count} total records from CSVs")
        return True
    except Exception as e:
        logging.error(f"Failed to populate stock data from CSVs: {e}")
        return False

def populate_stock_data(tickers: List[str], db_path: str) -> bool:
    """Populate stock database with data from yfinance."""
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
        
        for ticker in tickers:
            logging.info(f"Fetching data for {ticker}")
            df = yf.download(ticker, start="2020-01-01", end="2025-09-18", progress=False)
            if df.empty:
                logging.warning(f"No data retrieved for {ticker}")
                continue
            df = df.reset_index()
            df['ticker'] = ticker
            df = df[['ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
            df.columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            df.to_sql('stock_prices', conn, if_exists='append', index=False)
            logging.info(f"Inserted {len(df)} records for {ticker}")
        
        conn.commit()
        conn.close()
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stock_prices")
        count = cursor.fetchone()[0]
        conn.close()
        if count == 0:
            logging.error("No data inserted into stock_prices table")
            return False
        logging.info(f"Populated database with {count} total records")
        return True
    except Exception as e:
        logging.error(f"Failed to populate stock data: {e}")
        return False

def train_ticker(args):
    """Train model for a single ticker with robust error handling."""
    try:
        # Unpack arguments safely
        if not isinstance(args, (tuple, list)) or len(args) != 4:
            logging.error(f"Invalid args format: expected 4-tuple, got {type(args)} with length {len(args) if hasattr(args, '__len__') else 'unknown'}")
            return "Failed: Invalid arguments"
        
        ticker, config, stock_db_path, sentiment_db_path = args
        
        logging.info(f"Starting training for {ticker}")
        
        # Initialize components
        try:
            data_processor = DataProcessor()
            predictor = HybridStockPredictor(config)
        except Exception as e:
            logging.error(f"Failed to initialize components for {ticker}: {e}")
            return f"Failed training for {ticker}: Initialization error"
        
        # Train the model
        try:
            success = predictor.train(ticker, data_processor)
        except Exception as e:
            logging.error(f"Training execution failed for {ticker}: {e}")
            traceback.print_exc()
            return f"Failed training for {ticker}: {str(e)}"
        
        # Save model if training succeeded and we have trained components
        if success:
            try:
                # Check if we have any trained models
                has_models = any([
                    predictor.arima_model is not None,
                    bool(predictor.bayesian_models),
                    predictor.lstm_model is not None
                ])
                
                if has_models and predictor.feature_columns:
                    # Ensure model directory exists
                    model_dir = Path(config.model_dir)
                    model_dir.mkdir(exist_ok=True, parents=True)
                    
                    # Save the model
                    model_path = model_dir / f"hybrid_model_{ticker}.pkl"
                    predictor.save_model(str(model_path))
                    
                    logging.info(f"Model saved for {ticker} at {model_path}")
                    return f"Completed training for {ticker}"
                else:
                    logging.warning(f"No models trained successfully for {ticker}")
                    return f"Training completed for {ticker} but no models were successfully trained"
            except Exception as e:
                logging.error(f"Model saving failed for {ticker}: {e}")
                return f"Training completed for {ticker} but model saving failed: {str(e)}"
        else:
            logging.info(f"Training reported failure for {ticker}")
            return f"Failed training for {ticker}: Training process failed"
    
    except Exception as e:
        logging.error(f"Unexpected error in train_ticker for {ticker if 'ticker' in locals() else 'unknown'}: {e}")
        traceback.print_exc()
        return f"Failed training: Unexpected error - {str(e)}"

def main():
    """Main training script with improved error handling."""
    try:
        # Load configuration
        try:
            config_data = load_config()
            config = ModelConfig()
        except Exception as e:
            logging.error(f"Configuration loading failed: {e}")
            print("Training failed: Could not load configuration.")
            return
        
        # Setup paths
        stock_db_path = Path(config_data['stock_db_path']).resolve()
        sentiment_db_path = Path(config_data['sentiment_db_path']).resolve()
        csv_dir = Path(config_data['csv_dir']).resolve()
        tickers = config_data['tickers']
        
        logging.info(f"Starting main with {len(tickers)} tickers: {tickers}")
        
        # Initialize data processor and check database
        try:
            data_processor = DataProcessor()
            available_tickers = data_processor.inspect_database()
        except Exception as e:
            logging.error(f"Database inspection failed: {e}")
            print("Training failed: Could not access database.")
            return
        
        # Handle missing tickers
        missing_tickers = [t for t in tickers if t.upper() not in [at.upper() for at in available_tickers]]
        if missing_tickers:
            logging.info(f"Missing data for tickers: {missing_tickers}")
            
            # Try CSV population first
            if csv_dir.exists():
                logging.info("Attempting to populate from CSVs...")
                if not populate_stock_data_from_csv(missing_tickers, str(stock_db_path), str(csv_dir)):
                    logging.info("CSV population failed, trying yfinance...")
                    if not populate_stock_data(missing_tickers, str(stock_db_path)):
                        logging.error("Both CSV and yfinance population failed")
                        print("Training failed: Could not populate missing stock data.")
                        return
            else:
                logging.info("CSV directory not found, trying yfinance...")
                if not populate_stock_data(missing_tickers, str(stock_db_path)):
                    logging.error("yfinance population failed")
                    print("Training failed: Could not populate stock data.")
                    return
        
        # Re-check available tickers
        try:
            available_tickers = data_processor.inspect_database()
            if not available_tickers:
                logging.error("No tickers available after population attempts")
                print("Training failed: No data available in database.")
                return
        except Exception as e:
            logging.error(f"Database re-inspection failed: {e}")
            print("Training failed: Database access error.")
            return
        
        # Filter tickers to only those with available data
        valid_tickers = [t for t in tickers if t.upper() in [at.upper() for at in available_tickers]]
        if not valid_tickers:
            logging.error("No valid tickers found with available data")
            print("Training failed: No tickers have available data.")
            return
        
        logging.info(f"Training {len(valid_tickers)} valid tickers: {valid_tickers}")
        
        # Train models
        successful_tickers = []
        failed_tickers = []
        
        try:
            # Prepare arguments for multiprocessing
            args_list = [
                (ticker, config, str(stock_db_path), str(sentiment_db_path)) 
                for ticker in valid_tickers
            ]
            
            logging.info(f"Starting parallel training for {len(args_list)} tickers")
            
            # Use multiprocessing to train models
            with Pool() as pool:
                results = pool.map(train_ticker, args_list)
            
            # Process results
            for result, ticker in zip(results, valid_tickers):
                logging.info(f"Training result for {ticker}: {result}")
                
                if "Completed" in result:
                    successful_tickers.append(ticker)
                else:
                    failed_tickers.append(ticker)
        
        except Exception as e:
            logging.error(f"Multiprocessing training failed: {e}")
            traceback.print_exc()
            print(f"Training failed: Multiprocessing error - {str(e)}")
            return
        
        # Report results
        if successful_tickers:
            success_msg = f"Hybrid model training completed successfully for {len(successful_tickers)} tickers: {successful_tickers}"
            model_dir_msg = f"Models saved in '{config_data['model_dir']}' directory"
            
            print(success_msg)
            print(model_dir_msg)
            logging.info(success_msg)
            logging.info(model_dir_msg)
            
            if failed_tickers:
                fail_msg = f"Failed training for {len(failed_tickers)} tickers: {failed_tickers}"
                print(fail_msg)
                logging.warning(fail_msg)
        else:
            fail_msg = "Training failed: No models were trained successfully."
            print(fail_msg)
            logging.warning("No models were trained successfully.")
    
    except Exception as e:
        error_msg = f"Training failed with unexpected error: {str(e)}"
        logging.error(error_msg)
        traceback.print_exc()
        print(error_msg)
        raise

if __name__ == "__main__":
    main()