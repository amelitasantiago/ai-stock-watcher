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
        self.lstm_scalers = {}  # LSTM-specific, if separate
        self.scalers = {}  # General scalers (ARIMA/Ridge/LSTM-X/Y) – FIXED: Initialize here
        logging.debug(f"HybridStockPredictor initialized with config={config.__class__.__name__}, scalers={type(self.scalers)}")
    
    def train(self, ticker: str, data_processor: DataProcessor) -> bool:
        """Train hybrid model for a ticker."""
        try:
            logging.info(f"Starting hybrid model training for {ticker}")
            # Feature prep (log: 1254 rows, 34 cols)
            df = data_processor.load_stock_data(ticker)
            if df.empty:
                logging.warning(f"No data for {ticker}")
                return False
            
            df = data_processor.calculate_technical_indicators(df)
            sentiment_df = data_processor.load_sentiment_data(ticker)
            df = data_processor.merge_with_sentiment(df, sentiment_df)

            df = data_processor.add_technical_indicators(df, self.config)  # TA to 26 cols
            df = data_processor.merge_sentiment_data(df, ticker)  # Sentiment to 29 cols
            df = df.dropna(subset=['close', 'target_1d'])  # To 1249
            feature_list = [col for col in df.columns if col not in ['date', 'ticker'] + self.target_columns]
            X_features = df[feature_list].values
            y_targets = df[self.target_columns].values
            logging.info(f"Final feature set for {ticker}: {X_features.shape}")
            logging.info(f"Feature columns: {feature_list}")
            
            # ARIMA (working, log: AIC logged)
            if STATSMODELS_AVAILABLE:
                self.scalers['arima'] = StandardScaler()
                series = self.scalers['arima'].fit_transform(df[['close']].values).flatten()
                adf_result = adfuller(series)
                logging.info(f"Stationarity p-value: {adf_result[1]:.4f}")
                self.arima_model = ARIMA(series, order=self.config.arima_order).fit()
                logging.info(f"ARIMA{self.config.arima_order} fitted, AIC: {self.arima_model.aic:.2f}")
            else:
                logging.warning("ARIMA skipped")
            
            # Bayesian Ridge (FIXED: Isolated try, safe [0]/[1])
            try:
                ridge_result = self.train_bayesian_ridge(X_features, y_targets)
                if len(ridge_result) < 2:
                    logging.warning("Ridge short return; skipping")
                    #self.bayesian_models = {}
                else:
                    self.bayesian_models, r2_scores = ridge_result[0], ridge_result[1]
                    avg_r2 = np.mean(r2_scores)
                    logging.info(f"Bayesian Ridge trained (avg R²: {avg_r2:.3f})")

            except IndexError as e:
                logging.error(f"Ridge unpack IndexError: {e}")
                traceback.print_exc()
                self.bayesian_models = {}
            except Exception as e:
                logging.error(f"Ridge failed: {e}")
                traceback.print_exc()
                self.bayesian_models = {}
            
            # LSTM-Transformer (FIXED: Isolated try, safe unpack)
            if TORCH_AVAILABLE:
                try:
                    lstm_result = self.train_lstm_transformer(X_features, y_targets)
                    if lstm_result and len(lstm_result) >= 3:
                        self.lstm_model, optimizer, metrics = lstm_result
                        logging.info(f"LSTM trained (loss: {metrics.get('loss', 'N/A'):.4f})")
                    else:
                        logging.warning(f"LSTM return short: len={len(lstm_result) if lstm_result else 0}")
                        self.lstm_model = None
                except IndexError as e:
                    logging.error(f"LSTM IndexError: {e}")
                    traceback.print_exc()
                    self.lstm_model = None
                except Exception as e:
                    logging.error(f"LSTM failed: {e}")
                    traceback.print_exc()
                    self.lstm_model = None
            else:
                logging.warning("LSTM skipped (Torch unavailable)")
                self.lstm_model = None
            
            # Ensemble (weights fallback if components missing)
            total_weight = self.config.arima_weight + self.config.bayesian_ridge_weight + (self.config.lstm_transformer_weight if self.lstm_model else 0)
            self.ensemble_weights = {
                'arima': self.config.arima_weight / total_weight if total_weight > 0 else 0,
                'ridge': self.config.bayesian_ridge_weight / total_weight if total_weight > 0 else 0,
                'lstm': self.config.lstm_transformer_weight / total_weight if self.lstm_model and total_weight > 0 else 0
            }
            logging.info(f"Hybrid ensemble configured: {self.ensemble_weights}")
            
            self.feature_columns = feature_list
            return True
        
        except Exception as e:
            logging.error(f"Training failed for {ticker}: {e}")
            traceback.print_exc()
            return False
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (X_seq, y_vec) pairs where:
        - X_seq: sequence_length × n_features (from X)
        - y_vec: 1 × prediction_horizon       (from the *same row* as the sequence end, in y)
        """
        X_seqs, y_vecs = [], []
        # make sequences end at row i; sequence covers [i-seq_len+1, i]
        for i in range(sequence_length - 1, len(X)):
            X_seqs.append(X[i - sequence_length + 1 : i + 1, :])
            y_vecs.append(y[i, :])  # vector (prediction_horizon,)
        return np.array(X_seqs), np.array(y_vecs)
    
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
        """Train LSTM-Transformer with shape safeguards"""
        if not TORCH_AVAILABLE:
            logging.warning("PyTorch unavailable; skipping LSTM-Transformer")
            return None
    
        try:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            y_scaled = scaler_y.fit_transform(y)
            
            dataset = TimeSeriesDataset(torch.FloatTensor(X_scaled), torch.FloatTensor(y_scaled), self.config.sequence_length)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=True)
            
            model = LSTMTransformer(
                input_size=X.shape[2],
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.transformer_num_layers,
                num_heads=self.config.transformer_num_heads,
                dropout=self.config.dropout_rate,
                prediction_horizon=self.config.prediction_horizon
            )
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(self.config.epochs):
                total_loss = 0
                for X_batch, y_batch in dataloader:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    if epoch == 0:
                        logging.info(f"LSTM-Transformer: Outputs {outputs.shape}, Y {y_batch.shape}")
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()
                logging.info(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {total_loss / len(dataloader):.4f}")
            
            self.lstm_scalers = {'X': scaler_X, 'y': scaler_y}
            return model, optimizer, {'loss': total_loss / len(dataloader)}
    
        except RuntimeError as e:
            logging.error(f"LSTM-Transformer shape mismatch: {e}")
            return None
    
    def train_bayesian_ridge(self, X: np.ndarray, y_multi: np.ndarray) -> Tuple[Dict[int, BayesianRidge], List[float]]:
        """Train per-horizon Ridge with safe indexing."""
        self.bayesian_models = {}
        r2_scores = []  # List for append
        for h in range(1, self.config.prediction_horizon + 1):
            try:
                y_h = y_multi[:, h-1]  # 0-based
                X_train, X_test, y_train, y_test = train_test_split(X, y_h, test_size=self.config.test_size, shuffle=False)
                model_h = BayesianRidge(alpha_1=self.config.alpha_1, alpha_2=self.config.alpha_2,
                                        lambda_1=self.config.lambda_1, lambda_2=self.config.lambda_2)
                model_h.fit(X_train, y_train)
                y_pred = model_h.predict(X_test)
                r2_h = r2_score(y_test, y_pred)
                r2_scores.append(r2_h)  # FIXED: Append, no [h] index
                self.bayesian_models[h] = model_h
                logging.info(f"Ridge h{h}: R²={r2_h:.3f}")
            except IndexError as e:
                logging.error(f"Ridge loop IndexError at h={h}: {e}")
                traceback.print_exc()
                continue  # Skip horizon, continue loop
        if len(r2_scores) == 0:
            logging.warning("No Ridge horizons trained")
            return {}, []
        avg_r2 = np.mean(r2_scores)
        logging.info(f"Ridge avg R²: {avg_r2:.3f} across {len(r2_scores)} horizons")
        return self.bayesian_models, r2_scores
    
    def train(self, ticker: str, data_processor: DataProcessor):
        """Train all models."""
        logging.info(f"Starting hybrid model training for {ticker}")
        df = data_processor.prepare_features(ticker, self.config)
        if df.empty:
            logging.warning(f"Skipping training for {ticker} due to empty feature set")
            return False
        X_scaled, y_scaled, target_columns = self.prepare_data(df)
        price_series = df['close']
        self.train_arima(price_series)
        self.train_lstm_transformer(X_scaled, y_scaled)
        self.train_bayesian_ridge(X_scaled, y_scaled)
        logging.info(f"Hybrid model training completed for {ticker}")
        return True
    
    def predict(self, recent_data: np.ndarray, price_series: pd.Series) -> Dict[str, np.ndarray]:
        """Generate predictions from all models."""
        predictions = {}
        if self.arima_model is not None:
            try:
                arima_pred = self.arima_model.predict(self.config.prediction_horizon)
                predictions['arima'] = arima_pred
            except Exception as e:
                logging.error(f"ARIMA prediction failed: {e}")
                predictions['arima'] = np.zeros(self.config.prediction_horizon)
        else:
            predictions['arima'] = np.zeros(self.config.prediction_horizon)
        
        if self.lstm_transformer_model is not None and TORCH_AVAILABLE:
            try:
                self.lstm_transformer_model.eval()
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(recent_data[-self.config.sequence_length:]).unsqueeze(0)
                    lstm_pred = self.lstm_transformer_model(input_tensor)
                    predictions['lstm_transformer'] = lstm_pred.numpy()[0]
            except Exception as e:
                logging.error(f"LSTM-Transformer prediction failed: {e}")
                predictions['lstm_transformer'] = np.zeros(self.config.prediction_horizon)
        else:
            predictions['lstm_transformer'] = np.zeros(self.config.prediction_horizon)
        
        if self.bayesian_ridge_models is not None:
            try:
                br_predictions = []
                recent_flat = recent_data[-1:].reshape(1, -1)
                for model in self.bayesian_ridge_models:
                    pred = model.predict(recent_flat)[0]
                    br_predictions.append(pred)
                predictions['bayesian_ridge'] = np.array(br_predictions)
            except Exception as e:
                logging.error(f"Bayesian Ridge prediction failed: {e}")
                predictions['bayesian_ridge'] = np.zeros(self.config.prediction_horizon)
        else:
            predictions['bayesian_ridge'] = np.zeros(self.config.prediction_horizon)
        
        ensemble_pred = (
            predictions['arima'] * self.config.arima_weight +
            predictions['lstm_transformer'] * self.config.lstm_transformer_weight +
            predictions['bayesian_ridge'] * self.config.bayesian_ridge_weight
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
        model_data = {
            'config': self.config,
            'scalers': self.scalers,
            'feature_columns': self.feature_columns,
            'arima_model': self.arima_model,
            'bayesian_ridge_models': self.bayesian_ridge_models
        }
        if self.lstm_transformer_model is not None:
            torch.save(self.lstm_transformer_model.state_dict(), f"{filepath}_lstm_transformer.pth")
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.config = model_data['config']
        self.scalers = model_data['scalers']
        self.feature_columns = model_data['feature_columns']
        self.arima_model = model_data['arima_model']
        self.bayesian_ridge_models = model_data['bayesian_ridge_models']
        pytorch_path = f"{filepath}_lstm_transformer.pth"
        if Path(pytorch_path).exists() and TORCH_AVAILABLE:
            input_size = len(self.feature_columns) + self.config.prediction_horizon
            self.lstm_transformer_model = LSTMTransformerModel(
                input_size=input_size,
                hidden_size=self.config.lstm_hidden_size,
                num_heads=self.config.transformer_num_heads,
                num_layers=self.config.transformer_num_layers,
                dropout_rate=self.config.dropout_rate,
                prediction_horizon=self.config.prediction_horizon
            )
            self.lstm_transformer_model.load_state_dict(torch.load(pytorch_path))
        logging.info(f"Model loaded from {filepath}")

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
    """Train model for a single ticker."""
    try:
        ticker, config, stock_db_path, sentiment_db_path = args
        logging.debug(f"train_ticker: Received ticker={ticker}, config={config.__class__.__name__}")
        data_processor = DataProcessor()
        predictor = HybridStockPredictor(config)
        success = predictor.train(ticker, data_processor)
        if success and predictor.feature_columns:
            Path(config.model_dir).mkdir(exist_ok=True, parents=True)
            model_path = f"{config.model_dir}/hybrid_model_{ticker}.pkl"
            predictor.save_model(model_path)
            return f"Completed training for {ticker}"
        return f"Skipped training for {ticker} due to no features or training failure"
    except ValueError as e:
        logging.error(f"train_ticker args unpacking failed: {e}, args={args}")
        return f"Failed training for {ticker}: Invalid args"
    except Exception as e:
        logging.error(f"train_ticker failed for {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return f"Failed training for {ticker}: {str(e)}"

def main():
    """Main training script."""
    try:
        config_data = load_config()
        config = ModelConfig()  # Defined here
        stock_db_path = Path(config_data['stock_db_path']).resolve()
        sentiment_db_path = Path(config_data['sentiment_db_path']).resolve()
        csv_dir = Path(config_data['csv_dir']).resolve()
        tickers = config_data['tickers']
        
        logging.info(f"Starting main with {len(tickers)} tickers: {tickers}")
        data_processor = DataProcessor()
        available_tickers = data_processor.inspect_database()
        
        missing_tickers = [t for t in tickers if t.upper() not in [at.upper() for at in available_tickers]]
        if missing_tickers:
            logging.info(f"Missing data for tickers: {missing_tickers}. Attempting to populate from CSVs.")
            if not populate_stock_data_from_csv(missing_tickers, str(stock_db_path), str(csv_dir)):
                logging.info("CSV population failed. Falling back to yfinance.")
                if not populate_stock_data(missing_tickers, str(stock_db_path)):
                    logging.error("Failed to populate stock data. Exiting.")
                    print("Training failed: Could not populate stock data.")
                    return
        
        available_tickers = data_processor.inspect_database()
        if not available_tickers:
            logging.error("No tickers available after population attempt. Exiting.")
            print("Training failed: No data available in database.")
            return
        
        successful_tickers = []
        with Pool() as pool:
            args_list = [(ticker, config, str(stock_db_path), str(sentiment_db_path)) for ticker in tickers]
            logging.debug(f"Args list created: {len(args_list)} tickers")
            results = pool.map(train_ticker, args_list)
            for result, ticker in zip(results, tickers):
                logging.info(result)
                if "Completed" in result:
                    successful_tickers.append(ticker)
        
        if successful_tickers:
            print(f"Hybrid model training completed successfully for {len(successful_tickers)} tickers: {successful_tickers}")
            print(f"Models saved in '{config_data['model_dir']}' directory")
        else:
            print("Training failed: No models were trained due to missing data.")
            logging.warning("No models were trained successfully.")
    
    except Exception as e:
        logging.error(f"Training failed: {e}")
        print(f"Training failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()