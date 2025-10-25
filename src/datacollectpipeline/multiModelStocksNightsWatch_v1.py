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
from sklearn.metrics import mean_absolute_percentage_error as mape

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
    sequence_length: int = 60
    prediction_horizon: int = 5
    test_size: float = 0.2
    validation_size: float = 0.1
    arima_order: tuple = (1, 1, 1)  # Simplified ARIMA for stability
    lstm_hidden_size: int = 64  # Reduced complexity
    transformer_num_heads: int = 4
    transformer_num_layers: int = 2
    dropout_rate: float = 0.3
    learning_rate: float = 0.0005  # Lower learning rate
    batch_size: int = 16  # Smaller batch size
    epochs: int = 30  # Reduced epochs
    alpha_1: float = 1e-6
    alpha_2: float = 1e-6
    lambda_1: float = 1e-6
    lambda_2: float = 1e-6
    arima_weight: float = 0.4  # Increased ARIMA weight
    bayesian_ridge_weight: float = 0.3
    lstm_transformer_weight: float = 0.3
    model_dir: str = "models"
    reports_dir: str = "reports"
    include_technical_indicators: bool = True
    include_sentiment: bool = True
    enable_eval: bool = True
    sentiment_backfill: bool = True

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
        """Load sentiment data for a specific ticker with enhanced backfill."""
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
            
            # Enhanced sparse data handling
            if df.empty or len(df) < 100:  # Threshold for sparse data
                logging.warning(f"Sparse sentiment for {ticker}; applying enhanced backfill")
                
                # Get stock data date range
                stock_query_min = "SELECT MIN(date) as min_date FROM stock_prices WHERE UPPER(ticker) = UPPER(?)"
                stock_query_max = "SELECT MAX(date) as max_date FROM stock_prices WHERE UPPER(ticker) = UPPER(?)"
                
                stock_conn = sqlite3.connect(self.stock_db_path)
                min_date = pd.to_datetime(pd.read_sql_query(stock_query_min, stock_conn, params=(ticker,)).iloc[0, 0])
                max_date = pd.to_datetime(pd.read_sql_query(stock_query_max, stock_conn, params=(ticker,)).iloc[0, 0])
                stock_conn.close()
                
                dates = pd.date_range(start=min_date, end=max_date, freq='B')
                df = pd.DataFrame({
                    'avg_sentiment': 0.0,  # Neutral baseline
                    'avg_magnitude': 0.5,  # Moderate magnitude
                    'article_count': 1
                }, index=dates)
                df.index.name = 'date'
                logging.info(f"Generated neutral baseline for {len(df)} days for {ticker}")
            else:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                logging.info(f"Loaded raw sentiment data for {len(df)} days for {ticker}")
            
            conn.close()
            
            # Apply momentum-based smoothing
            df['avg_sentiment'] = df['avg_sentiment'].fillna(0).ewm(span=5, adjust=False).mean()
            df['avg_magnitude'] = df['avg_magnitude'].fillna(0.5).ewm(span=5, adjust=False).mean()
            df['article_count'] = df['article_count'].fillna(0)
            
            non_zero_ratio = (df['avg_sentiment'] != 0).sum() / len(df) * 100
            logging.info(f"Processed sentiment for {len(df)} days; non-zero ratio: {non_zero_ratio:.2f}%")
            
            return df   
     
        except sqlite3.Error as e:
            logging.error(f"Sentiment database error for ticker {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators with proper lag to prevent data leakage."""
        df = df.copy()
        
        # Basic price and volume features (lagged to prevent leakage)
        df['price_change'] = df['close'].pct_change().shift(1)
        df['price_change_abs'] = df['price_change'].abs()
        df['high_low_ratio'] = (df['high'] / df['low']).shift(1)
        df['volume_change'] = df['volume'].pct_change().shift(1)
        
        # Moving averages
        windows = [5, 10, 20, 50]  # Reduced windows for stability
        for window in windows:
            df[f'sma_{window}'] = df['close'].rolling(window=window, min_periods=1).mean().shift(1)
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False, min_periods=1).mean().shift(1)
        
        # Volatility measures
        df['volatility_10'] = df['price_change'].rolling(window=10, min_periods=1).std().shift(1)
        df['volatility_20'] = df['price_change'].rolling(window=20, min_periods=1).std().shift(1)
        
        # Price position indicators
        df['price_position_sma20'] = (df['close'] / df['sma_20']).shift(1)
        df['price_position_sma50'] = (df['close'] / df['sma_50']).shift(1)
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = (df['volume'] / df['volume_sma_20']).shift(1)
        
        # TA-Lib indicators if available
        if TALIB_AVAILABLE:
            try:
                # RSI
                df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
                df['rsi'] = df['rsi'].shift(1)
                
                # MACD
                macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
                df['macd'] = pd.Series(macd, index=df.index).shift(1)
                df['macd_signal'] = pd.Series(macd_signal, index=df.index).shift(1)
                df['macd_histogram'] = pd.Series(macd_hist, index=df.index).shift(1)
                
                # Bollinger Bands
                upper, middle, lower = talib.BBANDS(df['close'].values)
                df['bb_upper'] = pd.Series(upper, index=df.index).shift(1)
                df['bb_middle'] = pd.Series(middle, index=df.index).shift(1)
                df['bb_lower'] = pd.Series(lower, index=df.index).shift(1)
                df['bb_width'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle'])
                df['bb_position'] = ((df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])).shift(1)
                
            except Exception as e:
                logging.warning(f"Error calculating TA-Lib indicators: {e}")
        else:
            # Simple RSI approximation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = (100 - (100 / (1 + rs))).shift(1)
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        return df
    
    def merge_with_sentiment(self, stock_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Merge stock data with sentiment features using backward-looking approach."""
        if sentiment_df.empty:
            logging.info("No sentiment data available; using neutral values")
            stock_df['avg_sentiment_t1'] = 0.0
            stock_df['avg_magnitude_t1'] = 0.5
            stock_df['article_count_t1'] = 0
            stock_df['sent_3d_mean'] = 0.0
            stock_df['sent_7d_mean'] = 0.0
            return stock_df
        
        # Merge data
        merged = stock_df.join(sentiment_df, how="left")
        
        # Create backward-looking sentiment features
        for col in ["avg_sentiment", "avg_magnitude", "article_count"]:
            merged[f"{col}_t1"] = merged[col].shift(1)
        
        # Rolling sentiment means (backward-looking)
        merged["sent_3d_mean"] = merged["avg_sentiment"].rolling(3, min_periods=1).mean().shift(1)
        merged["sent_7d_mean"] = merged["avg_sentiment"].rolling(7, min_periods=1).mean().shift(1)

        # Drop raw columns to prevent leakage
        merged = merged.drop(columns=["avg_sentiment", "avg_magnitude", "article_count"], errors='ignore')

        # Fill remaining NaN values
        merged = merged.fillna(0)

        sentiment_coverage = (merged["sent_7d_mean"] != 0).sum()
        logging.info(f"Merged sentiment (backward-only): {sentiment_coverage} days with sentiment features")
        return merged        
    
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
        if config.include_sentiment:
            sentiment_df = self.load_sentiment_data(ticker)
            stock_df = self.merge_with_sentiment(stock_df, sentiment_df)
            logging.info(f"After sentiment merge: {stock_df.shape}")
        
        # Create return targets (more stable than price levels)
        for horizon in range(1, config.prediction_horizon + 1):
            future_price = stock_df['close'].shift(-horizon)
            stock_df[f'target_{horizon}d'] = np.log(future_price / stock_df['close'])
        
        # Remove rows with NaN targets
        target_columns = [f'target_{i}d' for i in range(1, config.prediction_horizon + 1)]
        before_drop = len(stock_df)
        stock_df = stock_df.dropna(subset=target_columns)
        after_drop = len(stock_df)
        logging.info(f"Dropped {before_drop - after_drop} rows with NaN targets")
        
        # Fill remaining NaN values in features
        feature_cols = [col for col in stock_df.columns if not col.startswith('target_')]
        stock_df[feature_cols] = stock_df[feature_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        if stock_df.empty:
            logging.warning(f"Feature set for {ticker} is empty after processing")
        else:
            logging.info(f"Final feature set for {ticker}: {stock_df.shape}")
        
        return stock_df

class LSTMTransformer(nn.Module):
    """Simplified LSTM + Transformer hybrid"""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.3, prediction_horizon: int = 5):
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
            batch_first=True, activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers with regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, prediction_horizon)
        
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
        
        # Output prediction
        output = self.fc(final_hidden)
        return output

class HybridStockPredictor:
    """Enhanced Hybrid Stock Predictor with improved training stability"""
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
    
    def train_arima(self, price_series: pd.Series) -> bool:
        """Train ARIMA model with enhanced stability."""
        if not STATSMODELS_AVAILABLE:
            logging.warning("ARIMA training skipped - statsmodels not available")
            return False
        
        try:
            # Check stationarity
            adf_result = adfuller(price_series.dropna())
            if adf_result[1] > 0.05:
                logging.info(f"Series may not be stationary (p-value: {adf_result[1]:.4f})")
            
            # Use log returns for stationarity
            returns = np.log(price_series / price_series.shift(1)).dropna()
            
            # Fit ARIMA model
            arima = ARIMA(returns, order=self.config.arima_order)
            self.arima_model = arima.fit()
            
            logging.info(f"ARIMA{self.config.arima_order} fitted successfully")
            logging.info(f"AIC: {self.arima_model.aic:.2f}")
            return True
            
        except Exception as e:
            logging.error(f"ARIMA training failed: {e}")
            self.arima_model = None
            return False
    
    def train_bayesian_ridge(self, X: np.ndarray, y_multi: np.ndarray) -> Tuple[Dict[int, BayesianRidge], List[float]]:
        """Train Bayesian Ridge models with improved regularization."""
        bayesian_models = {}
        r2_scores = []
        
        if X.shape[0] < 50:  # Minimum samples check
            logging.warning("Insufficient data for Bayesian Ridge training")
            return bayesian_models, r2_scores
        
        # Scale features
        feature_scaler = StandardScaler()
        X_scaled = feature_scaler.fit_transform(X)
        
        # Use time-based split instead of random split
        split_idx = int(len(X_scaled) * (1 - self.config.test_size))
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        
        for h in range(1, min(self.config.prediction_horizon + 1, y_multi.shape[1] + 1)):
            try:
                y_h = y_multi[:, h-1]
                y_train, y_test = y_h[:split_idx], y_h[split_idx:]
                
                # Enhanced regularization for financial data
                model_h = BayesianRidge(
                    alpha_1=1e-4,  # Increased regularization
                    alpha_2=1e-4,
                    lambda_1=1e-4,
                    lambda_2=1e-4,
                    fit_intercept=True,
                    compute_score=True
                )
                
                model_h.fit(X_train, y_train)
                y_pred = model_h.predict(X_test)
                
                # Calculate metrics
                r2_h = r2_score(y_test, y_pred)
                mae_h = mean_absolute_error(y_test, y_pred)
                
                bayesian_models[h] = model_h
                r2_scores.append(r2_h)
                
                logging.info(f"Ridge horizon {h}: R²={r2_h:.3f}, MAE={mae_h:.4f}")
                
            except Exception as e:
                logging.error(f"Bayesian Ridge training failed for horizon {h}: {e}")
                continue
        
        if bayesian_models:
            self.scalers['ridge_features'] = feature_scaler
            avg_r2 = np.mean(r2_scores)
            logging.info(f"Bayesian Ridge: {len(bayesian_models)} horizons trained, avg R²: {avg_r2:.3f}")
        
        return bayesian_models, r2_scores
    
    def train_lstm_transformer(self, X_seq: np.ndarray, y_seq: np.ndarray) -> Optional[Tuple]:
        """Train LSTM-Transformer with enhanced stability."""
        if not TORCH_AVAILABLE:
            logging.warning("PyTorch unavailable; skipping LSTM-Transformer")
            return None
        
        try:
            n_samples, seq_len, n_features = X_seq.shape
            
            # Time-based splits
            n_train = int(n_samples * 0.7)
            n_val = int(n_samples * 0.15)
            
            X_train = X_seq[:n_train]
            X_val = X_seq[n_train:n_train+n_val]
            X_test = X_seq[n_train+n_val:]
            
            y_train = y_seq[:n_train]
            y_val = y_seq[n_train:n_train+n_val]
            y_test = y_seq[n_train+n_val:]
            
            # Robust scaling
            scaler_X = StandardScaler()
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
            
            # Create data loaders
            train_dataset = list(zip(torch.FloatTensor(X_train), torch.FloatTensor(y_train)))
            val_dataset = list(zip(torch.FloatTensor(X_val), torch.FloatTensor(y_val)))
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Initialize model
            model = LSTMTransformer(
                input_size=n_features,
                hidden_size=self.config.lstm_hidden_size,
                num_layers=self.config.transformer_num_layers,
                num_heads=self.config.transformer_num_heads,
                dropout=self.config.dropout_rate,
                prediction_horizon=self.config.prediction_horizon
            )
            
            # Training setup with enhanced stability
            optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            criterion = nn.MSELoss()
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 10
            
            for epoch in range(self.config.epochs):
                # Training
                model.train()
                train_loss = 0.0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
                
                # Logging
                if epoch % 5 == 0:
                    logging.info(f"[LSTM] Epoch {epoch+1}/{self.config.epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")
                
                # Early stopping
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
            
            # Restore best model
            if 'best_model_state' in locals():
                model.load_state_dict(best_model_state)
            
            # Store scalers
            self.lstm_scalers = {'X': scaler_X, 'y': scaler_y}
            
            # Simple evaluation on test set
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                test_tensor = torch.FloatTensor(X_test)
                test_outputs = model(test_tensor)
                test_loss = criterion(test_outputs, torch.FloatTensor(y_test)).item()
            
            metrics = {
                'best_val_loss': best_val_loss,
                'test_loss': test_loss,
                'epochs_trained': epoch + 1
            }
            
            logging.info("LSTM-Transformer trained successfully")
            return model, optimizer, metrics
            
        except Exception as e:
            logging.error(f"LSTM training failed: {e}")
            traceback.print_exc()
            return None
    
    def train(self, ticker: str, data_processor: DataProcessor) -> bool:
        """Main training method for hybrid model."""
        try:
            logging.info(f"Starting hybrid model training for {ticker}")
            
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
            
            # Filter numeric columns
            numeric_features = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
            self.feature_columns = numeric_features
            
            if len(self.feature_columns) == 0:
                logging.error(f"No numeric features available for {ticker}")
                return False
            
            logging.info(f"Using {len(self.feature_columns)} features for {ticker}")
            
            X_features = df[self.feature_columns].values
            y_targets = df[target_columns].values
            
            # Validate data
            if X_features.shape[0] < 100:
                logging.warning(f"Insufficient data for {ticker}: {X_features.shape[0]} samples")
                return False
            
            # Train models
            models_trained = 0
            
            # 1. ARIMA Training
            if STATSMODELS_AVAILABLE and 'close' in df.columns:
                if self.train_arima(df['close']):
                    models_trained += 1
            
            # 2. Bayesian Ridge Training
            try:
                self.bayesian_models, r2_scores = self.train_bayesian_ridge(X_features, y_targets)
                if self.bayesian_models:
                    models_trained += 1
                    avg_r2 = np.mean(r2_scores) if r2_scores else 0
                    logging.info(f"Bayesian Ridge trained (avg R²: {avg_r2:.3f})")
            except Exception as e:
                logging.error(f"Bayesian Ridge training failed: {e}")
                self.bayesian_models = {}
            
            # 3. LSTM-Transformer Training
            if TORCH_AVAILABLE and len(X_features) > self.config.sequence_length * 2:
                try:
                    X_seq, y_seq = self.create_sequences(X_features, y_targets, self.config.sequence_length)
                    if X_seq.shape[0] > 50:  # Minimum sequences
                        lstm_results = self.train_lstm_transformer(X_seq, y_seq)
                        if lstm_results:
                            self.lstm_model = lstm_results[0]
                            models_trained += 1
                            logging.info("LSTM-Transformer trained successfully")
                except Exception as e:
                    logging.error(f"LSTM training failed: {e}")
                    self.lstm_model = None
            
            # Configure ensemble weights
            if models_trained == 0:
                logging.error(f"No models successfully trained for {ticker}")
                return False
            
            # Set weights based on available models
            total_weight = 0
            if self.arima_model:
                total_weight += self.config.arima_weight
            if self.bayesian_models:
                total_weight += self.config.bayesian_ridge_weight
            if self.lstm_model:
                total_weight += self.config.lstm_transformer_weight
            
            if total_weight > 0:
                self.ensemble_weights = {
                    'arima': (self.config.arima_weight / total_weight) if self.arima_model else 0,
                    'ridge': (self.config.bayesian_ridge_weight / total_weight) if self.bayesian_models else 0,
                    'lstm': (self.config.lstm_transformer_weight / total_weight) if self.lstm_model else 0
                }
            else:
                # Fallback to equal weights
                equal_weight = 1.0 / models_trained
                self.ensemble_weights = {
                    'arima': equal_weight if self.arima_model else 0,
                    'ridge': equal_weight if self.bayesian_models else 0,
                    'lstm': equal_weight if self.lstm_model else 0
                }
            
            logging.info(f"Hybrid ensemble configured: {self.ensemble_weights}")
            logging.info(f"Successfully trained {models_trained} out of 3 models for {ticker}")
            
            return True
            
        except Exception as e:
            logging.error(f"Training failed for {ticker}: {e}")
            traceback.print_exc()
            return False
    
    def predict(self, recent_data: np.ndarray, price_series: pd.Series = None) -> Dict[str, np.ndarray]:
        """Generate ensemble predictions."""
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
        
        # Bayesian Ridge predictions
        if self.bayesian_models and 'ridge_features' in self.scalers:
            try:
                recent_flat = recent_data[-1:] if len(recent_data.shape) > 1 else recent_data.reshape(1, -1)
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
        
        # Ensemble prediction
        ensemble_pred = (
            predictions['arima'] * self.ensemble_weights.get('arima', 0) +
            predictions['ridge'] * self.ensemble_weights.get('ridge', 0) +
            predictions['lstm'] * self.ensemble_weights.get('lstm', 0)
        )
        predictions['ensemble'] = ensemble_pred
        
        return predictions
    
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
                'ensemble_weights': self.ensemble_weights
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
    """Train model for a single ticker."""
    try:
        ticker, config, stock_db_path, sentiment_db_path = args
        logging.info(f"Starting training for {ticker}")
        
        # Initialize components
        data_processor = DataProcessor()
        predictor = HybridStockPredictor(config)
        
        # Train the model
        success = predictor.train(ticker, data_processor)
        
        if success:
            # Save model
            model_dir = Path(config.model_dir)
            model_dir.mkdir(exist_ok=True, parents=True)
            
            model_path = model_dir / f"hybrid_model_{ticker}.pkl"
            predictor.save_model(str(model_path))
            
            logging.info(f"Model saved for {ticker}")
            return f"Completed training for {ticker}"
        else:
            return f"Failed training for {ticker}"
    
    except Exception as e:
        logging.error(f"Training error for {ticker}: {e}")
        return f"Failed training for {ticker}: {str(e)}"

def main():
    """Main training function."""
    try:
        # Load configuration
        config_data = load_config()
        config = ModelConfig()
        
        # Setup logging directory
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Setup paths
        stock_db_path = Path(config_data['stock_db_path']).resolve()
        sentiment_db_path = Path(config_data['sentiment_db_path']).resolve()
        csv_dir = Path(config_data['csv_dir']).resolve()
        tickers = config_data['tickers']
        
        logging.info(f"Starting training for {len(tickers)} tickers: {tickers}")
        
        # Check database and populate if needed
        data_processor = DataProcessor()
        available_tickers = data_processor.inspect_database()
        
        missing_tickers = [t for t in tickers if t.upper() not in [at.upper() for at in available_tickers]]
        
        if missing_tickers:
            logging.info(f"Missing tickers: {missing_tickers}")
            
            # Try CSV first, then yfinance
            if csv_dir.exists():
                logging.info("Attempting CSV population...")
                if not populate_stock_data_from_csv(missing_tickers, str(stock_db_path), str(csv_dir)):
                    logging.info("CSV failed, trying yfinance...")
                    populate_stock_data_yfinance(missing_tickers, str(stock_db_path))
            else:
                logging.info("No CSV directory, trying yfinance...")
                populate_stock_data_yfinance(missing_tickers, str(stock_db_path))
        
        # Re-check available tickers
        available_tickers = data_processor.inspect_database()
        valid_tickers = [t for t in tickers if t.upper() in [at.upper() for at in available_tickers]]
        
        if not valid_tickers:
            logging.error("No valid tickers found")
            print("Error: No data available for training")
            return
        
        logging.info(f"Training {len(valid_tickers)} valid tickers: {valid_tickers}")
        
        # Train models (sequential for presentation stability)
        successful_tickers = []
        failed_tickers = []
        
        for ticker in valid_tickers:
            args = (ticker, config, str(stock_db_path), str(sentiment_db_path))
            result = train_ticker(args)
            
            if "Completed" in result:
                successful_tickers.append(ticker)
                logging.info(f"✓ {ticker} training completed")
            else:
                failed_tickers.append(ticker)
                logging.warning(f"✗ {ticker} training failed")
        
        # Report results
        print(f"\n{'='*60}")
        print("TRAINING RESULTS SUMMARY")
        print(f"{'='*60}")
        
        if successful_tickers:
            print(f"✓ Successfully trained: {len(successful_tickers)} models")
            print(f"  Tickers: {', '.join(successful_tickers)}")
            print(f"  Models saved in: {config.model_dir}")
        
        if failed_tickers:
            print(f"✗ Failed training: {len(failed_tickers)} models")
            print(f"  Tickers: {', '.join(failed_tickers)}")
        
        print(f"\n{'='*60}")
        print("PRESENTATION NOTES:")
        print("• This is a research prototype demonstrating hybrid ML methodology")
        print("• Focus on ensemble approach combining ARIMA, Ridge, and LSTM-Transformer")
        print("• Emphasize proper data handling and leakage prevention")
        print("• Recommend additional validation before production use")
        print(f"{'='*60}")
        
        if successful_tickers:
            print(f"\nTraining completed successfully! {len(successful_tickers)} models ready for presentation.")
        else:
            print("\nWarning: No models trained successfully. Check data availability and logs.")
    
    except Exception as e:
        error_msg = f"Training failed: {str(e)}"
        logging.error(error_msg)
        traceback.print_exc()
        print(error_msg)

if __name__ == "__main__":
    main()