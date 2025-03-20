"""
Data Fetching Module for Cryptocurrency Trading Bot

This module handles all data-related operations including:
- Connecting to exchange APIs
- Fetching historical market data
- Streaming real-time market data
- Storing data in the database
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import sqlite3
from typing import Dict, List, Optional, Union, Tuple

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger('crypto_trading_bot.data')

class ExchangeConnector:
    """Base class for connecting to cryptocurrency exchanges"""
    
    def __init__(self, exchange_id: str, api_key: str = '', api_secret: str = '', additional_params: Dict = None):
        """
        Initialize exchange connector
        
        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'okx', 'bybit')
            api_key: API key for authenticated requests
            api_secret: API secret for authenticated requests
            additional_params: Additional parameters required by specific exchanges
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.additional_params = additional_params or {}
        
        # Initialize exchange connection
        self.exchange = self._initialize_exchange()
        logger.info(f"Initialized connection to {exchange_id} exchange")
    
    def _initialize_exchange(self):
        """Initialize connection to the exchange"""
        exchange_class = getattr(ccxt, self.exchange_id)
        
        # Prepare exchange parameters
        params = {
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'enableRateLimit': True,
        }
        
        # Add additional parameters if provided
        params.update(self.additional_params)
        
        # Create exchange instance
        exchange = exchange_class(params)
        
        # Load markets
        exchange.load_markets()
        
        return exchange
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        return {
            'id': self.exchange.id,
            'name': self.exchange.name,
            'markets': len(self.exchange.markets),
            'timeframes': self.exchange.timeframes,
            'has': self.exchange.has,
        }
    
    def get_symbols(self) -> List[str]:
        """Get available trading symbols"""
        return self.exchange.symbols
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get current ticker data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary containing ticker data
        """
        return self.exchange.fetch_ticker(symbol)
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', 
                 since: Optional[int] = None, limit: int = 100) -> List:
        """
        Get OHLCV (Open, High, Low, Close, Volume) data
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            since: Timestamp in milliseconds for start time
            limit: Number of candles to fetch
            
        Returns:
            List of OHLCV data
        """
        return self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Dict:
        """
        Get order book for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            limit: Depth of the order book to fetch
            
        Returns:
            Dictionary containing order book data
        """
        return self.exchange.fetch_order_book(symbol, limit)
    
    def get_balance(self) -> Dict:
        """
        Get account balance
        
        Returns:
            Dictionary containing balance information
        """
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret are required for authenticated requests")
        
        return self.exchange.fetch_balance()


class DataManager:
    """Manager class for handling all data operations"""
    
    def __init__(self, exchange_id: str = None):
        """
        Initialize data manager
        
        Args:
            exchange_id: Exchange identifier (default from config)
        """
        self.exchange_id = exchange_id or config.DEFAULT_EXCHANGE
        self.db_conn = self._initialize_database()
        self.exchange_connector = self._initialize_exchange_connector()
        logger.info(f"Initialized DataManager with {self.exchange_id} exchange")
    
    def _initialize_database(self):
        """Initialize database connection"""
        db_config = config.DATABASE
        
        if db_config['type'] == 'sqlite':
            # Ensure directory exists
            db_dir = os.path.dirname(db_config['sqlite_path'])
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)
            
            # Connect to SQLite database
            conn = sqlite3.connect(db_config['sqlite_path'])
            
            # Create tables if they don't exist
            self._create_tables(conn)
            
            return conn
        elif db_config['type'] == 'postgresql':
            # PostgreSQL support would be implemented here
            raise NotImplementedError("PostgreSQL support not implemented yet")
        else:
            raise ValueError(f"Unsupported database type: {db_config['type']}")
    
    def _create_tables(self, conn):
        """Create necessary database tables if they don't exist"""
        cursor = conn.cursor()
        
        # Create OHLCV data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS ohlcv_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            UNIQUE(symbol, timeframe, timestamp)
        )
        ''')
        
        # Create trades table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            order_id TEXT,
            timestamp INTEGER NOT NULL,
            type TEXT NOT NULL,
            side TEXT NOT NULL,
            price REAL NOT NULL,
            amount REAL NOT NULL,
            cost REAL NOT NULL,
            fee REAL,
            fee_currency TEXT
        )
        ''')
        
        conn.commit()
    
    def _initialize_exchange_connector(self):
        """Initialize exchange connector"""
        exchange_config = config.EXCHANGE_CONFIGS.get(self.exchange_id, {})
        
        # Extract credentials and parameters
        api_key = exchange_config.get('api_key', '')
        api_secret = exchange_config.get('api_secret', '')
        
        # Additional parameters specific to each exchange
        additional_params = {}
        
        # Add testnet parameter if available
        if 'testnet' in exchange_config:
            additional_params['testnet'] = exchange_config['testnet']
        
        # Add password for exchanges that require it (like OKX)
        if 'password' in exchange_config:
            additional_params['password'] = exchange_config['password']
        
        return ExchangeConnector(self.exchange_id, api_key, api_secret, additional_params)
    
    def fetch_historical_data(self, symbol: str, timeframe: str = '1h', 
                             days_back: int = 30) -> pd.DataFrame:
        """
        Fetch historical OHLCV data and store in database
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            days_back: Number of days to fetch data for
            
        Returns:
            DataFrame containing historical data
        """
        logger.info(f"Fetching historical data for {symbol} on {timeframe} timeframe")
        
        # Calculate start time
        since = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        # Determine appropriate batch size based on timeframe
        if timeframe.endswith('m'):
            # For minute timeframes, fetch in smaller batches
            limit = 500
        else:
            # For hour/day timeframes, fetch in larger batches
            limit = 1000
        
        all_ohlcv = []
        current_since = since
        
        # Fetch data in batches
        while True:
            try:
                # Fetch batch of OHLCV data
                ohlcv_data = self.exchange_connector.get_ohlcv(
                    symbol, timeframe, current_since, limit
                )
                
                if not ohlcv_data:
                    break
                
                all_ohlcv.extend(ohlcv_data)
                
                # Update since for next batch
                current_since = ohlcv_data[-1][0] + 1
                
                # If we've reached current time, stop
                if current_since > int(datetime.now().timestamp() * 1000):
                    break
                
                # Rate limiting
                time.sleep(self.exchange_connector.exchange.rateLimit / 1000)
                
            except Exception as e:
                logger.error(f"Error fetching historical data: {e}")
                break
        
        # Convert to DataFrame
        if all_ohlcv:
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Store in database
            self._store_ohlcv_data(symbol, timeframe, df)
            
            return df
        else:
            logger.warning(f"No historical data found for {symbol} on {timeframe} timeframe")
            return pd.DataFrame()
    
    def _store_ohlcv_data(self, symbol: str, timeframe: str, df: pd.DataFrame):
        """
        Store OHLCV data in database
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            df: DataFrame containing OHLCV data
        """
        cursor = self.db_conn.cursor()
        
        # Convert timestamp to milliseconds for storage
        df_for_db = df.copy()
        df_for_db['timestamp'] = df_for_db['timestamp'].astype(np.int64) // 10**6
        
        # Insert data into database
        for _, row in df_for_db.iterrows():
            cursor.execute('''
            INSERT OR REPLACE INTO ohlcv_data 
            (symbol, timeframe, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, timeframe, int(row['timestamp']), 
                float(row['open']), float(row['high']), 
                float(row['low']), float(row['close']), 
                float(row['volume'])
            ))
        
        self.db_conn.commit()
        logger.info(f"Stored {len(df)} OHLCV records for {symbol} ({timeframe}) in database")
    
    def get_historical_data(self, symbol: str, timeframe: str = '1h', 
                           limit: int = 100, from_db: bool = True) -> pd.DataFrame:
        """
        Get historical OHLCV data from database or fetch from exchange if not available
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            limit: Number of candles to get
            from_db: Whether to try getting data from database first
            
        Returns:
            DataFrame containing historical data
        """
        if from_db:
            # Try to get data from database first
            cursor = self.db_conn.cursor()
            cursor.execute('''
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT ?
            ''', (symbol, timeframe, limit))
            
            rows = cursor.fetchall()
            
            if rows and len(rows) >= limit:
                # Convert to DataFrame
                df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                return df.sort_values('timestamp').reset_index(drop=True)
        
        # If not in database or not enough data, fetch from exchange
        try:
            ohlcv_data = self.exchange_connector.get_ohlcv(symbol, timeframe, limit=limit)
            
            if ohlcv_data:
                df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Store in database
                self._store_ohlcv_data(symbol, timeframe, df)
                
                return df
            else:
                logger.warning(f"No data found for {symbol} on {timeframe} timeframe")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching data from exchange: {e}")
            return pd.DataFrame()
    
    def get_real_time_data(self, symbol: str) -> Dict:
        """
        Get real-time market data for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary containing real-time market data
        """
        try:
            # Get ticker data
            ticker = self.exchange_connector.get_ticker(symbol)
            
            # Get order book
            order_book = self.exchange_connector.get_order_book(symbol)
            
            # Combine data
            real_time_data = {
                'symbol': symbol,
                'timestamp': datetime.now(),
                'last_price': ticker['last'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'volume_24h': ticker['quoteVolume'],
                'change_24h': ticker['percentage'],
                'order_book': {
                    'bids': order_book['bids'][:5],  # Top 5 bids
                    'asks': order_book['asks'][:5],  # Top 5 asks
                }
            }
            
            return real_time_data
            
        except Exception as e:
            logger.error(f"Error fetching real-time data: {e}")
            return {}
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'db_conn') and self.db_conn:
            self.db_conn.close()
            logger.info("Closed database connection")


# Factory function to create data manager for specific exchange
def create_data_manager(exchange_id: str = None) -> DataManager:
    """
    Create a data manager for a specific exchange
    
    Args:
        exchange_id: Exchange identifier (default from config)
        
    Returns:
        DataManager instance
    """
    return DataManager(exchange_id)


# Test function to verify the module works correctly
def test_data_fetching():
    """Test data fetching functionality"""
    # Create data manager
    data_manager = create_data_manager()
    
    # Get exchange info
    exchange_info = data_manager.exchange_connector.get_exchange_info()
    print(f"Connected to {exchange_info['name']} exchange")
    
    # Get available symbols
    symbols = data_manager.exchange_connector.get_symbols()
    print(f"Available symbols: {len(symbols)}")
    
    # Test with BTC/USDT
    symbol = 'BTC/USDT'
    timeframe = '1h'
    
    # Fetch historical data
    print(f"Fetching historical data for {symbol}...")
    df = data_manager.fetch_historical_data(symbol, timeframe, days_back=7)
    
    if not df.empty:
        print(f"Fetched {len(df)} candles")
        print(df.head())
    
    # Get real-time data
    print(f"Fetching real-time data for {symbol}...")
    real_time_data = data_manager.get_real_time_data(symbol)
    
    if real_time_data:
        print(f"Current price: {real_time_data['last_price']}")
        print(f"24h change: {real_time_data['change_24h']}%")
    
    # Close connections
    data_manager.close()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run test
    test_data_fetching()
