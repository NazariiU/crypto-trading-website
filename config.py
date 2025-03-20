"""
Configuration file for the Cryptocurrency Trading Bot
"""

# Exchange API configurations
EXCHANGE_CONFIGS = {
    'binance': {
        'api_key': '',  # Your Binance API key
        'api_secret': '',  # Your Binance API secret
        'testnet': True,  # Use testnet for development
    },
    'okx': {
        'api_key': '',  # Your OKX API key
        'api_secret': '',  # Your OKX API secret
        'password': '',  # Your OKX API password
        'testnet': True,  # Use testnet for development
    },
    'bybit': {
        'api_key': '',  # Your Bybit API key
        'api_secret': '',  # Your Bybit API secret
        'testnet': True,  # Use testnet for development
    }
}

# Default exchange to use
DEFAULT_EXCHANGE = 'binance'

# Database configuration
DATABASE = {
    'type': 'sqlite',  # 'sqlite' or 'postgresql'
    'sqlite_path': 'data/trading_data.db',
    'postgresql': {
        'host': 'localhost',
        'port': 5432,
        'database': 'crypto_trading',
        'user': 'postgres',
        'password': '',
    }
}

# Trading parameters
TRADING_PARAMS = {
    'default_symbols': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'],
    'default_timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
    'default_strategies': ['rsi_macd', 'bollinger_bands', 'volume_profile'],
    'risk_management': {
        'max_position_size_percent': 5.0,  # Maximum percentage of balance per trade
        'default_stop_loss_percent': 2.0,  # Default stop loss percentage
        'default_take_profit_percent': 5.0,  # Default take profit percentage
        'max_open_trades': 3,  # Maximum number of concurrent open trades
    }
}

# Technical indicators configuration
INDICATORS = {
    'sma': {
        'short_period': 9,
        'medium_period': 21,
        'long_period': 50,
    },
    'ema': {
        'short_period': 9,
        'medium_period': 21,
        'long_period': 50,
    },
    'rsi': {
        'period': 14,
        'overbought': 70,
        'oversold': 30,
    },
    'macd': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
    },
    'bollinger_bands': {
        'period': 20,
        'std_dev': 2,
    },
    'volume_profile': {
        'bins': 20,
        'lookback_periods': 100,
    }
}

# Notification settings
NOTIFICATIONS = {
    'telegram': {
        'enabled': False,
        'bot_token': '',  # Your Telegram bot token
        'chat_id': '',  # Your Telegram chat ID
    },
    'email': {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender_email': '',  # Your email address
        'sender_password': '',  # Your email password or app password
        'recipient_email': '',  # Recipient email address
    },
    'voice': {
        'enabled': False,
    }
}

# UI Configuration
UI_CONFIG = {
    'theme': 'dark',  # 'dark' or 'light'
    'default_charts': ['candlestick', 'volume', 'indicators'],
    'refresh_rate': 5,  # Data refresh rate in seconds
    'port': 8501,  # Streamlit port
}

# Logging configuration
LOGGING = {
    'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_to_file': True,
    'log_file': 'logs/trading_bot.log',
}
