"""
Database configuration module for the cryptocurrency trading web application.

This module provides functions for database connection, initialization,
and management for the web version of the cryptocurrency trading bot.
"""

import os
import logging
import sqlite3
import psycopg2
from psycopg2 import pool
from dotenv import load_dotenv
from contextlib import contextmanager

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger('crypto_trading_bot.database')

# Database configuration
DB_TYPE = os.getenv('DB_TYPE', 'sqlite')  # 'sqlite' or 'postgres'
DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'crypto_trading.db'))
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'crypto_trading')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
DB_POOL_MIN = int(os.getenv('DB_POOL_MIN', '1'))
DB_POOL_MAX = int(os.getenv('DB_POOL_MAX', '10'))

# Connection pool for PostgreSQL
pg_pool = None

def init_db():
    """
    Initialize database connection and create tables if they don't exist
    """
    if DB_TYPE == 'postgres':
        _init_postgres()
    else:
        _init_sqlite()

def _init_postgres():
    """
    Initialize PostgreSQL database connection and create tables if they don't exist
    """
    global pg_pool
    
    try:
        # Create connection pool
        pg_pool = psycopg2.pool.ThreadedConnectionPool(
            DB_POOL_MIN,
            DB_POOL_MAX,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        logger.info("PostgreSQL connection pool created")
        
        # Create tables
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Create trades table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(100) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        strategy VARCHAR(50) NOT NULL,
                        side VARCHAR(10) NOT NULL,
                        entry_time TIMESTAMP NOT NULL,
                        entry_price NUMERIC(20, 8) NOT NULL,
                        amount NUMERIC(20, 8) NOT NULL,
                        exit_time TIMESTAMP,
                        exit_price NUMERIC(20, 8),
                        pnl NUMERIC(20, 8),
                        pnl_percent NUMERIC(10, 4),
                        status VARCHAR(10) NOT NULL,
                        stop_loss NUMERIC(20, 8),
                        take_profit NUMERIC(20, 8),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create api_keys table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(100) NOT NULL,
                        exchange VARCHAR(50) NOT NULL,
                        api_key VARCHAR(100) NOT NULL,
                        api_secret VARCHAR(100) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, exchange)
                    )
                """)
                
                # Create settings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS settings (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(100) NOT NULL,
                        setting_key VARCHAR(50) NOT NULL,
                        setting_value TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, setting_key)
                    )
                """)
                
                # Create alerts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS alerts (
                        id SERIAL PRIMARY KEY,
                        user_id VARCHAR(100) NOT NULL,
                        symbol VARCHAR(20) NOT NULL,
                        alert_type VARCHAR(20) NOT NULL,
                        condition VARCHAR(20) NOT NULL,
                        price NUMERIC(20, 8) NOT NULL,
                        message TEXT,
                        triggered BOOLEAN DEFAULT FALSE,
                        triggered_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                
                logger.info("PostgreSQL tables created")
                
    except Exception as e:
        logger.error(f"Error initializing PostgreSQL database: {e}")
        raise

def _init_sqlite():
    """
    Initialize SQLite database connection and create tables if they don't exist
    """
    try:
        # Create directory if not exists
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        # Create tables
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    entry_price REAL NOT NULL,
                    amount REAL NOT NULL,
                    exit_time TIMESTAMP,
                    exit_price REAL,
                    pnl REAL,
                    pnl_percent REAL,
                    status TEXT NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create api_keys table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    api_secret TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, exchange)
                )
            """)
            
            # Create settings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    setting_key TEXT NOT NULL,
                    setting_value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, setting_key)
                )
            """)
            
            # Create alerts table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    price REAL NOT NULL,
                    message TEXT,
                    triggered INTEGER DEFAULT 0,
                    triggered_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            
            logger.info("SQLite tables created")
            
    except Exception as e:
        logger.error(f"Error initializing SQLite database: {e}")
        raise

@contextmanager
def get_db_connection():
    """
    Get database connection
    
    Yields:
        Connection: Database connection
    """
    conn = None
    try:
        if DB_TYPE == 'postgres':
            conn = pg_pool.getconn()
        else:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
        
        yield conn
        
    except Exception as e:
        logger.error(f"Error getting database connection: {e}")
        raise
    
    finally:
        if conn:
            if DB_TYPE == 'postgres':
                pg_pool.putconn(conn)
            else:
                conn.close()

def save_api_keys(user_id, exchange, api_key, api_secret):
    """
    Save API keys for a user
    
    Args:
        user_id (str): User ID
        exchange (str): Exchange name
        api_key (str): API key
        api_secret (str): API secret
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            if DB_TYPE == 'postgres':
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO api_keys (user_id, exchange, api_key, api_secret)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (user_id, exchange)
                        DO UPDATE SET
                            api_key = EXCLUDED.api_key,
                            api_secret = EXCLUDED.api_secret
                    """, (user_id, exchange, api_key, api_secret))
                    conn.commit()
            else:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO api_keys (user_id, exchange, api_key, api_secret)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (user_id, exchange)
                    DO UPDATE SET
                        api_key = excluded.api_key,
                        api_secret = excluded.api_secret
                """, (user_id, exchange, api_key, api_secret))
                conn.commit()
        
        logger.info(f"API keys saved for user {user_id} and exchange {exchange}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving API keys: {e}")
        return False

def get_api_keys(user_id, exchange):
    """
    Get API keys for a user
    
    Args:
        user_id (str): User ID
        exchange (str): Exchange name
        
    Returns:
        dict: API keys or None if not found
    """
    try:
        with get_db_connection() as conn:
            if DB_TYPE == 'postgres':
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT api_key, api_secret
                        FROM api_keys
                        WHERE user_id = %s AND exchange = %s
                    """, (user_id, exchange))
                    result = cursor.fetchone()
            else:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT api_key, api_secret
                    FROM api_keys
                    WHERE user_id = ? AND exchange = ?
                """, (user_id, exchange))
                result = cursor.fetchone()
            
            if result:
                if DB_TYPE == 'postgres':
                    return {
                        'api_key': result[0],
                        'api_secret': result[1]
                    }
                else:
                    return {
                        'api_key': result['api_key'],
                        'api_secret': result['api_secret']
                    }
            else:
                return None
                
    except Exception as e:
        logger.error(f"Error getting API keys: {e}")
        return None

def save_setting(user_id, key, value):
    """
    Save a setting for a user
    
    Args:
        user_id (str): User ID
        key (str): Setting key
        value (str): Setting value
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            if DB_TYPE == 'postgres':
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO settings (user_id, setting_key, setting_value)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (user_id, setting_key)
                        DO UPDATE SET
                            setting_value = EXCLUDED.setting_value,
                            updated_at = CURRENT_TIMESTAMP
                    """, (user_id, key, value))
                    conn.commit()
            else:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO settings (user_id, setting_key, setting_value)
                    VALUES (?, ?, ?)
                    ON CONFLICT (user_id, setting_key)
                    DO UPDATE SET
                        setting_value = excluded.setting_value,
                        updated_at = CURRENT_TIMESTAMP
                """, (user_id, key, value))
                conn.commit()
        
        logger.info(f"Setting {key} saved for user {user_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving setting: {e}")
        return False

def get_setting(user_id, key, default=None):
    """
    Get a setting for a user
    
    Args:
        user_id (str): User ID
        key (str): Setting key
        default: Default value if setting not found
        
    Returns:
        str: Setting value or default if not found
    """
    try:
        with get_db_connection() as conn:
            if DB_TYPE == 'postgres':
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT setting_value
                        FROM settings
                        WHERE user_id = %s AND setting_key = %s
                    """, (user_id, key))
                    result = cursor.fetchone()
            else:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT setting_value
                    FROM settings
                    WHERE user_id = ? AND setting_key = ?
                """, (user_id, key))
                result = cursor.fetchone()
            
            if result:
                if DB_TYPE == 'postgres':
                    return result[0]
                else:
                    return result['setting_value']
            else:
                return default
                
    except Exception as e:
        logger.error(f"Error getting setting: {e}")
        return default

def log_trade(user_id, trade_data):
    """
    Log a trade
    
    Args:
        user_id (str): User ID
        trade_data (dict): Trade data
        
    Returns:
        int: Trade ID or None if failed
    """
    try:
        with get_db_connection() as conn:
            if DB_TYPE == 'postgres':
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO trades (
                            user_id, symbol, strategy, side, entry_time, entry_price,
                            amount, status, stop_loss, take_profit
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        user_id,
                        trade_data['symbol'],
                        trade_data['strategy'],
                        trade_data['side'],
                        trade_data['entry_time'],
                        trade_data['entry_price'],
                        trade_data['amount'],
                        trade_data['status'],
                        trade_data.get('stop_loss'),
                        trade_data.get('take_profit')
                    ))
                    trade_id = cursor.fetchone()[0]
                    conn.commit()
            else:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trades (
                        user_id, symbol, strategy, side, entry_time, entry_price,
                        amount, status, stop_loss, take_profit
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    trade_data['symbol'],
                    trade_data['strategy'],
                    trade_data['side'],
                    trade_data['entry_time'],
                    trade_data['entry_price'],
                    trade_data['amount'],
                    trade_data['status'],
                    trade_data.get('stop_loss'),
                    trade_data.get('take_profit')
                ))
                trade_id = cursor.lastrowid
                conn.commit()
        
        logger.info(f"Trade logged for user {user_id}, ID: {trade_id}")
        return trade_id
        
    except Exception as e:
        logger.error(f"Error logging trade: {e}")
        return None

def update_trade(user_id, trade_id, trade_data):
    """
    Update a trade
    
    Args:
        user_id (str): User ID
        trade_id (int): Trade ID
        trade_data (dict): Trade data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            if DB_TYPE == 'postgres':
                with conn.cursor() as cursor:
                    # Build update query dynamically based on provided fields
                    update_fields = []
                    params = []
                    
                    for key, value in trade_data.items():
                        if key not in ['id', 'user_id', 'created_at']:
                            update_fields.append(f"{key} = %s")
                            params.append(value)
                    
                    # Add trade_id and user_id to params
                    params.append(trade_id)
                    params.append(user_id)
                    
                    query = f"""
                        UPDATE trades
                        SET {', '.join(update_fields)}
                        WHERE id = %s AND user_id = %s
                    """
                    
                    cursor.execute(query, params)
                    conn.commit()
                    
                    return cursor.rowcount > 0
            else:
                cursor = conn.cursor()
                
                # Build update query dynamically based on provided fields
                update_fields = []
                params = []
                
                for key, value in trade_data.items():
                    if key not in ['id', 'user_id', 'created_at']:
                        update_fields.append(f"{key} = ?")
                        params.append(value)
                
                # Add trade_id and user_id to params
                params.append(trade_id)
                params.append(user_id)
                
                query = f"""
                    UPDATE trades
                    SET {', '.join(update_fields)}
                    WHERE id = ? AND user_id = ?
                """
                
                cursor.execute(query, params)
                conn.commit()
                
                return cursor.rowcount > 0
                
    except Exception as e:
        logger.error(f"Error updating trade: {e}")
        return False

def get_trades(user_id, symbol=None, strategy=None, status=None, limit=100):
    """
    Get trades for a user
    
    Args:
        user_id (str): User ID
        symbol (str, optional): Filter by symbol
        strategy (str, optional): Filter by strategy
        status (str, optional): Filter by status
        limit (int, optional): Limit number of results
        
    Returns:
        list: List of trades
    """
    try:
        with get_db_connection() as conn:
            # Build query dynamically based on filters
            query = "SELECT * FROM trades WHERE user_id = ?"
            params = [user_id]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY entry_time DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            if DB_TYPE == 'postgres':
                # Convert ? to %s for PostgreSQL
                query = query.replace('?', '%s')
                
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    columns = [desc[0] for desc in cursor.description]
                    trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
            else:
                cursor = conn.cursor()
                cursor.execute(query, params)
                trades = [dict(row) for row in cursor.fetchall()]
            
            return trades
                
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return []

def create_alert(user_id, alert_data):
    """
    Create an alert
    
    Args:
        user_id (str): User ID
        alert_data (dict): Alert data
        
    Returns:
        int: Alert ID or None if failed
    """
    try:
        with get_db_connection() as conn:
            if DB_TYPE == 'postgres':
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO alerts (
                            user_id, symbol, alert_type, condition, price, message
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (
                        user_id,
                        alert_data['symbol'],
                        alert_data['alert_type'],
                        alert_data['condition'],
                        alert_data['price'],
                        alert_data.get('message')
                    ))
                    alert_id = cursor.fetchone()[0]
                    conn.commit()
            else:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO alerts (
                        user_id, symbol, alert_type, condition, price, message
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    alert_data['symbol'],
                    alert_data['alert_type'],
                    alert_data['condition'],
                    alert_data['price'],
                    alert_data.get('message')
                ))
                alert_id = cursor.lastrowid
                conn.commit()
        
        logger.info(f"Alert created for user {user_id}, ID: {alert_id}")
        return alert_id
        
    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        return None

def get_alerts(user_id, symbol=None, triggered=None):
    """
    Get alerts for a user
    
    Args:
        user_id (str): User ID
        symbol (str, optional): Filter by symbol
        triggered (bool, optional): Filter by triggered status
        
    Returns:
        list: List of alerts
    """
    try:
        with get_db_connection() as conn:
            # Build query dynamically based on filters
            query = "SELECT * FROM alerts WHERE user_id = ?"
            params = [user_id]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if triggered is not None:
                if DB_TYPE == 'postgres':
                    query += " AND triggered = %s"
                    params.append(triggered)
                else:
                    query += " AND triggered = ?"
                    params.append(1 if triggered else 0)
            
            query += " ORDER BY created_at DESC"
            
            if DB_TYPE == 'postgres':
                # Convert ? to %s for PostgreSQL
                query = query.replace('?', '%s')
                
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    columns = [desc[0] for desc in cursor.description]
                    alerts = [dict(zip(columns, row)) for row in cursor.fetchall()]
            else:
                cursor = conn.cursor()
                cursor.execute(query, params)
                alerts = [dict(row) for row in cursor.fetchall()]
            
            return alerts
                
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return []

# Initialize database on module import
init_db()
