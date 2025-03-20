"""
API key management module for the cryptocurrency trading web application.

This module provides functions for secure storage and retrieval of API keys
for various cryptocurrency exchanges.
"""

import os
import logging
import base64
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

# Import database manager
from database.db_manager import save_api_keys, get_api_keys

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger('crypto_trading_bot.api_keys')

# Encryption settings
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', 'default_encryption_key_please_change_in_production')
SALT = os.getenv('ENCRYPTION_SALT', 'default_salt_please_change_in_production').encode()

def _get_encryption_key(user_id):
    """
    Generate encryption key for a user
    
    Args:
        user_id (str): User ID
        
    Returns:
        bytes: Encryption key
    """
    # Derive a key from the master key and user_id
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=100000,
    )
    
    # Use user_id as part of the key derivation
    key_material = (ENCRYPTION_KEY + user_id).encode()
    key = base64.urlsafe_b64encode(kdf.derive(key_material))
    
    return key

def encrypt_api_secret(user_id, api_secret):
    """
    Encrypt API secret
    
    Args:
        user_id (str): User ID
        api_secret (str): API secret
        
    Returns:
        str: Encrypted API secret
    """
    try:
        key = _get_encryption_key(user_id)
        f = Fernet(key)
        encrypted_secret = f.encrypt(api_secret.encode())
        return encrypted_secret.decode()
    except Exception as e:
        logger.error(f"Error encrypting API secret: {e}")
        raise

def decrypt_api_secret(user_id, encrypted_secret):
    """
    Decrypt API secret
    
    Args:
        user_id (str): User ID
        encrypted_secret (str): Encrypted API secret
        
    Returns:
        str: Decrypted API secret
    """
    try:
        key = _get_encryption_key(user_id)
        f = Fernet(key)
        decrypted_secret = f.decrypt(encrypted_secret.encode())
        return decrypted_secret.decode()
    except Exception as e:
        logger.error(f"Error decrypting API secret: {e}")
        raise

def store_api_keys(user_id, exchange, api_key, api_secret):
    """
    Store API keys for a user
    
    Args:
        user_id (str): User ID
        exchange (str): Exchange name
        api_key (str): API key
        api_secret (str): API secret
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Encrypt API secret
        encrypted_secret = encrypt_api_secret(user_id, api_secret)
        
        # Save to database
        return save_api_keys(user_id, exchange, api_key, encrypted_secret)
    except Exception as e:
        logger.error(f"Error storing API keys: {e}")
        return False

def retrieve_api_keys(user_id, exchange):
    """
    Retrieve API keys for a user
    
    Args:
        user_id (str): User ID
        exchange (str): Exchange name
        
    Returns:
        dict: API keys or None if not found
    """
    try:
        # Get from database
        api_keys = get_api_keys(user_id, exchange)
        
        if api_keys:
            # Decrypt API secret
            api_keys['api_secret'] = decrypt_api_secret(user_id, api_keys['api_secret'])
            return api_keys
        else:
            return None
    except Exception as e:
        logger.error(f"Error retrieving API keys: {e}")
        return None

def validate_api_keys(exchange, api_key, api_secret):
    """
    Validate API keys with the exchange
    
    Args:
        exchange (str): Exchange name
        api_key (str): API key
        api_secret (str): API secret
        
    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Import ccxt
        import ccxt
        
        # Create exchange instance
        exchange_class = getattr(ccxt, exchange.lower())
        exchange_instance = exchange_class({
            'apiKey': api_key,
            'secret': api_secret,
            'enableRateLimit': True
        })
        
        # Test API keys by fetching balance
        exchange_instance.fetch_balance()
        
        return True
    except Exception as e:
        logger.error(f"Error validating API keys: {e}")
        return False

def get_available_exchanges():
    """
    Get list of available exchanges
    
    Returns:
        list: List of exchange names
    """
    try:
        # Import ccxt
        import ccxt
        
        # Get list of exchanges
        exchanges = ccxt.exchanges
        
        # Filter to include only major exchanges
        major_exchanges = [
            'binance', 'okx', 'bybit', 'coinbase', 'kraken', 
            'kucoin', 'bitfinex', 'huobi', 'ftx', 'bitstamp'
        ]
        
        # Return intersection of available and major exchanges
        return [exchange for exchange in exchanges if exchange in major_exchanges]
    except Exception as e:
        logger.error(f"Error getting available exchanges: {e}")
        return ['binance', 'okx', 'bybit']  # Default list if ccxt import fails
