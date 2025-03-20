"""
User authentication module for the cryptocurrency trading web application.

This module provides functions for user authentication, registration,
and session management using streamlit-authenticator.
"""

import os
import yaml
import streamlit as st
import streamlit_authenticator as stauth
import logging
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger('crypto_trading_bot.auth')

# Authentication configuration path
AUTH_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

def load_auth_config():
    """
    Load authentication configuration from YAML file
    
    Returns:
        dict: Authentication configuration
    """
    if not os.path.exists(AUTH_CONFIG_PATH):
        # Create default configuration
        auth_config = {
            'credentials': {
                'usernames': {
                    'admin': {
                        'email': 'admin@example.com',
                        'name': 'Admin User',
                        'password': stauth.Hasher(['admin']).generate()[0]
                    }
                }
            },
            'cookie': {
                'expiry_days': 30,
                'key': 'crypto_trading_bot_auth',
                'name': 'crypto_trading_bot_auth'
            }
        }
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(AUTH_CONFIG_PATH), exist_ok=True)
        
        # Save configuration
        with open(AUTH_CONFIG_PATH, 'w') as f:
            yaml.dump(auth_config, f)
        
        return auth_config
    
    # Load existing configuration
    with open(AUTH_CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_auth_config(auth_config):
    """
    Save authentication configuration to YAML file
    
    Args:
        auth_config (dict): Authentication configuration
    """
    with open(AUTH_CONFIG_PATH, 'w') as f:
        yaml.dump(auth_config, f)

def create_authenticator():
    """
    Create authenticator object
    
    Returns:
        Authenticate: Authenticator object
    """
    auth_config = load_auth_config()
    
    authenticator = stauth.Authenticate(
        auth_config['credentials'],
        auth_config['cookie']['name'],
        auth_config['cookie']['key'],
        auth_config['cookie']['expiry_days']
    )
    
    return authenticator

def register_user(username, name, email, password):
    """
    Register a new user
    
    Args:
        username (str): Username
        name (str): Full name
        email (str): Email address
        password (str): Password
        
    Returns:
        bool: True if registration successful, False otherwise
        str: Error message if registration failed
    """
    # Load current config
    auth_config = load_auth_config()
    
    # Check if username already exists
    if username in auth_config['credentials']['usernames']:
        return False, "Username already exists"
    
    # Add new user
    auth_config['credentials']['usernames'][username] = {
        'email': email,
        'name': name,
        'password': stauth.Hasher([password]).generate()[0]
    }
    
    # Save updated config
    save_auth_config(auth_config)
    
    logger.info(f"New user registered: {username}")
    
    return True, "Registration successful"

def change_password(username, current_password, new_password):
    """
    Change user password
    
    Args:
        username (str): Username
        current_password (str): Current password
        new_password (str): New password
        
    Returns:
        bool: True if password change successful, False otherwise
        str: Error message if password change failed
    """
    # Load current config
    auth_config = load_auth_config()
    
    # Check if username exists
    if username not in auth_config['credentials']['usernames']:
        return False, "User not found"
    
    # Verify current password
    hasher = stauth.Hasher([current_password])
    if not hasher.check(auth_config['credentials']['usernames'][username]['password']):
        return False, "Current password is incorrect"
    
    # Update password
    auth_config['credentials']['usernames'][username]['password'] = stauth.Hasher([new_password]).generate()[0]
    
    # Save updated config
    save_auth_config(auth_config)
    
    logger.info(f"Password changed for user: {username}")
    
    return True, "Password changed successfully"

def update_user_profile(username, name=None, email=None):
    """
    Update user profile
    
    Args:
        username (str): Username
        name (str, optional): New full name
        email (str, optional): New email address
        
    Returns:
        bool: True if profile update successful, False otherwise
        str: Error message if profile update failed
    """
    # Load current config
    auth_config = load_auth_config()
    
    # Check if username exists
    if username not in auth_config['credentials']['usernames']:
        return False, "User not found"
    
    # Update profile
    if name:
        auth_config['credentials']['usernames'][username]['name'] = name
    
    if email:
        auth_config['credentials']['usernames'][username]['email'] = email
    
    # Save updated config
    save_auth_config(auth_config)
    
    logger.info(f"Profile updated for user: {username}")
    
    return True, "Profile updated successfully"

def get_user_info(username):
    """
    Get user information
    
    Args:
        username (str): Username
        
    Returns:
        dict: User information
    """
    # Load current config
    auth_config = load_auth_config()
    
    # Check if username exists
    if username not in auth_config['credentials']['usernames']:
        return None
    
    # Return user info
    user_info = auth_config['credentials']['usernames'][username].copy()
    user_info.pop('password', None)  # Remove password from returned info
    
    return user_info

def is_admin(username):
    """
    Check if user is an admin
    
    Args:
        username (str): Username
        
    Returns:
        bool: True if user is admin, False otherwise
    """
    # For now, only the 'admin' user is an admin
    return username == 'admin'
