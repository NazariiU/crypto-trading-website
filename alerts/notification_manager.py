"""
Signal Alerts and Notifications Module for Cryptocurrency Trading Bot

This module handles sending alerts and notifications to users through various channels:
- Telegram messages
- Email notifications
- Voice alerts
"""

import os
import sys
import logging
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import threading
import queue
import time
import requests

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logger = logging.getLogger('crypto_trading_bot.alerts')

class AlertType:
    """Alert type constants"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SIGNAL = "signal"
    TRADE = "trade"
    PRICE = "price"

class AlertChannel:
    """Alert channel constants"""
    TELEGRAM = "telegram"
    EMAIL = "email"
    VOICE = "voice"
    ALL = "all"

class Alert:
    """Class representing an alert"""
    
    def __init__(self, alert_type: str, message: str, data: Dict = None, 
                channel: str = AlertChannel.ALL, priority: int = 1):
        """
        Initialize alert
        
        Args:
            alert_type: Alert type
            message: Alert message
            data: Additional data
            channel: Alert channel
            priority: Alert priority (1-5, 5 being highest)
        """
        self.alert_type = alert_type
        self.message = message
        self.data = data or {}
        self.channel = channel
        self.priority = min(max(priority, 1), 5)  # Ensure priority is between 1 and 5
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'type': self.alert_type,
            'message': self.message,
            'data': self.data,
            'channel': self.channel,
            'priority': self.priority,
            'timestamp': self.timestamp
        }


class TelegramNotifier:
    """Class for sending notifications via Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token
            chat_id: Telegram chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.enabled = bool(bot_token and chat_id)
        
        if self.enabled:
            logger.info("Telegram notifier initialized")
        else:
            logger.warning("Telegram notifier disabled (missing bot token or chat ID)")
    
    def send_message(self, message: str) -> bool:
        """
        Send message via Telegram
        
        Args:
            message: Message to send
            
        Returns:
            True if message was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Telegram notifier is disabled")
            return False
        
        try:
            url = f"{self.api_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data)
            
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Failed to send Telegram message: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending Telegram message: {e}")
            return False


class EmailNotifier:
    """Class for sending notifications via email"""
    
    def __init__(self, smtp_server: str, smtp_port: int, sender_email: str, 
                sender_password: str, recipient_email: str):
        """
        Initialize email notifier
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            sender_email: Sender email address
            sender_password: Sender email password
            recipient_email: Recipient email address
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.enabled = bool(smtp_server and smtp_port and sender_email and 
                          sender_password and recipient_email)
        
        if self.enabled:
            logger.info("Email notifier initialized")
        else:
            logger.warning("Email notifier disabled (missing configuration)")
    
    def send_email(self, subject: str, message: str, html: bool = False) -> bool:
        """
        Send email
        
        Args:
            subject: Email subject
            message: Email message
            html: Whether message is HTML
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Email notifier is disabled")
            return False
        
        try:
            # Create message
            email_message = MIMEMultipart("alternative")
            email_message["Subject"] = subject
            email_message["From"] = self.sender_email
            email_message["To"] = self.recipient_email
            
            # Add message
            if html:
                email_message.attach(MIMEText(message, "html"))
            else:
                email_message.attach(MIMEText(message, "plain"))
            
            # Create secure connection and send email
            context = ssl.create_default_context()
            
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, context=context) as server:
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, self.recipient_email, email_message.as_string())
            
            logger.info("Email sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False


class VoiceNotifier:
    """Class for sending voice notifications"""
    
    def __init__(self):
        """Initialize voice notifier"""
        self.enabled = False
        logger.warning("Voice notifier is not implemented yet")
    
    def speak(self, message: str) -> bool:
        """
        Speak message
        
        Args:
            message: Message to speak
            
        Returns:
            True if message was spoken successfully, False otherwise
        """
        if not self.enabled:
            logger.warning("Voice notifier is disabled")
            return False
        
        logger.warning("Voice notifications not implemented yet")
        return False


class NotificationManager:
    """Manager class for handling alerts and notifications"""
    
    def __init__(self):
        """Initialize notification manager"""
        # Initialize notification channels
        self._init_telegram()
        self._init_email()
        self._init_voice()
        
        # Initialize alert queue and history
        self.alert_queue = queue.Queue()
        self.alert_history = []
        
        # Initialize notification thread
        self.notification_thread = None
        self.is_running = False
        
        logger.info("Notification manager initialized")
    
    def _init_telegram(self):
        """Initialize Telegram notifier"""
        telegram_config = config.NOTIFICATIONS.get('telegram', {})
        
        if telegram_config.get('enabled', False):
            self.telegram = TelegramNotifier(
                bot_token=telegram_config.get('bot_token', ''),
                chat_id=telegram_config.get('chat_id', '')
            )
        else:
            self.telegram = None
    
    def _init_email(self):
        """Initialize email notifier"""
        email_config = config.NOTIFICATIONS.get('email', {})
        
        if email_config.get('enabled', False):
            self.email = EmailNotifier(
                smtp_server=email_config.get('smtp_server', ''),
                smtp_port=email_config.get('smtp_port', 587),
                sender_email=email_config.get('sender_email', ''),
                sender_password=email_config.get('sender_password', ''),
                recipient_email=email_config.get('recipient_email', '')
            )
        else:
            self.email = None
    
    def _init_voice(self):
        """Initialize voice notifier"""
        voice_config = config.NOTIFICATIONS.get('voice', {})
        
        if voice_config.get('enabled', False):
            self.voice = VoiceNotifier()
        else:
            self.voice = None
    
    def start(self):
        """Start notification manager"""
        if self.is_running:
            logger.warning("Notification manager is already running")
            return
        
        # Start notification thread
        self.is_running = True
        self.notification_thread = threading.Thread(target=self._notification_loop)
        self.notification_thread.daemon = True
        self.notification_thread.start()
        
        logger.info("Notification manager started")
    
    def stop(self):
        """Stop notification manager"""
        if not self.is_running:
            logger.warning("Notification manager is not running")
            return
        
        # Stop notification thread
        self.is_running = False
        if self.notification_thread:
            self.notification_thread.join(timeout=5.0)
        
        logger.info("Notification manager stopped")
    
    def _notification_loop(self):
        """Main notification loop"""
        while self.is_running:
            try:
                # Process alerts from queue
                while not self.alert_queue.empty():
                    alert = self.alert_queue.get()
                    self._process_alert(alert)
                
                # Sleep to avoid high CPU usage
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in notification loop: {e}")
                time.sleep(5.0)
    
    def _process_alert(self, alert: Alert):
        """
        Process alert and send notifications
        
        Args:
            alert: Alert to process
        """
        # Add alert to history
        self.alert_history.append(alert.to_dict())
        
        # Limit history size
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        # Format alert message
        formatted_message = self._format_alert_message(alert)
        
        # Send notifications based on channel
        if alert.channel in [AlertChannel.TELEGRAM, AlertChannel.ALL]:
            self._send_telegram_notification(alert, formatted_message)
        
        if alert.channel in [AlertChannel.EMAIL, AlertChannel.ALL]:
            self._send_email_notification(alert, formatted_message)
        
        if alert.channel in [AlertChannel.VOICE, AlertChannel.ALL]:
            self._send_voice_notification(alert, formatted_message)
    
    def _format_alert_message(self, alert: Alert) -> str:
        """
        Format alert message
        
        Args:
            alert: Alert to format
            
        Returns:
            Formatted message
        """
        # Format timestamp
        timestamp_str = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format message based on alert type
        if alert.alert_type == AlertType.SIGNAL:
            symbol = alert.data.get('symbol', 'Unknown')
            signal = alert.data.get('signal', 0)
            price = alert.data.get('price', 0.0)
            strategy = alert.data.get('strategy', 'Unknown')
            
            signal_type = "BUY" if signal > 0 else "SELL" if signal < 0 else "NEUTRAL"
            
            return (
                f"🚨 TRADING SIGNAL: {signal_type} 🚨\n\n"
                f"Symbol: {symbol}\n"
                f"Price: ${price:.2f}\n"
                f"Strategy: {strategy}\n"
                f"Time: {timestamp_str}"
            )
        
        elif alert.alert_type == AlertType.TRADE:
            symbol = alert.data.get('symbol', 'Unknown')
            side = alert.data.get('side', 'Unknown')
            amount = alert.data.get('amount', 0.0)
            price = alert.data.get('price', 0.0)
            
            return (
                f"💰 TRADE EXECUTED: {side.upper()} 💰\n\n"
                f"Symbol: {symbol}\n"
                f"Amount: {amount}\n"
                f"Price: ${price:.2f}\n"
                f"Time: {timestamp_str}"
            )
        
        elif alert.alert_type == AlertType.PRICE:
            symbol = alert.data.get('symbol', 'Unknown')
            price = alert.data.get('price', 0.0)
            change = alert.data.get('change', 0.0)
            
            return (
                f"📊 PRICE ALERT 📊\n\n"
                f"Symbol: {symbol}\n"
                f"Price: ${price:.2f}\n"
                f"Change: {change:.2f}%\n"
                f"Time: {timestamp_str}"
            )
        
        else:
            # Default formatting
            return f"[{alert.alert_type.upper()}] {alert.message} ({timestamp_str})"
    
    def _send_telegram_notification(self, alert: Alert, message: str):
        """
        Send notification via Telegram
        
        Args:
            alert: Alert to send
            message: Formatted message
        """
        if not self.telegram:
            return
        
        try:
            self.telegram.send_message(message)
        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
    
    def _send_email_notification(self, alert: Alert, message: str):
        """
        Send notification via email
        
        Args:
            alert: Alert to send
            message: Formatted message
        """
        if not self.email:
            return
        
        try:
            subject = f"Crypto Trading Bot: {alert.alert_type.upper()} Alert"
            self.email.send_email(subject, message)
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def _send_voice_notification(self, alert: Alert, message: str):
        """
        Send notification via voice
        
        Args:
            alert: Alert to send
            message: Formatted message
        """
        if not self.voice:
            return
        
        try:
            self.voice.speak(message)
        except Exception as e:
            logger.error(f"Error sending voice notification: {e}")
    
    def add_alert(self, alert_type: str, message: str, data: Dict = None, 
                channel: str = AlertChannel.ALL, priority: int = 1):
        """
        Add alert to queue
        
        Args:
            alert_type: Alert type
            message: Alert message
            data: Additional data
            channel: Alert channel
            priority: Alert priority
        """
        alert = Alert(alert_type, message, data, channel, priority)
        self.alert_queue.put(alert)
        logger.info(f"Added {alert_type} alert to queue: {message}")
    
    def add_signal_alert(self, symbol: str, signal: int, price: float, strategy: str, 
                       channel: str = AlertChannel.ALL, priority: int = 3):
        """
        Add signal alert
        
        Args:
            symbol: Trading symbol
            signal: Signal value (1 for buy, -1 for sell, 0 for neutral)
            price: Current price
            strategy: Strategy name
            channel: Alert channel
            priority: Alert priority
        """
        signal_type = "BUY" if signal > 0 else "SELL" if signal < 0 else "NEUTRAL"
        
        self.add_alert(
            alert_type=AlertType.SIGNAL,
            message=f"{signal_type} signal for {symbol} at ${price:.2f}",
            data={
                'symbol': symbol,
                'signal': signal,
                'price': price,
                'strategy': strategy
            },
            channel=channel,
            priority=priority
        )
    
    def add_trade_alert(self, symbol: str, side: str, amount: float, price: float, 
                      channel: str = AlertChannel.ALL, priority: int = 4):
        """
        Add trade alert
        
        Args:
            symbol: Trading symbol
            side: Trade side (buy, sell)
            amount: Trade amount
            price: Trade price
            channel: Alert channel
            priority: Alert priority
        """
        self.add_alert(
            alert_type=AlertType.TRADE,
            message=f"{side.upper()} {amount} {symbol} at ${price:.2f}",
            data={
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price
            },
            channel=channel,
            priority=priority
        )
    
    def add_price_alert(self, symbol: str, price: float, change: float, 
                      channel: str = AlertChannel.ALL, priority: int = 2):
        """
        Add price alert
        
        Args:
            symbol: Trading symbol
            price: Current price
            change: Price change percentage
            channel: Alert channel
            priority: Alert priority
        """
        self.add_alert(
            alert_type=AlertType.PRICE,
            message=f"Price alert for {symbol}: ${price:.2f} ({change:.2f}%)",
            data={
                'symbol': symbol,
                'price': price,
                'change': change
            },
            channel=channel,
            priority=priority
        )
    
    def get_alert_history(self) -> List[Dict]:
        """
        Get alert history
        
        Returns:
            List of alerts
        """
        return self.alert_history


# Factory function to create notification manager
def create_notification_manager() -> NotificationManager:
    """
    Create a notification manager
    
    Returns:
        NotificationManager instance
    """
    return NotificationManager()


# Test function to verify the module works correctly
def test_notifications():
    """Test notification functionality"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create notification manager
    manager = create_notification_manager()
    
    try:
        # Start notification manager
        manager.start()
        
        # Add test alerts
        manager.add_alert(
            alert_type=AlertType.INFO,
            message="This is a test info alert",
            priority=1
        )
        
        manager.add_signal_alert(
            symbol="BTC/USDT",
            signal=1,
            price=50000.0,
            strategy="RSI_MACD"
        )
        
        manager.add_trade_alert(
            symbol="BTC/USDT",
            side="buy",
            amount=0.1,
            price=50000.0
        )
        
        manager.add_price_alert(
            symbol="BTC/USDT",
            price=50000.0,
            change=5.0
        )
        
        # Wait for alerts to be processed
        logger.info("Waiting for alerts to be processed (10 seconds)...")
        time.sleep(10)
        
        # Get alert history
        history = manager.get_alert_history()
        logger.info(f"Alert history: {len(history)} alerts")
        
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        # Stop notification manager
        manager.stop()
        logger.info("Test completed")


if __name__ == "__main__":
    # Run test
    test_notifications()
