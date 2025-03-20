"""
Monitoring and maintenance module for the cryptocurrency trading web application.

This module provides functions for monitoring application health, logging,
error tracking, and automated maintenance tasks.
"""

import os
import sys
import logging
import time
import threading
import psutil
import toml
import smtplib
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import configuration
from database.db_manager import get_db_connection

# Load monitoring configuration
MONITORING_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'monitoring_config.toml')
with open(MONITORING_CONFIG_PATH, 'r') as f:
    MONITORING_CONFIG = toml.load(f)

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, MONITORING_CONFIG['logging']['level']),
    format=MONITORING_CONFIG['logging']['format'],
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'monitoring.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('crypto_trading_bot.monitoring')

class HealthCheck:
    """Health check for the application"""
    
    def __init__(self):
        """Initialize health check"""
        self.enabled = MONITORING_CONFIG['health_check']['enabled']
        self.interval = MONITORING_CONFIG['health_check']['interval']
        self.endpoints = MONITORING_CONFIG['health_check']['endpoints']
        self.last_check = None
        self.status = "UNKNOWN"
        self.issues = []
    
    def check_database(self):
        """Check database connection"""
        try:
            with get_db_connection() as conn:
                if conn:
                    return True, "Database connection successful"
                else:
                    return False, "Failed to connect to database"
        except Exception as e:
            return False, f"Database error: {str(e)}"
    
    def check_disk_space(self):
        """Check available disk space"""
        try:
            disk_usage = psutil.disk_usage('/')
            if disk_usage.percent < 90:
                return True, f"Disk space OK: {disk_usage.percent}% used"
            else:
                return False, f"Low disk space: {disk_usage.percent}% used"
        except Exception as e:
            return False, f"Disk space check error: {str(e)}"
    
    def check_memory_usage(self):
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            if memory.percent < 90:
                return True, f"Memory usage OK: {memory.percent}% used"
            else:
                return False, f"High memory usage: {memory.percent}% used"
        except Exception as e:
            return False, f"Memory check error: {str(e)}"
    
    def run_health_check(self):
        """Run health check"""
        if not self.enabled:
            return
        
        self.issues = []
        all_checks_passed = True
        
        # Check database
        db_status, db_message = self.check_database()
        if not db_status:
            all_checks_passed = False
            self.issues.append(db_message)
        
        # Check disk space
        disk_status, disk_message = self.check_disk_space()
        if not disk_status:
            all_checks_passed = False
            self.issues.append(disk_message)
        
        # Check memory usage
        memory_status, memory_message = self.check_memory_usage()
        if not memory_status:
            all_checks_passed = False
            self.issues.append(memory_message)
        
        # Update status
        self.status = "HEALTHY" if all_checks_passed else "UNHEALTHY"
        self.last_check = datetime.now()
        
        # Log status
        if all_checks_passed:
            logger.info(f"Health check passed: {self.status}")
        else:
            logger.warning(f"Health check failed: {self.status}, issues: {self.issues}")
            
            # Send notification if error tracking is enabled
            if MONITORING_CONFIG['error_tracking']['enabled']:
                self.send_alert("Health Check Failed", f"Health check status: {self.status}\nIssues: {self.issues}")
    
    def start_monitoring(self):
        """Start health check monitoring in a separate thread"""
        if not self.enabled:
            logger.info("Health check monitoring is disabled")
            return
        
        def monitor():
            while True:
                try:
                    self.run_health_check()
                    time.sleep(self.interval)
                except Exception as e:
                    logger.error(f"Error in health check monitoring: {e}")
                    time.sleep(60)  # Wait a minute before retrying
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        logger.info(f"Health check monitoring started with interval {self.interval} seconds")
    
    def send_alert(self, subject, message):
        """Send alert email"""
        try:
            # This is a placeholder for actual email sending
            # In a production environment, you would configure SMTP settings
            logger.warning(f"ALERT: {subject} - {message}")
            
            # Log to error tracking file
            with open(os.path.join(log_dir, 'error_tracking.log'), 'a') as f:
                f.write(f"{datetime.now()} - {subject}: {message}\n")
        except Exception as e:
            logger.error(f"Error sending alert: {e}")

class PerformanceMonitor:
    """Performance monitoring for the application"""
    
    def __init__(self):
        """Initialize performance monitor"""
        self.enabled = MONITORING_CONFIG['performance']['enabled'] if 'enabled' in MONITORING_CONFIG['performance'] else True
        self.track_memory = MONITORING_CONFIG['performance']['track_memory_usage']
        self.track_response_time = MONITORING_CONFIG['performance']['track_response_time']
        self.track_db_queries = MONITORING_CONFIG['performance']['track_database_queries']
        self.metrics = {
            'memory_usage': [],
            'response_times': [],
            'db_query_times': []
        }
    
    def track_memory_usage(self):
        """Track memory usage"""
        if not self.enabled or not self.track_memory:
            return
        
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage = memory_info.rss / 1024 / 1024  # Convert to MB
            
            self.metrics['memory_usage'].append({
                'timestamp': datetime.now(),
                'value': memory_usage
            })
            
            # Keep only the last 1000 measurements
            if len(self.metrics['memory_usage']) > 1000:
                self.metrics['memory_usage'] = self.metrics['memory_usage'][-1000:]
            
            # Log if memory usage is high
            if memory_usage > 500:  # More than 500 MB
                logger.warning(f"High memory usage: {memory_usage:.2f} MB")
        except Exception as e:
            logger.error(f"Error tracking memory usage: {e}")
    
    def track_response_time(self, endpoint, response_time):
        """
        Track API response time
        
        Args:
            endpoint (str): API endpoint
            response_time (float): Response time in seconds
        """
        if not self.enabled or not self.track_response_time:
            return
        
        try:
            self.metrics['response_times'].append({
                'timestamp': datetime.now(),
                'endpoint': endpoint,
                'value': response_time
            })
            
            # Keep only the last 1000 measurements
            if len(self.metrics['response_times']) > 1000:
                self.metrics['response_times'] = self.metrics['response_times'][-1000:]
            
            # Log if response time is slow
            if response_time > 1.0:  # More than 1 second
                logger.warning(f"Slow response time for {endpoint}: {response_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error tracking response time: {e}")
    
    def track_db_query(self, query, query_time):
        """
        Track database query time
        
        Args:
            query (str): SQL query
            query_time (float): Query time in seconds
        """
        if not self.enabled or not self.track_db_queries:
            return
        
        try:
            # Truncate long queries
            if len(query) > 100:
                query = query[:97] + "..."
            
            self.metrics['db_query_times'].append({
                'timestamp': datetime.now(),
                'query': query,
                'value': query_time
            })
            
            # Keep only the last 1000 measurements
            if len(self.metrics['db_query_times']) > 1000:
                self.metrics['db_query_times'] = self.metrics['db_query_times'][-1000:]
            
            # Log if query time is slow
            if query_time > 0.5:  # More than 0.5 seconds
                logger.warning(f"Slow database query: {query} - {query_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error tracking database query: {e}")
    
    def get_performance_report(self):
        """
        Get performance report
        
        Returns:
            dict: Performance metrics
        """
        if not self.enabled:
            return {"status": "Performance monitoring is disabled"}
        
        try:
            # Calculate memory usage statistics
            memory_values = [m['value'] for m in self.metrics['memory_usage']]
            avg_memory = sum(memory_values) / len(memory_values) if memory_values else 0
            max_memory = max(memory_values) if memory_values else 0
            
            # Calculate response time statistics
            response_values = [r['value'] for r in self.metrics['response_times']]
            avg_response = sum(response_values) / len(response_values) if response_values else 0
            max_response = max(response_values) if response_values else 0
            
            # Calculate database query time statistics
            query_values = [q['value'] for q in self.metrics['db_query_times']]
            avg_query = sum(query_values) / len(query_values) if query_values else 0
            max_query = max(query_values) if query_values else 0
            
            return {
                "memory_usage": {
                    "average": avg_memory,
                    "maximum": max_memory,
                    "current": memory_values[-1] if memory_values else 0,
                    "unit": "MB"
                },
                "response_time": {
                    "average": avg_response,
                    "maximum": max_response,
                    "unit": "seconds"
                },
                "db_query_time": {
                    "average": avg_query,
                    "maximum": max_query,
                    "unit": "seconds"
                }
            }
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"status": f"Error: {str(e)}"}

class BackupManager:
    """Backup manager for the application"""
    
    def __init__(self):
        """Initialize backup manager"""
        self.enabled = MONITORING_CONFIG['backup']['enabled']
        self.interval = MONITORING_CONFIG['backup']['interval']
        self.retention_days = MONITORING_CONFIG['backup']['retention_days']
        self.backup_database = MONITORING_CONFIG['backup']['backup_database']
        self.backup_user_data = MONITORING_CONFIG['backup']['backup_user_data']
        self.backup_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backups')
        os.makedirs(self.backup_dir, exist_ok=True)
    
    def create_database_backup(self):
        """Create database backup"""
        if not self.enabled or not self.backup_database:
            return
        
        try:
            # Get database path from environment or config
            db_path = os.getenv('DB_PATH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'crypto_trading.db'))
            
            # Only backup SQLite databases
            if not db_path.endswith('.db'):
                logger.info("Database backup skipped: not a SQLite database")
                return
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_filename = f"database_backup_{timestamp}.db"
            backup_path = os.path.join(self.backup_dir, backup_filename)
            
            # Create backup
            with sqlite3.connect(db_path) as conn:
                with sqlite3.connect(backup_path) as backup_conn:
                    conn.backup(backup_conn)
            
            logger.info(f"Database backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return None
    
    def create_user_data_backup(self):
        """Create user data backup"""
        if not self.enabled or not self.backup_user_data:
            return
        
        try:
            # Define directories to backup
            dirs_to_backup = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auth'),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
            ]
            
            # Create backup filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir = os.path.join(self.backup_dir, f"user_data_backup_{timestamp}")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Copy files
            import shutil
            for dir_path in dirs_to_backup:
                if os.path.exists(dir_path):
                    dir_name = os.path.basename(dir_path)
                    backup_path = os.path.join(backup_dir, dir_name)
                    shutil.copytree(dir_path, backup_path, dirs_exist_ok=True)
            
            logger.info(f"User data backup created: {backup_dir}")
            return backup_dir
        except Exception as e:
            logger.error(f"Error creating user data backup: {e}")
            return None
    
    def cleanup_old_backups(self):
        """Clean up old backups"""
        if not self.enabled:
            return
        
        try:
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            # Find and remove old backups
            for item in os.listdir(self.backup_dir):
                item_path = os.path.join(self.backup_dir, item)
                
                # Get item creation time
                created_time = datetime.fromtimestamp(os.path.getctime(item_path))
                
                # Remove if older than retention period
                if created_time < cutoff_date:
                    if os.path.isdir(item_path):
                        import shutil
                        shutil.rmtree(item_path)
                    else:
                        os.remove(item_path)
                    
                    logger.info(f"Removed old backup: {item_path}")
            
            logger.info("Backup cleanup completed")
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
    
    def start_backup_schedule(self):
        """Start backup schedule in a separate thread"""
        if not self.enabled:
            logger.info("Backup scheduling is disabled")
            return
        
        def run_backups():
            while True:
                try:
                    # Create backups
                    self.create_database_backup()
                    self.create_user_data_backup()
                    
                    # Clean up old backups
                    self.cleanup_old_backups()
                    
                    # Wait for next backup interval
                    time.sleep(self.interval)
                except Exception as e:
                    logger.error(f"Error in backup schedule: {e}")
                    time.sleep(3600)  # Wait an hour before retrying
        
        thread = threading.Thread(target=run_backups, daemon=True)
        thread.start()
        logger.info(f"Backup schedule started with interval {self.interval} seconds")

# Initialize monitoring components
health_check = HealthCheck()
performance_monitor = PerformanceMonitor()
backup_manager = BackupManager()

def start_monitoring():
    """Start all monitoring components"""
    try:
        # Start health check monitoring
        health_check.start_monitoring()
        
        # Start backup schedule
        backup_manager.start_backup_schedule()
        
        logger.info("Monitoring and maintenance systems started")
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")

# Start monitoring when module is imported
if __name__ != "__main__":
    start_monitoring()
