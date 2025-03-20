"""
Performance Analytics Module for Cryptocurrency Trading Bot

This module handles trade logging, performance metrics calculation,
and visualization of trading results.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sqlite3

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Configure logging
logger = logging.getLogger('crypto_trading_bot.analytics')

class TradeLogger:
    """Class for logging and storing trade information"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize trade logger
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'trades.db'
        )
        self._init_database()
        logger.info(f"Trade logger initialized with database at {self.db_path}")
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                entry_order_id TEXT,
                exit_order_id TEXT
            )
            ''')
            
            # Create daily performance table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                trades INTEGER NOT NULL,
                win_trades INTEGER NOT NULL,
                loss_trades INTEGER NOT NULL,
                profit REAL NOT NULL,
                loss REAL NOT NULL,
                win_rate REAL NOT NULL,
                profit_factor REAL NOT NULL,
                max_drawdown REAL NOT NULL
            )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def log_trade(self, trade_data: Dict) -> int:
        """
        Log trade to database
        
        Args:
            trade_data: Trade data dictionary
            
        Returns:
            Trade ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data
            columns = []
            values = []
            placeholders = []
            
            for key, value in trade_data.items():
                columns.append(key)
                values.append(value)
                placeholders.append('?')
            
            # Insert trade
            query = f"INSERT INTO trades ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            cursor.execute(query, values)
            
            # Get trade ID
            trade_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            logger.info(f"Logged trade {trade_id}: {trade_data['symbol']} {trade_data['side']}")
            
            return trade_id
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
            return -1
    
    def update_trade(self, trade_id: int, update_data: Dict) -> bool:
        """
        Update existing trade
        
        Args:
            trade_id: Trade ID
            update_data: Data to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare update statement
            set_clause = []
            values = []
            
            for key, value in update_data.items():
                set_clause.append(f"{key} = ?")
                values.append(value)
            
            # Add trade ID
            values.append(trade_id)
            
            # Update trade
            query = f"UPDATE trades SET {', '.join(set_clause)} WHERE id = ?"
            cursor.execute(query, values)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Updated trade {trade_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
            return False
    
    def get_trade(self, trade_id: int) -> Optional[Dict]:
        """
        Get trade by ID
        
        Args:
            trade_id: Trade ID
            
        Returns:
            Trade data dictionary or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get trade
            cursor.execute("SELECT * FROM trades WHERE id = ?", (trade_id,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return dict(row)
            else:
                return None
            
        except Exception as e:
            logger.error(f"Error getting trade: {e}")
            return None
    
    def get_trades(self, symbol: Optional[str] = None, 
                 strategy: Optional[str] = None,
                 status: Optional[str] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 limit: int = 100) -> List[Dict]:
        """
        Get trades with optional filtering
        
        Args:
            symbol: Filter by symbol
            strategy: Filter by strategy
            status: Filter by status
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of trades to return
            
        Returns:
            List of trade data dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if start_date:
                query += " AND entry_time >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND entry_time <= ?"
                params.append(end_date.isoformat())
            
            query += " ORDER BY entry_time DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    def log_daily_performance(self, performance_data: Dict) -> bool:
        """
        Log daily performance
        
        Args:
            performance_data: Performance data dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Prepare data
            columns = []
            values = []
            placeholders = []
            
            for key, value in performance_data.items():
                columns.append(key)
                values.append(value)
                placeholders.append('?')
            
            # Insert or replace performance
            query = f"INSERT OR REPLACE INTO daily_performance ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            cursor.execute(query, values)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Logged daily performance for {performance_data.get('date')}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging daily performance: {e}")
            return False
    
    def get_daily_performance(self, start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            limit: int = 30) -> List[Dict]:
        """
        Get daily performance
        
        Args:
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)
            limit: Maximum number of days to return
            
        Returns:
            List of daily performance dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Build query
            query = "SELECT * FROM daily_performance WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date DESC LIMIT ?"
            params.append(limit)
            
            # Execute query
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conn.close()
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting daily performance: {e}")
            return []


class PerformanceMetrics:
    """Class for calculating trading performance metrics"""
    
    @staticmethod
    def calculate_metrics(trades: List[Dict]) -> Dict:
        """
        Calculate performance metrics from trades
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'win_trades': 0,
                'loss_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'net_profit': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'largest_profit': 0.0,
                'largest_loss': 0.0,
                'avg_trade': 0.0,
                'avg_bars_win': 0,
                'avg_bars_loss': 0,
                'max_drawdown': 0.0,
                'max_drawdown_percent': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'profit_per_month': 0.0,
                'monthly_win_rate': 0.0
            }
        
        # Filter closed trades
        closed_trades = [t for t in trades if t.get('status') == 'closed' and t.get('pnl') is not None]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'win_trades': 0,
                'loss_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'net_profit': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'largest_profit': 0.0,
                'largest_loss': 0.0,
                'avg_trade': 0.0,
                'avg_bars_win': 0,
                'avg_bars_loss': 0,
                'max_drawdown': 0.0,
                'max_drawdown_percent': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'profit_per_month': 0.0,
                'monthly_win_rate': 0.0
            }
        
        # Basic metrics
        total_trades = len(closed_trades)
        win_trades = len([t for t in closed_trades if t.get('pnl', 0) > 0])
        loss_trades = len([t for t in closed_trades if t.get('pnl', 0) <= 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0.0
        
        # Profit/loss metrics
        total_profit = sum([t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) > 0])
        total_loss = abs(sum([t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) <= 0]))
        net_profit = total_profit - total_loss
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Average metrics
        avg_profit = total_profit / win_trades if win_trades > 0 else 0.0
        avg_loss = total_loss / loss_trades if loss_trades > 0 else 0.0
        avg_trade = net_profit / total_trades if total_trades > 0 else 0.0
        
        # Extreme metrics
        largest_profit = max([t.get('pnl', 0) for t in closed_trades]) if closed_trades else 0.0
        largest_loss = min([t.get('pnl', 0) for t in closed_trades]) if closed_trades else 0.0
        
        # Time metrics
        win_trade_durations = []
        loss_trade_durations = []
        
        for trade in closed_trades:
            entry_time = datetime.fromisoformat(trade.get('entry_time')) if isinstance(trade.get('entry_time'), str) else trade.get('entry_time')
            exit_time = datetime.fromisoformat(trade.get('exit_time')) if isinstance(trade.get('exit_time'), str) else trade.get('exit_time')
            
            if entry_time and exit_time:
                duration = (exit_time - entry_time).total_seconds() / 3600  # Duration in hours
                
                if trade.get('pnl', 0) > 0:
                    win_trade_durations.append(duration)
                else:
                    loss_trade_durations.append(duration)
        
        avg_bars_win = sum(win_trade_durations) / len(win_trade_durations) if win_trade_durations else 0
        avg_bars_loss = sum(loss_trade_durations) / len(loss_trade_durations) if loss_trade_durations else 0
        
        # Drawdown calculation
        equity_curve = PerformanceMetrics.calculate_equity_curve(closed_trades)
        max_drawdown, max_drawdown_percent = PerformanceMetrics.calculate_drawdown(equity_curve)
        
        # Risk-adjusted metrics
        returns = PerformanceMetrics.calculate_daily_returns(equity_curve)
        sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(returns)
        sortino_ratio = PerformanceMetrics.calculate_sortino_ratio(returns)
        
        # Monthly metrics
        monthly_stats = PerformanceMetrics.calculate_monthly_stats(closed_trades)
        profit_per_month = monthly_stats.get('avg_profit', 0.0)
        monthly_win_rate = monthly_stats.get('win_rate', 0.0)
        
        return {
            'total_trades': total_trades,
            'win_trades': win_trades,
            'loss_trades': loss_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_profit': net_profit,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'largest_profit': largest_profit,
            'largest_loss': largest_loss,
            'avg_trade': avg_trade,
            'avg_bars_win': avg_bars_win,
            'avg_bars_loss': avg_bars_loss,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_percent,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_per_month': profit_per_month,
            'monthly_win_rate': monthly_win_rate
        }
    
    @staticmethod
    def calculate_equity_curve(trades: List[Dict], initial_balance: float = 10000.0) -> pd.DataFrame:
        """
        Calculate equity curve from trades
        
        Args:
            trades: List of trade dictionaries
            initial_balance: Initial account balance
            
        Returns:
            DataFrame with equity curve
        """
        if not trades:
            return pd.DataFrame(columns=['date', 'equity'])
        
        # Sort trades by entry time
        sorted_trades = sorted(trades, key=lambda t: t.get('entry_time'))
        
        # Create equity curve
        equity_points = []
        balance = initial_balance
        
        # Add initial point
        first_trade_time = datetime.fromisoformat(sorted_trades[0].get('entry_time')) if isinstance(sorted_trades[0].get('entry_time'), str) else sorted_trades[0].get('entry_time')
        equity_points.append({
            'date': first_trade_time - timedelta(days=1),
            'equity': balance
        })
        
        # Add points for each trade
        for trade in sorted_trades:
            if trade.get('status') == 'closed' and trade.get('pnl') is not None:
                exit_time = datetime.fromisoformat(trade.get('exit_time')) if isinstance(trade.get('exit_time'), str) else trade.get('exit_time')
                balance += trade.get('pnl', 0)
                
                equity_points.append({
                    'date': exit_time,
                    'equity': balance
                })
        
        # Create DataFrame
        equity_df = pd.DataFrame(equity_points)
        
        # Ensure date is datetime
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        
        # Set date as index
        equity_df = equity_df.set_index('date')
        
        # Resample to daily frequency
        equity_df = equity_df.resample('D').last().fillna(method='ffill').reset_index()
        
        return equity_df
    
    @staticmethod
    def calculate_drawdown(equity_curve: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate maximum drawdown from equity curve
        
        Args:
            equity_curve: DataFrame with equity curve
            
        Returns:
            Tuple of (max_drawdown, max_drawdown_percent)
        """
        if equity_curve.empty:
            return 0.0, 0.0
        
        # Calculate drawdown
        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = equity_curve['peak'] - equity_curve['equity']
        equity_curve['drawdown_percent'] = (equity_curve['drawdown'] / equity_curve['peak']) * 100
        
        # Get maximum drawdown
        max_drawdown = equity_curve['drawdown'].max()
        max_drawdown_percent = equity_curve['drawdown_percent'].max()
        
        return max_drawdown, max_drawdown_percent
    
    @staticmethod
    def calculate_daily_returns(equity_curve: pd.DataFrame) -> pd.Series:
        """
        Calculate daily returns from equity curve
        
        Args:
            equity_curve: DataFrame with equity curve
            
        Returns:
            Series of daily returns
        """
        if equity_curve.empty:
            return pd.Series()
        
        # Calculate daily returns
        equity_curve['return'] = equity_curve['equity'].pct_change()
        
        return equity_curve['return'].dropna()
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        if returns.empty:
            return 0.0
        
        # Calculate Sharpe ratio
        excess_returns = returns - risk_free_rate
        sharpe_ratio = excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0
        
        # Annualize (assuming daily returns)
        sharpe_ratio *= np.sqrt(252)
        
        return sharpe_ratio
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sortino ratio
        """
        if returns.empty:
            return 0.0
        
        # Calculate Sortino ratio
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std() if not downside_returns.empty else 0.0
        
        sortino_ratio = excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0.0
        
        # Annualize (assuming daily returns)
        sortino_ratio *= np.sqrt(252)
        
        return sortino_ratio
    
    @staticmethod
    def calculate_monthly_stats(trades: List[Dict]) -> Dict:
        """
        Calculate monthly statistics
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            Dictionary of monthly statistics
        """
        if not trades:
            return {
                'months': 0,
                'profitable_months': 0,
                'losing_months': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'best_month': 0.0,
                'worst_month': 0.0
            }
        
        # Filter closed trades
        closed_trades = [t for t in trades if t.get('status') == 'closed' and t.get('pnl') is not None]
        
        if not closed_trades:
            return {
                'months': 0,
                'profitable_months': 0,
                'losing_months': 0,
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'avg_loss': 0.0,
                'best_month': 0.0,
                'worst_month': 0.0
            }
        
        # Group trades by month
        monthly_pnl = {}
        
        for trade in closed_trades:
            exit_time = datetime.fromisoformat(trade.get('exit_time')) if isinstance(trade.get('exit_time'), str) else trade.get('exit_time')
            
            if exit_time:
                month_key = exit_time.strftime('%Y-%m')
                
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0.0
                
                monthly_pnl[month_key] += trade.get('pnl', 0)
        
        # Calculate statistics
        months = len(monthly_pnl)
        profitable_months = len([pnl for pnl in monthly_pnl.values() if pnl > 0])
        losing_months = len([pnl for pnl in monthly_pnl.values() if pnl <= 0])
        win_rate = profitable_months / months if months > 0 else 0.0
        
        profitable_pnl = [pnl for pnl in monthly_pnl.values() if pnl > 0]
        losing_pnl = [pnl for pnl in monthly_pnl.values() if pnl <= 0]
        
        avg_profit = sum(profitable_pnl) / len(profitable_pnl) if profitable_pnl else 0.0
        avg_loss = sum(losing_pnl) / len(losing_pnl) if losing_pnl else 0.0
        
        best_month = max(monthly_pnl.values()) if monthly_pnl else 0.0
        worst_month = min(monthly_pnl.values()) if monthly_pnl else 0.0
        
        return {
            'months': months,
            'profitable_months': profitable_months,
            'losing_months': losing_months,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'best_month': best_month,
            'worst_month': worst_month
        }


class PerformanceVisualizer:
    """Class for visualizing trading performance"""
    
    @staticmethod
    def plot_equity_curve(equity_curve: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot equity curve
        
        Args:
            equity_curve: DataFrame with equity curve
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if equity_curve.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot equity curve
        ax.plot(equity_curve['date'], equity_curve['equity'], label='Equity', color='blue')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity')
        ax.set_title('Equity Curve')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend()
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    @staticmethod
    def plot_drawdown(equity_curve: pd.DataFrame, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot drawdown
        
        Args:
            equity_curve: DataFrame with equity curve
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if equity_curve.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Calculate drawdown
        equity_curve['peak'] = equity_curve['equity'].cummax()
        equity_curve['drawdown'] = equity_curve['peak'] - equity_curve['equity']
        equity_curve['drawdown_percent'] = (equity_curve['drawdown'] / equity_curve['peak']) * 100
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot drawdown
        ax.fill_between(equity_curve['date'], 0, equity_curve['drawdown_percent'], color='red', alpha=0.3)
        ax.plot(equity_curve['date'], equity_curve['drawdown_percent'], color='red', label='Drawdown')
        
        # Add labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        # Invert y-axis
        ax.invert_yaxis()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add legend
        ax.legend()
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    @staticmethod
    def plot_monthly_returns(trades: List[Dict], save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot monthly returns
        
        Args:
            trades: List of trade dictionaries
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Filter closed trades
        closed_trades = [t for t in trades if t.get('status') == 'closed' and t.get('pnl') is not None]
        
        if not closed_trades:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Group trades by month
        monthly_pnl = {}
        
        for trade in closed_trades:
            exit_time = datetime.fromisoformat(trade.get('exit_time')) if isinstance(trade.get('exit_time'), str) else trade.get('exit_time')
            
            if exit_time:
                month_key = exit_time.strftime('%Y-%m')
                
                if month_key not in monthly_pnl:
                    monthly_pnl[month_key] = 0.0
                
                monthly_pnl[month_key] += trade.get('pnl', 0)
        
        # Create DataFrame
        monthly_df = pd.DataFrame({
            'month': list(monthly_pnl.keys()),
            'pnl': list(monthly_pnl.values())
        })
        
        # Sort by month
        monthly_df = monthly_df.sort_values('month')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot monthly returns
        colors = ['green' if pnl >= 0 else 'red' for pnl in monthly_df['pnl']]
        ax.bar(monthly_df['month'], monthly_df['pnl'], color=colors)
        
        # Add labels and title
        ax.set_xlabel('Month')
        ax.set_ylabel('Profit/Loss')
        ax.set_title('Monthly Returns')
        
        # Format x-axis
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    @staticmethod
    def plot_trade_distribution(trades: List[Dict], save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trade profit/loss distribution
        
        Args:
            trades: List of trade dictionaries
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Filter closed trades
        closed_trades = [t for t in trades if t.get('status') == 'closed' and t.get('pnl') is not None]
        
        if not closed_trades:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Extract PnL values
        pnl_values = [t.get('pnl', 0) for t in closed_trades]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot distribution
        sns.histplot(pnl_values, kde=True, ax=ax)
        
        # Add labels and title
        ax.set_xlabel('Profit/Loss')
        ax.set_ylabel('Frequency')
        ax.set_title('Trade Profit/Loss Distribution')
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    @staticmethod
    def plot_win_loss_ratio(trades: List[Dict], save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot win/loss ratio
        
        Args:
            trades: List of trade dictionaries
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Filter closed trades
        closed_trades = [t for t in trades if t.get('status') == 'closed' and t.get('pnl') is not None]
        
        if not closed_trades:
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Count wins and losses
        win_trades = len([t for t in closed_trades if t.get('pnl', 0) > 0])
        loss_trades = len([t for t in closed_trades if t.get('pnl', 0) <= 0])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot pie chart
        ax.pie(
            [win_trades, loss_trades],
            labels=['Win', 'Loss'],
            colors=['green', 'red'],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.1, 0)
        )
        
        # Add title
        ax.set_title('Win/Loss Ratio')
        
        # Equal aspect ratio
        ax.axis('equal')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig
    
    @staticmethod
    def plot_strategy_comparison(strategy_metrics: Dict[str, Dict], save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot strategy comparison
        
        Args:
            strategy_metrics: Dictionary mapping strategy names to metrics
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        if not strategy_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No data available", ha='center', va='center')
            return fig
        
        # Extract metrics
        strategies = list(strategy_metrics.keys())
        net_profits = [metrics.get('net_profit', 0) for metrics in strategy_metrics.values()]
        win_rates = [metrics.get('win_rate', 0) * 100 for metrics in strategy_metrics.values()]
        profit_factors = [metrics.get('profit_factor', 0) for metrics in strategy_metrics.values()]
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot net profit
        axes[0].bar(strategies, net_profits, color='blue')
        axes[0].set_title('Net Profit')
        axes[0].set_ylabel('Profit')
        axes[0].grid(True, alpha=0.3)
        
        # Plot win rate
        axes[1].bar(strategies, win_rates, color='green')
        axes[1].set_title('Win Rate')
        axes[1].set_ylabel('Win Rate (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot profit factor
        axes[2].bar(strategies, profit_factors, color='purple')
        axes[2].set_title('Profit Factor')
        axes[2].set_ylabel('Profit Factor')
        axes[2].grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        for ax in axes:
            plt.sca(ax)
            plt.xticks(rotation=45)
        
        # Tight layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path)
        
        return fig


class PerformanceAnalyzer:
    """Main class for analyzing trading performance"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize performance analyzer
        
        Args:
            db_path: Path to SQLite database file
        """
        self.trade_logger = TradeLogger(db_path)
        logger.info("Performance analyzer initialized")
    
    def log_trade(self, trade_data: Dict) -> int:
        """
        Log trade
        
        Args:
            trade_data: Trade data dictionary
            
        Returns:
            Trade ID
        """
        return self.trade_logger.log_trade(trade_data)
    
    def update_trade(self, trade_id: int, update_data: Dict) -> bool:
        """
        Update trade
        
        Args:
            trade_id: Trade ID
            update_data: Data to update
            
        Returns:
            True if successful, False otherwise
        """
        return self.trade_logger.update_trade(trade_id, update_data)
    
    def get_trade(self, trade_id: int) -> Optional[Dict]:
        """
        Get trade by ID
        
        Args:
            trade_id: Trade ID
            
        Returns:
            Trade data dictionary or None if not found
        """
        return self.trade_logger.get_trade(trade_id)
    
    def get_trades(self, symbol: Optional[str] = None, 
                 strategy: Optional[str] = None,
                 status: Optional[str] = None,
                 start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None,
                 limit: int = 100) -> List[Dict]:
        """
        Get trades with optional filtering
        
        Args:
            symbol: Filter by symbol
            strategy: Filter by strategy
            status: Filter by status
            start_date: Filter by start date
            end_date: Filter by end date
            limit: Maximum number of trades to return
            
        Returns:
            List of trade data dictionaries
        """
        return self.trade_logger.get_trades(
            symbol=symbol,
            strategy=strategy,
            status=status,
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
    
    def calculate_metrics(self, symbol: Optional[str] = None, 
                        strategy: Optional[str] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> Dict:
        """
        Calculate performance metrics
        
        Args:
            symbol: Filter by symbol
            strategy: Filter by strategy
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            Dictionary of performance metrics
        """
        trades = self.get_trades(
            symbol=symbol,
            strategy=strategy,
            status='closed',
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        return PerformanceMetrics.calculate_metrics(trades)
    
    def calculate_equity_curve(self, symbol: Optional[str] = None, 
                             strategy: Optional[str] = None,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None,
                             initial_balance: float = 10000.0) -> pd.DataFrame:
        """
        Calculate equity curve
        
        Args:
            symbol: Filter by symbol
            strategy: Filter by strategy
            start_date: Filter by start date
            end_date: Filter by end date
            initial_balance: Initial account balance
            
        Returns:
            DataFrame with equity curve
        """
        trades = self.get_trades(
            symbol=symbol,
            strategy=strategy,
            status='closed',
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        return PerformanceMetrics.calculate_equity_curve(trades, initial_balance)
    
    def compare_strategies(self, strategies: List[str], 
                         symbol: Optional[str] = None,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None) -> Dict[str, Dict]:
        """
        Compare multiple strategies
        
        Args:
            strategies: List of strategy names
            symbol: Filter by symbol
            start_date: Filter by start date
            end_date: Filter by end date
            
        Returns:
            Dictionary mapping strategy names to metrics
        """
        results = {}
        
        for strategy in strategies:
            metrics = self.calculate_metrics(
                symbol=symbol,
                strategy=strategy,
                start_date=start_date,
                end_date=end_date
            )
            
            results[strategy] = metrics
        
        return results
    
    def generate_report(self, symbol: Optional[str] = None, 
                      strategy: Optional[str] = None,
                      start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None,
                      output_dir: str = None) -> Dict:
        """
        Generate performance report
        
        Args:
            symbol: Filter by symbol
            strategy: Filter by strategy
            start_date: Filter by start date
            end_date: Filter by end date
            output_dir: Directory to save report files
            
        Returns:
            Dictionary with report data and file paths
        """
        # Create output directory if not exists
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Get trades
        trades = self.get_trades(
            symbol=symbol,
            strategy=strategy,
            status='closed',
            start_date=start_date,
            end_date=end_date,
            limit=1000
        )
        
        # Calculate metrics
        metrics = PerformanceMetrics.calculate_metrics(trades)
        
        # Calculate equity curve
        equity_curve = PerformanceMetrics.calculate_equity_curve(trades)
        
        # Generate plots
        plots = {}
        
        if output_dir:
            # Equity curve
            equity_plot_path = os.path.join(output_dir, 'equity_curve.png')
            PerformanceVisualizer.plot_equity_curve(equity_curve, equity_plot_path)
            plots['equity_curve'] = equity_plot_path
            
            # Drawdown
            drawdown_plot_path = os.path.join(output_dir, 'drawdown.png')
            PerformanceVisualizer.plot_drawdown(equity_curve, drawdown_plot_path)
            plots['drawdown'] = drawdown_plot_path
            
            # Monthly returns
            monthly_plot_path = os.path.join(output_dir, 'monthly_returns.png')
            PerformanceVisualizer.plot_monthly_returns(trades, monthly_plot_path)
            plots['monthly_returns'] = monthly_plot_path
            
            # Trade distribution
            distribution_plot_path = os.path.join(output_dir, 'trade_distribution.png')
            PerformanceVisualizer.plot_trade_distribution(trades, distribution_plot_path)
            plots['trade_distribution'] = distribution_plot_path
            
            # Win/loss ratio
            win_loss_plot_path = os.path.join(output_dir, 'win_loss_ratio.png')
            PerformanceVisualizer.plot_win_loss_ratio(trades, win_loss_plot_path)
            plots['win_loss_ratio'] = win_loss_plot_path
        
        # Create report data
        report = {
            'metrics': metrics,
            'trades': [t for t in trades],
            'plots': plots
        }
        
        # Save report as JSON
        if output_dir:
            report_path = os.path.join(output_dir, 'report.json')
            
            with open(report_path, 'w') as f:
                json.dump({
                    'metrics': metrics,
                    'plots': plots
                }, f, indent=2)
            
            report['report_path'] = report_path
        
        return report


# Factory function to create performance analyzer
def create_performance_analyzer(db_path: str = None) -> PerformanceAnalyzer:
    """
    Create a performance analyzer
    
    Args:
        db_path: Path to SQLite database file
        
    Returns:
        PerformanceAnalyzer instance
    """
    return PerformanceAnalyzer(db_path)


# Test function to verify the module works correctly
def test_performance_analyzer():
    """Test performance analyzer functionality"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create performance analyzer
    analyzer = create_performance_analyzer()
    
    try:
        # Generate sample trades
        sample_trades = []
        
        # Add winning trades
        for i in range(10):
            entry_time = datetime.now() - timedelta(days=30) + timedelta(days=i)
            exit_time = entry_time + timedelta(hours=6)
            
            trade = {
                'symbol': 'BTC/USDT',
                'strategy': 'RSI_MACD',
                'side': 'buy',
                'entry_time': entry_time,
                'entry_price': 50000.0,
                'amount': 0.1,
                'exit_time': exit_time,
                'exit_price': 51000.0,
                'pnl': 100.0,
                'pnl_percent': 2.0,
                'status': 'closed',
                'stop_loss': 49000.0,
                'take_profit': 52000.0
            }
            
            trade_id = analyzer.log_trade(trade)
            sample_trades.append(trade_id)
        
        # Add losing trades
        for i in range(5):
            entry_time = datetime.now() - timedelta(days=15) + timedelta(days=i)
            exit_time = entry_time + timedelta(hours=4)
            
            trade = {
                'symbol': 'BTC/USDT',
                'strategy': 'BollingerBands',
                'side': 'buy',
                'entry_time': entry_time,
                'entry_price': 50000.0,
                'amount': 0.1,
                'exit_time': exit_time,
                'exit_price': 49500.0,
                'pnl': -50.0,
                'pnl_percent': -1.0,
                'status': 'closed',
                'stop_loss': 49000.0,
                'take_profit': 51000.0
            }
            
            trade_id = analyzer.log_trade(trade)
            sample_trades.append(trade_id)
        
        # Calculate metrics
        metrics = analyzer.calculate_metrics()
        logger.info(f"Performance metrics: {metrics}")
        
        # Calculate equity curve
        equity_curve = analyzer.calculate_equity_curve()
        logger.info(f"Equity curve: {len(equity_curve)} points")
        
        # Compare strategies
        strategy_comparison = analyzer.compare_strategies(['RSI_MACD', 'BollingerBands'])
        logger.info(f"Strategy comparison: {strategy_comparison}")
        
        # Generate report
        report = analyzer.generate_report(output_dir='performance_report')
        logger.info(f"Report generated: {report.get('report_path')}")
        
    except Exception as e:
        logger.error(f"Error testing performance analyzer: {e}")
    finally:
        logger.info("Test completed")


if __name__ == "__main__":
    # Run test
    test_performance_analyzer()
