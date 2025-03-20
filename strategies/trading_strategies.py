"""
Trading Strategy Module for Cryptocurrency Trading Bot

This module implements various trading strategies using technical indicators
to generate buy/sell signals and manage trading positions.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

# Import configuration and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from strategies.technical_indicators import TechnicalAnalyzer, create_technical_analyzer

logger = logging.getLogger('crypto_trading_bot.strategies')

class TradingStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, description: str, params: Dict = None):
        """
        Initialize trading strategy
        
        Args:
            name: Strategy name
            description: Strategy description
            params: Strategy parameters
        """
        self.name = name
        self.description = description
        self.params = params or {}
        self.analyzer = create_technical_analyzer()
        logger.info(f"Initialized {name} strategy")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with trading signals added
        """
        pass
    
    def calculate_position_size(self, signal: int, price: float, balance: float) -> float:
        """
        Calculate position size based on risk management rules
        
        Args:
            signal: Trading signal (1 for buy, -1 for sell, 0 for neutral)
            price: Current price
            balance: Available balance
            
        Returns:
            Position size in base currency
        """
        if signal == 0:
            return 0.0
        
        # Get risk management parameters
        risk_params = config.TRADING_PARAMS['risk_management']
        max_position_size_percent = risk_params.get('max_position_size_percent', 5.0)
        
        # Calculate position size
        max_position_size = balance * (max_position_size_percent / 100.0)
        
        # Convert to base currency units
        position_size = max_position_size / price
        
        return position_size
    
    def calculate_stop_loss(self, signal: int, price: float) -> float:
        """
        Calculate stop loss price
        
        Args:
            signal: Trading signal (1 for buy, -1 for sell, 0 for neutral)
            price: Current price
            
        Returns:
            Stop loss price
        """
        if signal == 0:
            return 0.0
        
        # Get risk management parameters
        risk_params = config.TRADING_PARAMS['risk_management']
        stop_loss_percent = risk_params.get('default_stop_loss_percent', 2.0)
        
        # Calculate stop loss price
        if signal > 0:  # Buy signal
            stop_loss = price * (1 - stop_loss_percent / 100.0)
        else:  # Sell signal
            stop_loss = price * (1 + stop_loss_percent / 100.0)
        
        return stop_loss
    
    def calculate_take_profit(self, signal: int, price: float) -> float:
        """
        Calculate take profit price
        
        Args:
            signal: Trading signal (1 for buy, -1 for sell, 0 for neutral)
            price: Current price
            
        Returns:
            Take profit price
        """
        if signal == 0:
            return 0.0
        
        # Get risk management parameters
        risk_params = config.TRADING_PARAMS['risk_management']
        take_profit_percent = risk_params.get('default_take_profit_percent', 5.0)
        
        # Calculate take profit price
        if signal > 0:  # Buy signal
            take_profit = price * (1 + take_profit_percent / 100.0)
        else:  # Sell signal
            take_profit = price * (1 - take_profit_percent / 100.0)
        
        return take_profit
    
    def backtest(self, data: pd.DataFrame, initial_balance: float = 10000.0) -> Dict:
        """
        Backtest strategy on historical data
        
        Args:
            data: DataFrame containing OHLCV data
            initial_balance: Initial balance for backtesting
            
        Returns:
            Dictionary containing backtest results
        """
        # Generate signals
        signals_df = self.generate_signals(data)
        
        # Initialize backtest variables
        balance = initial_balance
        position = 0.0
        entry_price = 0.0
        trades = []
        equity_curve = []
        
        # Iterate through data
        for i in range(1, len(signals_df)):
            timestamp = signals_df.index[i]
            signal = signals_df['signal'].iloc[i]
            price = signals_df['close'].iloc[i]
            
            # Record equity
            equity = balance
            if position != 0:
                equity += position * price
            equity_curve.append({'timestamp': timestamp, 'equity': equity})
            
            # Check for trade signals
            if signal == 1 and position == 0:  # Buy signal
                # Calculate position size
                position_size = self.calculate_position_size(signal, price, balance)
                
                # Execute buy
                position = position_size
                entry_price = price
                cost = position * price
                balance -= cost
                
                # Calculate stop loss and take profit
                stop_loss = self.calculate_stop_loss(signal, price)
                take_profit = self.calculate_take_profit(signal, price)
                
                # Record trade
                trades.append({
                    'timestamp': timestamp,
                    'type': 'buy',
                    'price': price,
                    'amount': position,
                    'cost': cost,
                    'balance': balance,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
                
            elif signal == -1 and position > 0:  # Sell signal
                # Execute sell
                cost = position * price
                balance += cost
                pnl = cost - (position * entry_price)
                pnl_percent = (price / entry_price - 1) * 100
                
                # Record trade
                trades.append({
                    'timestamp': timestamp,
                    'type': 'sell',
                    'price': price,
                    'amount': position,
                    'cost': cost,
                    'balance': balance,
                    'pnl': pnl,
                    'pnl_percent': pnl_percent
                })
                
                # Reset position
                position = 0.0
                entry_price = 0.0
        
        # Calculate backtest metrics
        total_trades = len([t for t in trades if t['type'] == 'buy'])
        winning_trades = len([t for t in trades if t['type'] == 'sell' and t.get('pnl', 0) > 0])
        losing_trades = len([t for t in trades if t['type'] == 'sell' and t.get('pnl', 0) <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit/loss
        final_balance = balance
        if position > 0:
            final_balance += position * signals_df['close'].iloc[-1]
        
        profit_loss = final_balance - initial_balance
        profit_loss_percent = (profit_loss / initial_balance) * 100
        
        # Calculate drawdown
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df['drawdown'] = equity_df['equity'].cummax() - equity_df['equity']
            equity_df['drawdown_percent'] = (equity_df['drawdown'] / equity_df['equity'].cummax()) * 100
            max_drawdown = equity_df['drawdown'].max()
            max_drawdown_percent = equity_df['drawdown_percent'].max()
        else:
            max_drawdown = 0
            max_drawdown_percent = 0
        
        # Return backtest results
        return {
            'initial_balance': initial_balance,
            'final_balance': final_balance,
            'profit_loss': profit_loss,
            'profit_loss_percent': profit_loss_percent,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'max_drawdown_percent': max_drawdown_percent,
            'trades': trades,
            'equity_curve': equity_curve
        }


class RSIMACDStrategy(TradingStrategy):
    """RSI + MACD combined strategy"""
    
    def __init__(self, rsi_period: int = 14, rsi_overbought: float = 70, rsi_oversold: float = 30,
                macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
        """
        Initialize RSI + MACD strategy
        
        Args:
            rsi_period: RSI period
            rsi_overbought: RSI overbought threshold
            rsi_oversold: RSI oversold threshold
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
        """
        super().__init__(
            name="RSI_MACD",
            description="Combined RSI and MACD strategy",
            params={
                'rsi_period': rsi_period,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'macd_fast': macd_fast,
                'macd_slow': macd_slow,
                'macd_signal': macd_signal
            }
        )
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI and MACD
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with trading signals added
        """
        # Calculate indicators
        indicators_data = self.analyzer.calculate_indicators(data, ['rsi', 'macd'])
        
        # Initialize signals column
        indicators_data['signal'] = 0
        
        # Generate buy signals: RSI < oversold AND MACD line > signal line
        buy_condition = (
            (indicators_data[f'rsi_{self.rsi_period}'] < self.rsi_oversold) & 
            (indicators_data['macd_line'] > indicators_data['macd_signal'])
        )
        indicators_data.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals: RSI > overbought OR MACD line < signal line
        sell_condition = (
            (indicators_data[f'rsi_{self.rsi_period}'] > self.rsi_overbought) | 
            (indicators_data['macd_line'] < indicators_data['macd_signal'])
        )
        indicators_data.loc[sell_condition, 'signal'] = -1
        
        return indicators_data


class BollingerBandsStrategy(TradingStrategy):
    """Bollinger Bands strategy"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands strategy
        
        Args:
            period: Bollinger Bands period
            std_dev: Standard deviation multiplier
        """
        super().__init__(
            name="BollingerBands",
            description="Bollinger Bands strategy",
            params={
                'period': period,
                'std_dev': std_dev
            }
        )
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with trading signals added
        """
        # Calculate indicators
        indicators_data = self.analyzer.calculate_indicators(data, ['bollinger_bands'])
        
        # Initialize signals column
        indicators_data['signal'] = 0
        
        # Generate buy signals: Price touches or crosses below lower band
        buy_condition = (indicators_data['close'] <= indicators_data['bb_lower'])
        indicators_data.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals: Price touches or crosses above upper band
        sell_condition = (indicators_data['close'] >= indicators_data['bb_upper'])
        indicators_data.loc[sell_condition, 'signal'] = -1
        
        return indicators_data


class MACrossoverStrategy(TradingStrategy):
    """Moving Average Crossover strategy"""
    
    def __init__(self, fast_period: int = 9, slow_period: int = 21):
        """
        Initialize Moving Average Crossover strategy
        
        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period
        """
        super().__init__(
            name="MACrossover",
            description="Moving Average Crossover strategy",
            params={
                'fast_period': fast_period,
                'slow_period': slow_period
            }
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MA crossover
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with trading signals added
        """
        # Calculate indicators
        indicators_data = self.analyzer.calculate_indicators(data, ['ema'])
        
        # Initialize signals column
        indicators_data['signal'] = 0
        
        # Generate buy signals: Fast MA crosses above slow MA
        buy_condition = (
            (indicators_data[f'ema_{self.fast_period}'] > indicators_data[f'ema_{self.slow_period}']) & 
            (indicators_data[f'ema_{self.fast_period}'].shift(1) <= indicators_data[f'ema_{self.slow_period}'].shift(1))
        )
        indicators_data.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals: Fast MA crosses below slow MA
        sell_condition = (
            (indicators_data[f'ema_{self.fast_period}'] < indicators_data[f'ema_{self.slow_period}']) & 
            (indicators_data[f'ema_{self.fast_period}'].shift(1) >= indicators_data[f'ema_{self.slow_period}'].shift(1))
        )
        indicators_data.loc[sell_condition, 'signal'] = -1
        
        return indicators_data


class VolumeBreakoutStrategy(TradingStrategy):
    """Volume Breakout strategy"""
    
    def __init__(self, volume_threshold: float = 2.0, price_period: int = 20):
        """
        Initialize Volume Breakout strategy
        
        Args:
            volume_threshold: Volume threshold multiplier
            price_period: Price breakout period
        """
        super().__init__(
            name="VolumeBreakout",
            description="Volume Breakout strategy",
            params={
                'volume_threshold': volume_threshold,
                'price_period': price_period
            }
        )
        self.volume_threshold = volume_threshold
        self.price_period = price_period
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on volume breakouts
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with trading signals added
        """
        # Calculate average volume
        data['avg_volume'] = data['volume'].rolling(window=self.price_period).mean()
        
        # Calculate price highs and lows
        data['price_high'] = data['close'].rolling(window=self.price_period).max()
        data['price_low'] = data['close'].rolling(window=self.price_period).min()
        
        # Initialize signals column
        data['signal'] = 0
        
        # Generate buy signals: Volume spike + price breakout above recent high
        buy_condition = (
            (data['volume'] > data['avg_volume'] * self.volume_threshold) & 
            (data['close'] > data['price_high'].shift(1))
        )
        data.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals: Volume spike + price breakdown below recent low
        sell_condition = (
            (data['volume'] > data['avg_volume'] * self.volume_threshold) & 
            (data['close'] < data['price_low'].shift(1))
        )
        data.loc[sell_condition, 'signal'] = -1
        
        return data


class CombinedStrategy(TradingStrategy):
    """Combined strategy using multiple sub-strategies"""
    
    def __init__(self, strategies: List[TradingStrategy], weights: List[float] = None):
        """
        Initialize Combined strategy
        
        Args:
            strategies: List of strategies to combine
            weights: List of weights for each strategy (must sum to 1)
        """
        if not strategies:
            raise ValueError("At least one strategy must be provided")
        
        # Validate weights
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)
        elif len(weights) != len(strategies):
            raise ValueError("Number of weights must match number of strategies")
        elif abs(sum(weights) - 1.0) > 0.0001:
            raise ValueError("Weights must sum to 1")
        
        strategy_names = [s.name for s in strategies]
        super().__init__(
            name="Combined",
            description=f"Combined strategy using {', '.join(strategy_names)}",
            params={
                'strategies': strategy_names,
                'weights': weights
            }
        )
        self.strategies = strategies
        self.weights = weights
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals by combining multiple strategies
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with trading signals added
        """
        # Initialize result with data
        result = data.copy()
        
        # Generate signals for each strategy
        for i, strategy in enumerate(self.strategies):
            strategy_data = strategy.generate_signals(data)
            
            # Add weighted signal
            if i == 0:
                result['signal'] = strategy_data['signal'] * self.weights[i]
            else:
                result['signal'] += strategy_data['signal'] * self.weights[i]
        
        # Threshold signals
        result['signal'] = result['signal'].apply(lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0))
        
        return result


class StrategyManager:
    """Manager class for trading strategies"""
    
    def __init__(self):
        """Initialize strategy manager"""
        self.strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register default strategies"""
        # Register RSI + MACD strategy
        self.register_strategy(RSIMACDStrategy())
        
        # Register Bollinger Bands strategy
        self.register_strategy(BollingerBandsStrategy())
        
        # Register MA Crossover strategy
        self.register_strategy(MACrossoverStrategy())
        
        # Register Volume Breakout strategy
        self.register_strategy(VolumeBreakoutStrategy())
        
        # Register Combined strategy
        combined_strategy = CombinedStrategy([
            RSIMACDStrategy(),
            BollingerBandsStrategy()
        ], [0.6, 0.4])
        self.register_strategy(combined_strategy)
    
    def register_strategy(self, strategy: TradingStrategy):
        """
        Register a trading strategy
        
        Args:
            strategy: Trading strategy instance
        """
        self.strategies[strategy.name] = strategy
        logger.info(f"Registered {strategy.name} strategy")
    
    def get_strategy(self, name: str) -> Optional[TradingStrategy]:
        """
        Get a strategy by name
        
        Args:
            name: Strategy name
            
        Returns:
            Trading strategy instance or None if not found
        """
        return self.strategies.get(name)
    
    def get_all_strategies(self) -> Dict[str, TradingStrategy]:
        """
        Get all registered strategies
        
        Returns:
            Dictionary mapping strategy names to instances
        """
        return self.strategies
    
    def create_combined_strategy(self, strategy_names: List[str], 
                               weights: List[float] = None) -> Optional[TradingStrategy]:
        """
        Create a combined strategy from multiple strategies
        
        Args:
            strategy_names: List of strategy names to combine
            weights: List of weights for each strategy
            
        Returns:
            Combined strategy instance or None if any strategy not found
        """
        strategies = []
        
        for name in strategy_names:
            strategy = self.get_strategy(name)
            if strategy is None:
                logger.error(f"Strategy {name} not found")
                return None
            strategies.append(strategy)
        
        return CombinedStrategy(strategies, weights)


# Factory function to create strategy manager
def create_strategy_manager() -> StrategyManager:
    """
    Create a strategy manager
    
    Returns:
        StrategyManager instance
    """
    return StrategyManager()


# Test function to verify the module works correctly
def test_trading_strategies():
    """Test trading strategies functionality"""
    import matplotlib.pyplot as plt
    from data.data_manager import create_data_manager
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create data manager and fetch data
    data_manager = create_data_manager()
    symbol = 'BTC/USDT'
    timeframe = '1h'
    
    try:
        # Fetch historical data
        print(f"Fetching historical data for {symbol}...")
        data = data_manager.fetch_historical_data(symbol, timeframe, days_back=30)
        
        if data.empty:
            print("No data available. Using sample data for testing.")
            # Create sample data
            dates = pd.date_range(start='2023-01-01', periods=500, freq='H')
            data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.normal(100, 5, 500),
                'high': np.random.normal(105, 5, 500),
                'low': np.random.normal(95, 5, 500),
                'close': np.random.normal(100, 5, 500),
                'volume': np.random.normal(1000, 200, 500)
            })
            
            # Ensure high >= open, close and low <= open, close
            for i in range(len(data)):
                data.loc[i, 'high'] = max(data.loc[i, 'high'], data.loc[i, 'open'], data.loc[i, 'close'])
                data.loc[i, 'low'] = min(data.loc[i, 'low'], data.loc[i, 'open'], data.loc[i, 'close'])
        
        # Set timestamp as index
        data = data.set_index('timestamp')
        
        # Create strategy manager
        strategy_manager = create_strategy_manager()
        
        # Test each strategy
        for name, strategy in strategy_manager.get_all_strategies().items():
            print(f"\nTesting {name} strategy...")
            
            # Generate signals
            signals_df = strategy.generate_signals(data)
            
            # Run backtest
            backtest_results = strategy.backtest(data)
            
            # Print backtest results
            print(f"Initial balance: ${backtest_results['initial_balance']:.2f}")
            print(f"Final balance: ${backtest_results['final_balance']:.2f}")
            print(f"Profit/Loss: ${backtest_results['profit_loss']:.2f} ({backtest_results['profit_loss_percent']:.2f}%)")
            print(f"Total trades: {backtest_results['total_trades']}")
            print(f"Win rate: {backtest_results['win_rate'] * 100:.2f}%")
            print(f"Max drawdown: ${backtest_results['max_drawdown']:.2f} ({backtest_results['max_drawdown_percent']:.2f}%)")
            
            # Plot results
            plt.figure(figsize=(12, 8))
            
            # Plot price and signals
            plt.subplot(2, 1, 1)
            plt.plot(signals_df['close'], label='Close Price')
            
            # Plot buy signals
            buy_signals = signals_df[signals_df['signal'] == 1]
            plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='Buy Signal')
            
            # Plot sell signals
            sell_signals = signals_df[signals_df['signal'] == -1]
            plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='Sell Signal')
            
            plt.title(f'{name} Strategy - Signals')
            plt.legend()
            
            # Plot equity curve
            plt.subplot(2, 1, 2)
            equity_df = pd.DataFrame(backtest_results['equity_curve'])
            if not equity_df.empty:
                plt.plot(equity_df['timestamp'], equity_df['equity'], label='Equity')
                plt.title(f'{name} Strategy - Equity Curve')
                plt.legend()
            
            plt.tight_layout()
            plt.savefig(f'{name}_strategy_test.png')
            plt.close()
            
            print(f"Results saved to {name}_strategy_test.png")
        
    except Exception as e:
        print(f"Error testing strategies: {e}")
    finally:
        # Close data manager
        data_manager.close()


if __name__ == "__main__":
    # Run test
    test_trading_strategies()
