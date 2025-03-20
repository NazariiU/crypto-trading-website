"""
Technical Analysis Module for Cryptocurrency Trading Bot

This module implements various technical indicators and analysis tools including:
- Moving Averages (SMA, EMA)
- Oscillators (RSI, MACD)
- Volatility Indicators (Bollinger Bands)
- Volume Indicators (OBV, MFI)
- Trend Indicators (ADX)
- Candlestick Patterns
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from enum import Enum

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class IndicatorCategory(Enum):
    """Enum for indicator categories"""
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    PATTERN = "pattern"


class TechnicalIndicator:
    """Base class for all technical indicators"""
    
    def __init__(self, name: str, category: IndicatorCategory, params: Dict = None):
        """
        Initialize technical indicator
        
        Args:
            name: Indicator name
            category: Indicator category
            params: Indicator parameters
        """
        self.name = name
        self.category = category
        self.params = params or {}
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicator values
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with indicator values added
        """
        raise NotImplementedError("Subclasses must implement calculate method")
    
    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on indicator values
        
        Args:
            data: DataFrame containing indicator values
            
        Returns:
            Series with trading signals (1 for buy, -1 for sell, 0 for neutral)
        """
        raise NotImplementedError("Subclasses must implement get_signal method")


class MovingAverage(TechnicalIndicator):
    """Moving Average indicator base class"""
    
    def __init__(self, name: str, period: int = 14, price_column: str = 'close'):
        """
        Initialize Moving Average indicator
        
        Args:
            name: Indicator name
            period: Moving average period
            price_column: Column to use for calculation
        """
        super().__init__(name, IndicatorCategory.TREND, {'period': period, 'price_column': price_column})
        self.period = period
        self.price_column = price_column
    
    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on moving average crossovers
        
        Args:
            data: DataFrame containing indicator values
            
        Returns:
            Series with trading signals (1 for buy, -1 for sell, 0 for neutral)
        """
        ma_col = f"{self.name}_{self.period}"
        
        if ma_col not in data.columns:
            raise ValueError(f"Column {ma_col} not found in data")
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Price crosses above MA: Buy signal
        signals[data[self.price_column] > data[ma_col]] = 1
        
        # Price crosses below MA: Sell signal
        signals[data[self.price_column] < data[ma_col]] = -1
        
        return signals


class SMA(MovingAverage):
    """Simple Moving Average indicator"""
    
    def __init__(self, period: int = 14, price_column: str = 'close'):
        """
        Initialize SMA indicator
        
        Args:
            period: SMA period
            price_column: Column to use for calculation
        """
        super().__init__('sma', period, price_column)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SMA values
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with SMA values added
        """
        if self.price_column not in data.columns:
            raise ValueError(f"Column {self.price_column} not found in data")
        
        # Calculate SMA
        data[f"sma_{self.period}"] = data[self.price_column].rolling(window=self.period).mean()
        
        return data


class EMA(MovingAverage):
    """Exponential Moving Average indicator"""
    
    def __init__(self, period: int = 14, price_column: str = 'close'):
        """
        Initialize EMA indicator
        
        Args:
            period: EMA period
            price_column: Column to use for calculation
        """
        super().__init__('ema', period, price_column)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA values
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with EMA values added
        """
        if self.price_column not in data.columns:
            raise ValueError(f"Column {self.price_column} not found in data")
        
        # Calculate EMA
        data[f"ema_{self.period}"] = data[self.price_column].ewm(span=self.period, adjust=False).mean()
        
        return data


class RSI(TechnicalIndicator):
    """Relative Strength Index indicator"""
    
    def __init__(self, period: int = 14, price_column: str = 'close', 
                overbought: float = 70, oversold: float = 30):
        """
        Initialize RSI indicator
        
        Args:
            period: RSI period
            price_column: Column to use for calculation
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        super().__init__('rsi', IndicatorCategory.MOMENTUM, {
            'period': period, 
            'price_column': price_column,
            'overbought': overbought,
            'oversold': oversold
        })
        self.period = period
        self.price_column = price_column
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI values
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with RSI values added
        """
        if self.price_column not in data.columns:
            raise ValueError(f"Column {self.price_column} not found in data")
        
        # Calculate price changes
        delta = data[self.price_column].diff()
        
        # Separate gains and losses
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = abs(loss)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        data[f"rsi_{self.period}"] = 100 - (100 / (1 + rs))
        
        return data
    
    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on RSI values
        
        Args:
            data: DataFrame containing RSI values
            
        Returns:
            Series with trading signals (1 for buy, -1 for sell, 0 for neutral)
        """
        rsi_col = f"rsi_{self.period}"
        
        if rsi_col not in data.columns:
            raise ValueError(f"Column {rsi_col} not found in data")
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # RSI below oversold threshold: Buy signal
        signals[data[rsi_col] < self.oversold] = 1
        
        # RSI above overbought threshold: Sell signal
        signals[data[rsi_col] > self.overbought] = -1
        
        return signals


class MACD(TechnicalIndicator):
    """Moving Average Convergence Divergence indicator"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, 
                signal_period: int = 9, price_column: str = 'close'):
        """
        Initialize MACD indicator
        
        Args:
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            price_column: Column to use for calculation
        """
        super().__init__('macd', IndicatorCategory.MOMENTUM, {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
            'price_column': price_column
        })
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_column = price_column
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD values
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with MACD values added
        """
        if self.price_column not in data.columns:
            raise ValueError(f"Column {self.price_column} not found in data")
        
        # Calculate fast and slow EMAs
        fast_ema = data[self.price_column].ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = data[self.price_column].ewm(span=self.slow_period, adjust=False).mean()
        
        # Calculate MACD line
        data['macd_line'] = fast_ema - slow_ema
        
        # Calculate signal line
        data['macd_signal'] = data['macd_line'].ewm(span=self.signal_period, adjust=False).mean()
        
        # Calculate histogram
        data['macd_histogram'] = data['macd_line'] - data['macd_signal']
        
        return data
    
    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MACD values
        
        Args:
            data: DataFrame containing MACD values
            
        Returns:
            Series with trading signals (1 for buy, -1 for sell, 0 for neutral)
        """
        if 'macd_line' not in data.columns or 'macd_signal' not in data.columns:
            raise ValueError("MACD columns not found in data")
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # MACD line crosses above signal line: Buy signal
        signals[(data['macd_line'] > data['macd_signal']) & 
                (data['macd_line'].shift(1) <= data['macd_signal'].shift(1))] = 1
        
        # MACD line crosses below signal line: Sell signal
        signals[(data['macd_line'] < data['macd_signal']) & 
                (data['macd_line'].shift(1) >= data['macd_signal'].shift(1))] = -1
        
        return signals


class BollingerBands(TechnicalIndicator):
    """Bollinger Bands indicator"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, price_column: str = 'close'):
        """
        Initialize Bollinger Bands indicator
        
        Args:
            period: Moving average period
            std_dev: Standard deviation multiplier
            price_column: Column to use for calculation
        """
        super().__init__('bollinger_bands', IndicatorCategory.VOLATILITY, {
            'period': period,
            'std_dev': std_dev,
            'price_column': price_column
        })
        self.period = period
        self.std_dev = std_dev
        self.price_column = price_column
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands values
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with Bollinger Bands values added
        """
        if self.price_column not in data.columns:
            raise ValueError(f"Column {self.price_column} not found in data")
        
        # Calculate middle band (SMA)
        data['bb_middle'] = data[self.price_column].rolling(window=self.period).mean()
        
        # Calculate standard deviation
        rolling_std = data[self.price_column].rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        data['bb_upper'] = data['bb_middle'] + (rolling_std * self.std_dev)
        data['bb_lower'] = data['bb_middle'] - (rolling_std * self.std_dev)
        
        # Calculate bandwidth and %B
        data['bb_bandwidth'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
        data['bb_percent_b'] = (data[self.price_column] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        return data
    
    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on Bollinger Bands values
        
        Args:
            data: DataFrame containing Bollinger Bands values
            
        Returns:
            Series with trading signals (1 for buy, -1 for sell, 0 for neutral)
        """
        if 'bb_upper' not in data.columns or 'bb_lower' not in data.columns:
            raise ValueError("Bollinger Bands columns not found in data")
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Price touches or crosses below lower band: Buy signal
        signals[data[self.price_column] <= data['bb_lower']] = 1
        
        # Price touches or crosses above upper band: Sell signal
        signals[data[self.price_column] >= data['bb_upper']] = -1
        
        return signals


class OBV(TechnicalIndicator):
    """On-Balance Volume indicator"""
    
    def __init__(self):
        """Initialize OBV indicator"""
        super().__init__('obv', IndicatorCategory.VOLUME)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate OBV values
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with OBV values added
        """
        if 'close' not in data.columns or 'volume' not in data.columns:
            raise ValueError("Required columns not found in data")
        
        # Calculate price changes
        price_change = data['close'].diff()
        
        # Initialize OBV with first volume value
        obv = [data['volume'].iloc[0]]
        
        # Calculate OBV values
        for i in range(1, len(data)):
            if price_change.iloc[i] > 0:
                obv.append(obv[-1] + data['volume'].iloc[i])
            elif price_change.iloc[i] < 0:
                obv.append(obv[-1] - data['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        
        # Add OBV to data
        data['obv'] = obv
        
        # Add OBV EMA for signal line
        data['obv_ema'] = data['obv'].ewm(span=20, adjust=False).mean()
        
        return data
    
    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on OBV values
        
        Args:
            data: DataFrame containing OBV values
            
        Returns:
            Series with trading signals (1 for buy, -1 for sell, 0 for neutral)
        """
        if 'obv' not in data.columns or 'obv_ema' not in data.columns:
            raise ValueError("OBV columns not found in data")
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # OBV crosses above EMA: Buy signal
        signals[(data['obv'] > data['obv_ema']) & 
                (data['obv'].shift(1) <= data['obv_ema'].shift(1))] = 1
        
        # OBV crosses below EMA: Sell signal
        signals[(data['obv'] < data['obv_ema']) & 
                (data['obv'].shift(1) >= data['obv_ema'].shift(1))] = -1
        
        return signals


class MFI(TechnicalIndicator):
    """Money Flow Index indicator"""
    
    def __init__(self, period: int = 14, overbought: float = 80, oversold: float = 20):
        """
        Initialize MFI indicator
        
        Args:
            period: MFI period
            overbought: Overbought threshold
            oversold: Oversold threshold
        """
        super().__init__('mfi', IndicatorCategory.VOLUME, {
            'period': period,
            'overbought': overbought,
            'oversold': oversold
        })
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MFI values
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with MFI values added
        """
        required_columns = ['high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")
        
        # Calculate typical price
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * data['volume']
        
        # Calculate money flow direction
        money_flow_positive = pd.Series(0, index=data.index)
        money_flow_negative = pd.Series(0, index=data.index)
        
        # Determine positive and negative money flow
        for i in range(1, len(data)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                money_flow_positive.iloc[i] = raw_money_flow.iloc[i]
            elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                money_flow_negative.iloc[i] = raw_money_flow.iloc[i]
        
        # Calculate money flow ratio
        positive_flow = money_flow_positive.rolling(window=self.period).sum()
        negative_flow = money_flow_negative.rolling(window=self.period).sum()
        
        # Handle division by zero
        money_flow_ratio = np.where(negative_flow != 0, positive_flow / negative_flow, 1)
        
        # Calculate MFI
        data[f"mfi_{self.period}"] = 100 - (100 / (1 + money_flow_ratio))
        
        return data
    
    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on MFI values
        
        Args:
            data: DataFrame containing MFI values
            
        Returns:
            Series with trading signals (1 for buy, -1 for sell, 0 for neutral)
        """
        mfi_col = f"mfi_{self.period}"
        
        if mfi_col not in data.columns:
            raise ValueError(f"Column {mfi_col} not found in data")
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # MFI below oversold threshold: Buy signal
        signals[data[mfi_col] < self.oversold] = 1
        
        # MFI above overbought threshold: Sell signal
        signals[data[mfi_col] > self.overbought] = -1
        
        return signals


class ADX(TechnicalIndicator):
    """Average Directional Index indicator"""
    
    def __init__(self, period: int = 14, threshold: float = 25):
        """
        Initialize ADX indicator
        
        Args:
            period: ADX period
            threshold: Trend strength threshold
        """
        super().__init__('adx', IndicatorCategory.TREND, {
            'period': period,
            'threshold': threshold
        })
        self.period = period
        self.threshold = threshold
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX values
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with ADX values added
        """
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")
        
        # Calculate True Range
        data['tr'] = np.maximum(
            data['high'] - data['low'],
            np.maximum(
                abs(data['high'] - data['close'].shift(1)),
                abs(data['low'] - data['close'].shift(1))
            )
        )
        
        # Calculate Directional Movement
        data['dm_plus'] = np.where(
            (data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']),
            np.maximum(data['high'] - data['high'].shift(1), 0),
            0
        )
        
        data['dm_minus'] = np.where(
            (data['low'].shift(1) - data['low']) > (data['high'] - data['high'].shift(1)),
            np.maximum(data['low'].shift(1) - data['low'], 0),
            0
        )
        
        # Calculate smoothed values
        data['tr_' + str(self.period)] = data['tr'].rolling(window=self.period).sum()
        data['dm_plus_' + str(self.period)] = data['dm_plus'].rolling(window=self.period).sum()
        data['dm_minus_' + str(self.period)] = data['dm_minus'].rolling(window=self.period).sum()
        
        # Calculate Directional Indicators
        data['di_plus_' + str(self.period)] = 100 * data['dm_plus_' + str(self.period)] / data['tr_' + str(self.period)]
        data['di_minus_' + str(self.period)] = 100 * data['dm_minus_' + str(self.period)] / data['tr_' + str(self.period)]
        
        # Calculate Directional Index
        data['dx_' + str(self.period)] = 100 * abs(
            data['di_plus_' + str(self.period)] - data['di_minus_' + str(self.period)]
        ) / (data['di_plus_' + str(self.period)] + data['di_minus_' + str(self.period)])
        
        # Calculate ADX
        data[f"adx_{self.period}"] = data['dx_' + str(self.period)].rolling(window=self.period).mean()
        
        return data
    
    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on ADX values
        
        Args:
            data: DataFrame containing ADX values
            
        Returns:
            Series with trading signals (1 for buy, -1 for sell, 0 for neutral)
        """
        adx_col = f"adx_{self.period}"
        di_plus_col = f"di_plus_{self.period}"
        di_minus_col = f"di_minus_{self.period}"
        
        required_columns = [adx_col, di_plus_col, di_minus_col]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Strong trend with +DI crossing above -DI: Buy signal
        signals[(data[adx_col] > self.threshold) & 
                (data[di_plus_col] > data[di_minus_col]) & 
                (data[di_plus_col].shift(1) <= data[di_minus_col].shift(1))] = 1
        
        # Strong trend with -DI crossing above +DI: Sell signal
        signals[(data[adx_col] > self.threshold) & 
                (data[di_minus_col] > data[di_plus_col]) & 
                (data[di_minus_col].shift(1) <= data[di_plus_col].shift(1))] = -1
        
        return signals


class CandlestickPattern(TechnicalIndicator):
    """Base class for candlestick pattern recognition"""
    
    def __init__(self, name: str):
        """
        Initialize candlestick pattern
        
        Args:
            name: Pattern name
        """
        super().__init__(name, IndicatorCategory.PATTERN)
    
    def get_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on pattern recognition
        
        Args:
            data: DataFrame containing pattern recognition results
            
        Returns:
            Series with trading signals (1 for buy, -1 for sell, 0 for neutral)
        """
        pattern_col = f"pattern_{self.name}"
        
        if pattern_col not in data.columns:
            raise ValueError(f"Column {pattern_col} not found in data")
        
        # Initialize signals
        signals = pd.Series(0, index=data.index)
        
        # Bullish pattern: Buy signal
        signals[data[pattern_col] == 1] = 1
        
        # Bearish pattern: Sell signal
        signals[data[pattern_col] == -1] = -1
        
        return signals


class Doji(CandlestickPattern):
    """Doji candlestick pattern"""
    
    def __init__(self, doji_size: float = 0.1):
        """
        Initialize Doji pattern
        
        Args:
            doji_size: Maximum size of body relative to range
        """
        super().__init__('doji')
        self.doji_size = doji_size
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify Doji patterns
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with Doji pattern recognition results added
        """
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")
        
        # Calculate body size and range
        body_size = abs(data['close'] - data['open'])
        range_size = data['high'] - data['low']
        
        # Identify Doji patterns
        data['pattern_doji'] = 0
        
        # Doji condition: body size is very small compared to range
        doji_condition = (body_size / range_size) < self.doji_size
        
        # Bullish Doji after downtrend
        data.loc[(doji_condition) & (data['close'].shift(1) < data['open'].shift(1)), 'pattern_doji'] = 1
        
        # Bearish Doji after uptrend
        data.loc[(doji_condition) & (data['close'].shift(1) > data['open'].shift(1)), 'pattern_doji'] = -1
        
        return data


class Hammer(CandlestickPattern):
    """Hammer and Hanging Man candlestick patterns"""
    
    def __init__(self, body_size_ratio: float = 0.3, shadow_ratio: float = 2.0):
        """
        Initialize Hammer pattern
        
        Args:
            body_size_ratio: Maximum size of body relative to range
            shadow_ratio: Minimum ratio of lower shadow to body
        """
        super().__init__('hammer')
        self.body_size_ratio = body_size_ratio
        self.shadow_ratio = shadow_ratio
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify Hammer and Hanging Man patterns
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with pattern recognition results added
        """
        required_columns = ['open', 'high', 'low', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")
        
        # Calculate body size and shadows
        body_size = abs(data['close'] - data['open'])
        range_size = data['high'] - data['low']
        
        # Calculate upper and lower shadows
        upper_shadow = data.apply(
            lambda x: x['high'] - max(x['open'], x['close']), axis=1
        )
        lower_shadow = data.apply(
            lambda x: min(x['open'], x['close']) - x['low'], axis=1
        )
        
        # Initialize pattern column
        data['pattern_hammer'] = 0
        
        # Hammer/Hanging Man conditions
        pattern_condition = (
            (body_size / range_size <= self.body_size_ratio) &  # Small body
            (lower_shadow / (body_size + 0.0001) >= self.shadow_ratio) &  # Long lower shadow
            (upper_shadow / (body_size + 0.0001) <= 0.5)  # Short upper shadow
        )
        
        # Hammer (bullish) after downtrend
        data.loc[(pattern_condition) & 
                 (data['close'].rolling(window=5).mean().shift(1) < 
                  data['close'].rolling(window=10).mean().shift(1)), 'pattern_hammer'] = 1
        
        # Hanging Man (bearish) after uptrend
        data.loc[(pattern_condition) & 
                 (data['close'].rolling(window=5).mean().shift(1) > 
                  data['close'].rolling(window=10).mean().shift(1)), 'pattern_hammer'] = -1
        
        return data


class EngulfingPattern(CandlestickPattern):
    """Bullish and Bearish Engulfing candlestick patterns"""
    
    def __init__(self):
        """Initialize Engulfing pattern"""
        super().__init__('engulfing')
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Identify Engulfing patterns
        
        Args:
            data: DataFrame containing OHLCV data
            
        Returns:
            DataFrame with pattern recognition results added
        """
        required_columns = ['open', 'close']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Column {col} not found in data")
        
        # Initialize pattern column
        data['pattern_engulfing'] = 0
        
        # Bullish Engulfing
        bullish_engulfing = (
            (data['open'] < data['close']) &  # Current candle is bullish
            (data['open'].shift(1) > data['close'].shift(1)) &  # Previous candle is bearish
            (data['open'] < data['close'].shift(1)) &  # Current open below previous close
            (data['close'] > data['open'].shift(1))  # Current close above previous open
        )
        
        # Bearish Engulfing
        bearish_engulfing = (
            (data['open'] > data['close']) &  # Current candle is bearish
            (data['open'].shift(1) < data['close'].shift(1)) &  # Previous candle is bullish
            (data['open'] > data['close'].shift(1)) &  # Current open above previous close
            (data['close'] < data['open'].shift(1))  # Current close below previous open
        )
        
        # Set pattern values
        data.loc[bullish_engulfing, 'pattern_engulfing'] = 1
        data.loc[bearish_engulfing, 'pattern_engulfing'] = -1
        
        return data


class TechnicalAnalyzer:
    """Manager class for technical analysis"""
    
    def __init__(self):
        """Initialize technical analyzer"""
        self.indicators = {}
        self._register_default_indicators()
    
    def _register_default_indicators(self):
        """Register default indicators based on configuration"""
        # Register moving averages
        sma_config = config.INDICATORS.get('sma', {})
        self.register_indicator(SMA(period=sma_config.get('short_period', 9)))
        self.register_indicator(SMA(period=sma_config.get('medium_period', 21)))
        self.register_indicator(SMA(period=sma_config.get('long_period', 50)))
        
        ema_config = config.INDICATORS.get('ema', {})
        self.register_indicator(EMA(period=ema_config.get('short_period', 9)))
        self.register_indicator(EMA(period=ema_config.get('medium_period', 21)))
        self.register_indicator(EMA(period=ema_config.get('long_period', 50)))
        
        # Register oscillators
        rsi_config = config.INDICATORS.get('rsi', {})
        self.register_indicator(RSI(
            period=rsi_config.get('period', 14),
            overbought=rsi_config.get('overbought', 70),
            oversold=rsi_config.get('oversold', 30)
        ))
        
        macd_config = config.INDICATORS.get('macd', {})
        self.register_indicator(MACD(
            fast_period=macd_config.get('fast_period', 12),
            slow_period=macd_config.get('slow_period', 26),
            signal_period=macd_config.get('signal_period', 9)
        ))
        
        # Register volatility indicators
        bb_config = config.INDICATORS.get('bollinger_bands', {})
        self.register_indicator(BollingerBands(
            period=bb_config.get('period', 20),
            std_dev=bb_config.get('std_dev', 2)
        ))
        
        # Register volume indicators
        self.register_indicator(OBV())
        self.register_indicator(MFI())
        
        # Register trend indicators
        self.register_indicator(ADX())
        
        # Register candlestick patterns
        self.register_indicator(Doji())
        self.register_indicator(Hammer())
        self.register_indicator(EngulfingPattern())
    
    def register_indicator(self, indicator: TechnicalIndicator):
        """
        Register a technical indicator
        
        Args:
            indicator: Technical indicator instance
        """
        key = f"{indicator.name}_{indicator.params.get('period', '')}"
        self.indicators[key] = indicator
    
    def calculate_indicators(self, data: pd.DataFrame, indicators: List[str] = None) -> pd.DataFrame:
        """
        Calculate technical indicators
        
        Args:
            data: DataFrame containing OHLCV data
            indicators: List of indicator names to calculate (if None, calculate all)
            
        Returns:
            DataFrame with indicator values added
        """
        # Make a copy of the data to avoid modifying the original
        result = data.copy()
        
        # Determine which indicators to calculate
        if indicators:
            indicator_keys = [key for key in self.indicators.keys() if any(i in key for i in indicators)]
        else:
            indicator_keys = list(self.indicators.keys())
        
        # Calculate each indicator
        for key in indicator_keys:
            try:
                result = self.indicators[key].calculate(result)
            except Exception as e:
                print(f"Error calculating {key}: {e}")
        
        return result
    
    def get_signals(self, data: pd.DataFrame, indicators: List[str] = None) -> Dict[str, pd.Series]:
        """
        Generate trading signals for indicators
        
        Args:
            data: DataFrame containing indicator values
            indicators: List of indicator names to get signals for (if None, get all)
            
        Returns:
            Dictionary mapping indicator names to signal series
        """
        signals = {}
        
        # Determine which indicators to get signals for
        if indicators:
            indicator_keys = [key for key in self.indicators.keys() if any(i in key for i in indicators)]
        else:
            indicator_keys = list(self.indicators.keys())
        
        # Get signals for each indicator
        for key in indicator_keys:
            try:
                signals[key] = self.indicators[key].get_signal(data)
            except Exception as e:
                print(f"Error getting signals for {key}: {e}")
        
        return signals
    
    def get_combined_signal(self, data: pd.DataFrame, indicators: List[str] = None, 
                           weights: Dict[str, float] = None) -> pd.Series:
        """
        Generate combined trading signal from multiple indicators
        
        Args:
            data: DataFrame containing indicator values
            indicators: List of indicator names to include (if None, include all)
            weights: Dictionary mapping indicator names to weights (if None, equal weights)
            
        Returns:
            Series with combined trading signals
        """
        # Get individual signals
        signals = self.get_signals(data, indicators)
        
        if not signals:
            return pd.Series(0, index=data.index)
        
        # Determine weights
        if weights is None:
            weights = {key: 1.0 / len(signals) for key in signals.keys()}
        
        # Combine signals
        combined_signal = pd.Series(0, index=data.index)
        
        for key, signal in signals.items():
            if key in weights:
                combined_signal += signal * weights[key]
        
        # Normalize to [-1, 1] range
        combined_signal = combined_signal.clip(-1, 1)
        
        return combined_signal


# Factory function to create technical analyzer
def create_technical_analyzer() -> TechnicalAnalyzer:
    """
    Create a technical analyzer
    
    Returns:
        TechnicalAnalyzer instance
    """
    return TechnicalAnalyzer()


# Test function to verify the module works correctly
def test_technical_analysis():
    """Test technical analysis functionality"""
    import matplotlib.pyplot as plt
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.normal(1000, 200, 100)
    })
    
    # Ensure high >= open, close and low <= open, close
    for i in range(len(data)):
        data.loc[i, 'high'] = max(data.loc[i, 'high'], data.loc[i, 'open'], data.loc[i, 'close'])
        data.loc[i, 'low'] = min(data.loc[i, 'low'], data.loc[i, 'open'], data.loc[i, 'close'])
    
    # Create technical analyzer
    analyzer = create_technical_analyzer()
    
    # Calculate indicators
    result = analyzer.calculate_indicators(data)
    
    # Get signals
    signals = analyzer.get_signals(result)
    
    # Get combined signal
    combined_signal = analyzer.get_combined_signal(result)
    
    # Print results
    print(f"Calculated {len(result.columns) - len(data.columns)} indicators")
    print(f"Generated signals for {len(signals)} indicators")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot price
    plt.subplot(3, 1, 1)
    plt.plot(result['close'], label='Close Price')
    plt.plot(result['sma_9'], label='SMA 9')
    plt.plot(result['ema_21'], label='EMA 21')
    plt.title('Price and Moving Averages')
    plt.legend()
    
    # Plot oscillators
    plt.subplot(3, 1, 2)
    plt.plot(result['rsi_14'], label='RSI 14')
    plt.axhline(y=70, color='r', linestyle='--')
    plt.axhline(y=30, color='g', linestyle='--')
    plt.title('RSI')
    plt.legend()
    
    # Plot combined signal
    plt.subplot(3, 1, 3)
    plt.plot(combined_signal, label='Combined Signal')
    plt.axhline(y=0, color='k', linestyle='-')
    plt.title('Combined Signal')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('technical_analysis_test.png')
    plt.close()
    
    print("Test completed. Results saved to technical_analysis_test.png")


if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Run test
    test_technical_analysis()
