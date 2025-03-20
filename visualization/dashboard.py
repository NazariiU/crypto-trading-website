"""
Visualization Interface for Cryptocurrency Trading Bot

This module implements the web-based user interface using Streamlit
to display charts, indicators, trading signals, and performance metrics.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

# Import configuration and other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.data_manager import create_data_manager
from strategies.technical_indicators import create_technical_analyzer
from strategies.trading_strategies import create_strategy_manager

# Configure logging
logger = logging.getLogger('crypto_trading_bot.visualization')

class TradingDashboard:
    """Main dashboard class for the trading bot UI"""
    
    def __init__(self):
        """Initialize the trading dashboard"""
        self.data_manager = create_data_manager()
        self.technical_analyzer = create_technical_analyzer()
        self.strategy_manager = create_strategy_manager()
        
        # Set page configuration
        st.set_page_config(
            page_title="Crypto Trading Bot",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Initialize session state if not exists
        if 'selected_symbol' not in st.session_state:
            st.session_state.selected_symbol = config.TRADING_PARAMS['default_symbols'][0]
        if 'selected_timeframe' not in st.session_state:
            st.session_state.selected_timeframe = config.TRADING_PARAMS['default_timeframes'][3]  # 1h
        if 'selected_strategy' not in st.session_state:
            st.session_state.selected_strategy = config.TRADING_PARAMS['default_strategies'][0]  # rsi_macd
        if 'trading_mode' not in st.session_state:
            st.session_state.trading_mode = 'signal'  # 'auto' or 'signal'
        if 'indicators' not in st.session_state:
            st.session_state.indicators = {
                'sma': True,
                'ema': True,
                'rsi': True,
                'macd': True,
                'bollinger_bands': True,
                'volume': True,
                'patterns': False
            }
    
    def run(self):
        """Run the dashboard application"""
        # Display header
        st.title("Cryptocurrency Trading Bot")
        
        # Create sidebar for controls
        self._create_sidebar()
        
        # Create main content area with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Market View", "Strategy Analysis", "Trading", "Performance"])
        
        with tab1:
            self._render_market_view()
        
        with tab2:
            self._render_strategy_analysis()
        
        with tab3:
            self._render_trading_view()
        
        with tab4:
            self._render_performance_view()
    
    def _create_sidebar(self):
        """Create sidebar with controls"""
        st.sidebar.header("Settings")
        
        # Exchange selection
        exchange_options = list(config.EXCHANGE_CONFIGS.keys())
        selected_exchange = st.sidebar.selectbox(
            "Exchange",
            options=exchange_options,
            index=exchange_options.index(config.DEFAULT_EXCHANGE)
        )
        
        # Symbol selection
        symbol_options = config.TRADING_PARAMS['default_symbols']
        selected_symbol = st.sidebar.selectbox(
            "Symbol",
            options=symbol_options,
            index=symbol_options.index(st.session_state.selected_symbol)
        )
        st.session_state.selected_symbol = selected_symbol
        
        # Timeframe selection
        timeframe_options = config.TRADING_PARAMS['default_timeframes']
        selected_timeframe = st.sidebar.selectbox(
            "Timeframe",
            options=timeframe_options,
            index=timeframe_options.index(st.session_state.selected_timeframe)
        )
        st.session_state.selected_timeframe = selected_timeframe
        
        # Strategy selection
        strategy_options = list(self.strategy_manager.get_all_strategies().keys())
        selected_strategy = st.sidebar.selectbox(
            "Strategy",
            options=strategy_options,
            index=strategy_options.index(st.session_state.selected_strategy) if st.session_state.selected_strategy in strategy_options else 0
        )
        st.session_state.selected_strategy = selected_strategy
        
        # Trading mode selection
        trading_mode = st.sidebar.radio(
            "Trading Mode",
            options=["Signal", "Auto"],
            index=0 if st.session_state.trading_mode == 'signal' else 1
        )
        st.session_state.trading_mode = trading_mode.lower()
        
        # Indicator selection
        st.sidebar.header("Indicators")
        
        # Create columns for indicator checkboxes
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            st.session_state.indicators['sma'] = st.checkbox("SMA", value=st.session_state.indicators['sma'])
            st.session_state.indicators['ema'] = st.checkbox("EMA", value=st.session_state.indicators['ema'])
            st.session_state.indicators['rsi'] = st.checkbox("RSI", value=st.session_state.indicators['rsi'])
            st.session_state.indicators['macd'] = st.checkbox("MACD", value=st.session_state.indicators['macd'])
        
        with col2:
            st.session_state.indicators['bollinger_bands'] = st.checkbox("Bollinger Bands", value=st.session_state.indicators['bollinger_bands'])
            st.session_state.indicators['volume'] = st.checkbox("Volume", value=st.session_state.indicators['volume'])
            st.session_state.indicators['patterns'] = st.checkbox("Patterns", value=st.session_state.indicators['patterns'])
        
        # Data range selection
        st.sidebar.header("Data Range")
        days_back = st.sidebar.slider("Days", min_value=1, max_value=90, value=30)
        
        # Refresh button
        if st.sidebar.button("Refresh Data"):
            st.session_state.data = self._fetch_data(
                st.session_state.selected_symbol,
                st.session_state.selected_timeframe,
                days_back
            )
            st.experimental_rerun()
    
    def _fetch_data(self, symbol: str, timeframe: str, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch market data for the selected symbol and timeframe
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            days_back: Number of days to fetch
            
        Returns:
            DataFrame containing market data with indicators
        """
        try:
            # Fetch historical data
            data = self.data_manager.fetch_historical_data(symbol, timeframe, days_back)
            
            if data.empty:
                st.error(f"No data available for {symbol} on {timeframe} timeframe")
                # Create sample data for demonstration
                dates = pd.date_range(start=datetime.now() - timedelta(days=days_back), periods=days_back * 24, freq='H')
                data = pd.DataFrame({
                    'timestamp': dates,
                    'open': np.random.normal(100, 5, days_back * 24),
                    'high': np.random.normal(105, 5, days_back * 24),
                    'low': np.random.normal(95, 5, days_back * 24),
                    'close': np.random.normal(100, 5, days_back * 24),
                    'volume': np.random.normal(1000, 200, days_back * 24)
                })
                
                # Ensure high >= open, close and low <= open, close
                for i in range(len(data)):
                    data.loc[i, 'high'] = max(data.loc[i, 'high'], data.loc[i, 'open'], data.loc[i, 'close'])
                    data.loc[i, 'low'] = min(data.loc[i, 'low'], data.loc[i, 'open'], data.loc[i, 'close'])
            
            # Calculate indicators
            indicators_to_calculate = [k for k, v in st.session_state.indicators.items() if v]
            data_with_indicators = self.technical_analyzer.calculate_indicators(data, indicators_to_calculate)
            
            # Generate signals for selected strategy
            strategy = self.strategy_manager.get_strategy(st.session_state.selected_strategy)
            if strategy:
                data_with_signals = strategy.generate_signals(data_with_indicators)
            else:
                data_with_signals = data_with_indicators
                data_with_signals['signal'] = 0
            
            return data_with_signals
            
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()
    
    def _render_market_view(self):
        """Render the market view tab"""
        st.header("Market View")
        
        # Fetch data if not in session state
        if 'data' not in st.session_state:
            st.session_state.data = self._fetch_data(
                st.session_state.selected_symbol,
                st.session_state.selected_timeframe
            )
        
        data = st.session_state.data
        
        if data.empty:
            st.warning("No data available. Please check your connection or try another symbol.")
            return
        
        # Get real-time data for the selected symbol
        try:
            real_time_data = self.data_manager.get_real_time_data(st.session_state.selected_symbol)
            
            if real_time_data:
                # Create columns for price information
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Current Price",
                        f"${real_time_data['last_price']:.2f}",
                        f"{real_time_data['change_24h']:.2f}%"
                    )
                
                with col2:
                    st.metric("24h Volume", f"${real_time_data['volume_24h']:.2f}")
                
                with col3:
                    st.metric("Bid", f"${real_time_data['bid']:.2f}")
                
                with col4:
                    st.metric("Ask", f"${real_time_data['ask']:.2f}")
        except Exception as e:
            logger.error(f"Error fetching real-time data: {e}")
        
        # Create main chart
        self._create_candlestick_chart(data)
        
        # Create indicator charts
        self._create_indicator_charts(data)
    
    def _create_candlestick_chart(self, data: pd.DataFrame):
        """
        Create candlestick chart with selected indicators
        
        Args:
            data: DataFrame containing market data with indicators
        """
        # Create figure with secondary y-axis for volume
        fig = make_subplots(
            rows=1, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=["Price Chart"]
        )
        
        # Add candlestick trace
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            )
        )
        
        # Add SMA if selected
        if st.session_state.indicators['sma']:
            if 'sma_9' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['sma_9'],
                        name="SMA 9",
                        line=dict(color='blue', width=1)
                    )
                )
            
            if 'sma_21' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['sma_21'],
                        name="SMA 21",
                        line=dict(color='orange', width=1)
                    )
                )
        
        # Add EMA if selected
        if st.session_state.indicators['ema']:
            if 'ema_9' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['ema_9'],
                        name="EMA 9",
                        line=dict(color='purple', width=1)
                    )
                )
            
            if 'ema_21' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['ema_21'],
                        name="EMA 21",
                        line=dict(color='green', width=1)
                    )
                )
        
        # Add Bollinger Bands if selected
        if st.session_state.indicators['bollinger_bands']:
            if all(col in data.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['bb_upper'],
                        name="BB Upper",
                        line=dict(color='rgba(250, 0, 0, 0.5)', width=1)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['bb_middle'],
                        name="BB Middle",
                        line=dict(color='rgba(0, 0, 250, 0.5)', width=1)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['bb_lower'],
                        name="BB Lower",
                        line=dict(color='rgba(250, 0, 0, 0.5)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(200, 200, 200, 0.2)'
                    )
                )
        
        # Add buy/sell signals if available
        if 'signal' in data.columns:
            # Buy signals
            buy_signals = data[data['signal'] == 1]
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['low'] * 0.99,  # Slightly below the candle
                        name="Buy Signal",
                        mode="markers",
                        marker=dict(
                            symbol="triangle-up",
                            size=10,
                            color="green",
                            line=dict(width=1, color="darkgreen")
                        )
                    )
                )
            
            # Sell signals
            sell_signals = data[data['signal'] == -1]
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['high'] * 1.01,  # Slightly above the candle
                        name="Sell Signal",
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=10,
                            color="red",
                            line=dict(width=1, color="darkred")
                        )
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f"{st.session_state.selected_symbol} - {st.session_state.selected_timeframe}",
            xaxis_title="Date",
            yaxis_title="Price",
            height=600,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_indicator_charts(self, data: pd.DataFrame):
        """
        Create charts for selected indicators
        
        Args:
            data: DataFrame containing market data with indicators
        """
        # Create volume chart if selected
        if st.session_state.indicators['volume']:
            fig_volume = go.Figure()
            
            fig_volume.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name="Volume",
                    marker=dict(
                        color='rgba(0, 0, 250, 0.5)'
                    )
                )
            )
            
            # Add OBV if available
            if 'obv' in data.columns:
                # Create a secondary y-axis for OBV
                fig_volume = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig_volume.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['volume'],
                        name="Volume",
                        marker=dict(
                            color='rgba(0, 0, 250, 0.5)'
                        )
                    ),
                    secondary_y=False
                )
                
                fig_volume.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['obv'],
                        name="OBV",
                        line=dict(color='red', width=1)
                    ),
                    secondary_y=True
                )
            
            fig_volume.update_layout(
                title="Volume",
                xaxis_title="Date",
                yaxis_title="Volume",
                height=300,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
        
        # Create RSI chart if selected
        if st.session_state.indicators['rsi'] and 'rsi_14' in data.columns:
            fig_rsi = go.Figure()
            
            fig_rsi.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['rsi_14'],
                    name="RSI (14)",
                    line=dict(color='purple', width=1)
                )
            )
            
            # Add overbought/oversold lines
            fig_rsi.add_shape(
                type="line",
                x0=data.index[0],
                y0=70,
                x1=data.index[-1],
                y1=70,
                line=dict(
                    color="red",
                    width=1,
                    dash="dash",
                )
            )
            
            fig_rsi.add_shape(
                type="line",
                x0=data.index[0],
                y0=30,
                x1=data.index[-1],
                y1=30,
                line=dict(
                    color="green",
                    width=1,
                    dash="dash",
                )
            )
            
            fig_rsi.update_layout(
                title="RSI (14)",
                xaxis_title="Date",
                yaxis_title="RSI",
                height=300,
                yaxis=dict(range=[0, 100])
            )
            
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        # Create MACD chart if selected
        if st.session_state.indicators['macd'] and all(col in data.columns for col in ['macd_line', 'macd_signal', 'macd_histogram']):
            fig_macd = make_subplots(rows=1, cols=1)
            
            fig_macd.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['macd_line'],
                    name="MACD Line",
                    line=dict(color='blue', width=1)
                )
            )
            
            fig_macd.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['macd_signal'],
                    name="Signal Line",
                    line=dict(color='red', width=1)
                )
            )
            
            # Add histogram
            colors = ['green' if val >= 0 else 'red' for val in data['macd_histogram']]
            
            fig_macd.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['macd_histogram'],
                    name="Histogram",
                    marker=dict(
                        color=colors
                    )
                )
            )
            
            fig_macd.update_layout(
                title="MACD",
                xaxis_title="Date",
                yaxis_title="MACD",
                height=300,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_macd, use_container_width=True)
    
    def _render_strategy_analysis(self):
        """Render the strategy analysis tab"""
        st.header("Strategy Analysis")
        
        # Get data
        if 'data' not in st.session_state:
            st.session_state.data = self._fetch_data(
                st.session_state.selected_symbol,
                st.session_state.selected_timeframe
            )
        
        data = st.session_state.data
        
        if data.empty:
            st.warning("No data available. Please check your connection or try another symbol.")
            return
        
        # Get selected strategy
        strategy = self.strategy_manager.get_strategy(st.session_state.selected_strategy)
        
        if not strategy:
            st.error(f"Strategy '{st.session_state.selected_strategy}' not found")
            return
        
        # Display strategy information
        st.subheader(f"Strategy: {strategy.name}")
        st.write(f"Description: {strategy.description}")
        st.write("Parameters:")
        for param, value in strategy.params.items():
            st.write(f"- {param}: {value}")
        
        # Run backtest
        st.subheader("Backtest Results")
        
        # Allow user to set initial balance
        initial_balance = st.number_input("Initial Balance ($)", min_value=100.0, max_value=1000000.0, value=10000.0, step=1000.0)
        
        if st.button("Run Backtest"):
            with st.spinner("Running backtest..."):
                backtest_results = strategy.backtest(data, initial_balance)
                
                # Display backtest metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Final Balance",
                        f"${backtest_results['final_balance']:.2f}",
                        f"{backtest_results['profit_loss_percent']:.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Total Trades",
                        f"{backtest_results['total_trades']}"
                    )
                
                with col3:
                    st.metric(
                        "Win Rate",
                        f"{backtest_results['win_rate'] * 100:.2f}%"
                    )
                
                col4, col5 = st.columns(2)
                
                with col4:
                    st.metric(
                        "Profit/Loss",
                        f"${backtest_results['profit_loss']:.2f}"
                    )
                
                with col5:
                    st.metric(
                        "Max Drawdown",
                        f"${backtest_results['max_drawdown']:.2f}",
                        f"-{backtest_results['max_drawdown_percent']:.2f}%",
                        delta_color="inverse"
                    )
                
                # Plot equity curve
                if backtest_results['equity_curve']:
                    equity_df = pd.DataFrame(backtest_results['equity_curve'])
                    
                    fig = go.Figure()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=equity_df['timestamp'],
                            y=equity_df['equity'],
                            name="Equity",
                            line=dict(color='blue', width=2)
                        )
                    )
                    
                    fig.update_layout(
                        title="Equity Curve",
                        xaxis_title="Date",
                        yaxis_title="Equity ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Display trade list
                if backtest_results['trades']:
                    st.subheader("Trade List")
                    
                    trades_df = pd.DataFrame(backtest_results['trades'])
                    
                    # Format trade data
                    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                    trades_df['price'] = trades_df['price'].map('${:.2f}'.format)
                    trades_df['cost'] = trades_df['cost'].map('${:.2f}'.format)
                    trades_df['balance'] = trades_df['balance'].map('${:.2f}'.format)
                    
                    if 'pnl' in trades_df.columns:
                        trades_df['pnl'] = trades_df['pnl'].map('${:.2f}'.format)
                    
                    if 'pnl_percent' in trades_df.columns:
                        trades_df['pnl_percent'] = trades_df['pnl_percent'].map('{:.2f}%'.format)
                    
                    st.dataframe(trades_df)
    
    def _render_trading_view(self):
        """Render the trading view tab"""
        st.header("Trading")
        
        # Check if API keys are configured
        exchange_config = config.EXCHANGE_CONFIGS.get(config.DEFAULT_EXCHANGE, {})
        api_key = exchange_config.get('api_key', '')
        
        if not api_key:
            st.warning("API keys not configured. Please add your API keys in the configuration file to enable trading.")
            
            # Display sample form for API key configuration
            with st.expander("Configure API Keys"):
                exchange = st.selectbox("Exchange", options=list(config.EXCHANGE_CONFIGS.keys()))
                api_key = st.text_input("API Key", type="password")
                api_secret = st.text_input("API Secret", type="password")
                
                if st.button("Save"):
                    st.success("API keys saved (this is a simulation, keys are not actually saved)")
            
            return
        
        # Display trading mode
        st.subheader(f"Trading Mode: {st.session_state.trading_mode.capitalize()}")
        
        # Get real-time data
        try:
            real_time_data = self.data_manager.get_real_time_data(st.session_state.selected_symbol)
            
            if real_time_data:
                current_price = real_time_data['last_price']
                
                # Display current price and trading pair
                st.metric(
                    f"Current Price ({st.session_state.selected_symbol})",
                    f"${current_price:.2f}",
                    f"{real_time_data['change_24h']:.2f}%"
                )
                
                # Create columns for order form
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Manual Trading")
                    
                    # Order form
                    order_type = st.selectbox("Order Type", options=["Market", "Limit"])
                    side = st.selectbox("Side", options=["Buy", "Sell"])
                    
                    # Amount input
                    amount = st.number_input("Amount", min_value=0.0001, step=0.01)
                    
                    # Price input for limit orders
                    price = None
                    if order_type == "Limit":
                        price = st.number_input("Price", min_value=0.01, value=float(current_price), step=0.01)
                    
                    # Calculate total
                    total = amount * (price if price else current_price)
                    st.write(f"Total: ${total:.2f}")
                    
                    # Add stop loss and take profit
                    use_sl_tp = st.checkbox("Use Stop Loss / Take Profit")
                    
                    if use_sl_tp:
                        sl_price = st.number_input(
                            "Stop Loss Price",
                            min_value=0.01,
                            value=float(current_price * 0.95) if side == "Buy" else float(current_price * 1.05),
                            step=0.01
                        )
                        
                        tp_price = st.number_input(
                            "Take Profit Price",
                            min_value=0.01,
                            value=float(current_price * 1.05) if side == "Buy" else float(current_price * 0.95),
                            step=0.01
                        )
                    
                    # Submit button
                    if st.button(f"Place {side} Order"):
                        st.success(f"{side} order placed successfully (simulation)")
                
                with col2:
                    st.subheader("Automated Trading")
                    
                    # Strategy selection
                    strategy_options = list(self.strategy_manager.get_all_strategies().keys())
                    auto_strategy = st.selectbox(
                        "Strategy",
                        options=strategy_options,
                        index=strategy_options.index(st.session_state.selected_strategy) if st.session_state.selected_strategy in strategy_options else 0
                    )
                    
                    # Risk management settings
                    st.write("Risk Management")
                    
                    max_position_size = st.slider(
                        "Max Position Size (%)",
                        min_value=1.0,
                        max_value=100.0,
                        value=float(config.TRADING_PARAMS['risk_management']['max_position_size_percent']),
                        step=1.0
                    )
                    
                    stop_loss_percent = st.slider(
                        "Stop Loss (%)",
                        min_value=0.5,
                        max_value=20.0,
                        value=float(config.TRADING_PARAMS['risk_management']['default_stop_loss_percent']),
                        step=0.5
                    )
                    
                    take_profit_percent = st.slider(
                        "Take Profit (%)",
                        min_value=0.5,
                        max_value=50.0,
                        value=float(config.TRADING_PARAMS['risk_management']['default_take_profit_percent']),
                        step=0.5
                    )
                    
                    # Enable/disable automated trading
                    auto_trading_enabled = st.checkbox("Enable Automated Trading", value=False)
                    
                    if auto_trading_enabled:
                        st.warning("Automated trading is enabled (simulation)")
                    else:
                        st.info("Automated trading is disabled")
        
        except Exception as e:
            st.error(f"Error fetching real-time data: {e}")
            logger.error(f"Error fetching real-time data: {e}")
    
    def _render_performance_view(self):
        """Render the performance view tab"""
        st.header("Performance Analytics")
        
        # Display sample performance metrics
        st.subheader("Trading Performance")
        
        # Create sample data for demonstration
        performance_data = {
            'date': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'balance': np.cumsum(np.random.normal(50, 200, 30)) + 10000,
            'trades': np.random.randint(0, 5, 30),
            'win_rate': np.random.uniform(0.4, 0.7, 30)
        }
        
        perf_df = pd.DataFrame(performance_data)
        perf_df['returns'] = perf_df['balance'].pct_change()
        perf_df['cumulative_returns'] = (1 + perf_df['returns']).cumprod() - 1
        
        # Calculate performance metrics
        total_trades = perf_df['trades'].sum()
        win_rate = perf_df['win_rate'].mean()
        total_return = perf_df['cumulative_returns'].iloc[-1]
        max_drawdown = (perf_df['balance'] / perf_df['balance'].cummax() - 1).min()
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", f"{total_trades}")
        
        with col2:
            st.metric("Win Rate", f"{win_rate:.2%}")
        
        with col3:
            st.metric("Total Return", f"{total_return:.2%}")
        
        with col4:
            st.metric("Max Drawdown", f"{max_drawdown:.2%}", delta_color="inverse")
        
        # Plot equity curve
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=perf_df['date'],
                y=perf_df['balance'],
                name="Account Balance",
                line=dict(color='blue', width=2)
            )
        )
        
        fig.update_layout(
            title="Account Balance Over Time",
            xaxis_title="Date",
            yaxis_title="Balance ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Plot returns distribution
        fig_returns = go.Figure()
        
        fig_returns.add_trace(
            go.Histogram(
                x=perf_df['returns'].dropna(),
                name="Daily Returns",
                marker=dict(color='green')
            )
        )
        
        fig_returns.update_layout(
            title="Distribution of Daily Returns",
            xaxis_title="Return",
            yaxis_title="Frequency",
            height=300
        )
        
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # Display trade history
        st.subheader("Trade History")
        
        # Create sample trade history for demonstration
        trade_history = []
        
        for i in range(20):
            side = "Buy" if np.random.random() > 0.5 else "Sell"
            entry_price = np.random.uniform(90, 110)
            exit_price = entry_price * (np.random.uniform(0.95, 1.10) if side == "Buy" else np.random.uniform(0.90, 1.05))
            amount = np.random.uniform(0.1, 2.0)
            
            pnl = (exit_price - entry_price) * amount if side == "Buy" else (entry_price - exit_price) * amount
            pnl_percent = (exit_price / entry_price - 1) * 100 if side == "Buy" else (entry_price / exit_price - 1) * 100
            
            trade_history.append({
                'date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=i),
                'symbol': st.session_state.selected_symbol,
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'amount': amount,
                'pnl': pnl,
                'pnl_percent': pnl_percent
            })
        
        trade_df = pd.DataFrame(trade_history)
        
        # Format trade data
        trade_df['entry_price'] = trade_df['entry_price'].map('${:.2f}'.format)
        trade_df['exit_price'] = trade_df['exit_price'].map('${:.2f}'.format)
        trade_df['pnl'] = trade_df['pnl'].map('${:.2f}'.format)
        trade_df['pnl_percent'] = trade_df['pnl_percent'].map('{:.2f}%'.format)
        
        st.dataframe(trade_df)


def run_dashboard():
    """Run the trading dashboard"""
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOGGING['level']),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run dashboard
    dashboard = TradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    run_dashboard()
