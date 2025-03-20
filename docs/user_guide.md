# Cryptocurrency Trading Web Application - User Guide

## Introduction

Welcome to the Cryptocurrency Trading Web Application! This guide will help you get started with using the platform for monitoring markets, analyzing trading strategies, and executing trades.

## Getting Started

### Accessing the Application

The application is deployed on Streamlit Cloud and can be accessed at the following URL:
[https://crypto-trading-bot.streamlit.app](https://crypto-trading-bot.streamlit.app)

### Creating an Account

1. Visit the application URL
2. Click on the "Sign Up" section below the login form
3. Fill in the required information:
   - Username
   - Full Name
   - Email
   - Password
4. Click "Sign Up" to create your account
5. You will receive a confirmation message when your account is created successfully

### Logging In

1. Visit the application URL
2. Enter your username and password
3. Click "Login" to access your dashboard

## Dashboard Overview

The application is organized into five main tabs:

1. **Market View**: Real-time market data and charts
2. **Strategy Analysis**: Backtest and analyze trading strategies
3. **Trading**: Execute trades manually or set up automated trading
4. **Performance**: Track and analyze your trading performance
5. **Settings**: Configure your account, API keys, and notification preferences

### Sidebar Controls

The sidebar contains controls for:
- Exchange selection
- Symbol selection
- Timeframe selection
- Strategy selection
- Trading mode (Signal or Auto)
- Indicator selection
- Data range selection

## Market View

The Market View tab provides real-time market data visualization:

1. **Price Information**: Current price, 24h change, volume, bid, and ask
2. **Candlestick Chart**: Price chart with selected technical indicators
3. **Volume Chart**: Trading volume with On-Balance Volume (OBV) indicator
4. **RSI Chart**: Relative Strength Index with overbought/oversold levels
5. **MACD Chart**: Moving Average Convergence Divergence indicator

### Customizing Charts

1. Use the sidebar to select which indicators to display
2. Adjust the timeframe to view different time periods
3. Change the data range to view more or less historical data
4. Click the "Refresh Data" button to update the charts with the latest data

## Strategy Analysis

The Strategy Analysis tab allows you to backtest trading strategies:

1. **Strategy Information**: Description and parameters of the selected strategy
2. **Backtest Configuration**: Set initial balance for backtesting
3. **Backtest Results**: Performance metrics including final balance, profit/loss, win rate, and drawdown
4. **Equity Curve**: Visual representation of account balance over time
5. **Trade List**: Detailed list of all trades executed during the backtest

### Running a Backtest

1. Select a strategy from the sidebar
2. Set the initial balance
3. Click "Run Backtest" to execute the backtest
4. Review the results and adjust strategy parameters as needed

## Trading

The Trading tab provides tools for executing trades:

1. **Manual Trading**: Place market or limit orders with optional stop-loss and take-profit levels
2. **Automated Trading**: Configure and enable automated trading based on selected strategies

### Manual Trading

1. Select order type (Market or Limit)
2. Choose side (Buy or Sell)
3. Enter amount to trade
4. For limit orders, specify the price
5. Optionally enable stop-loss and take-profit
6. Click "Place Buy/Sell Order" to execute

### Automated Trading

1. Select a strategy from the dropdown
2. Configure risk management settings:
   - Max position size (% of account)
   - Stop loss (%)
   - Take profit (%)
3. Enable automated trading by checking the box
4. The system will execute trades based on the selected strategy and risk parameters

## Performance Analytics

The Performance Analytics tab helps you track your trading performance:

1. **Performance Metrics**: Total trades, win rate, total return, and maximum drawdown
2. **Account Balance Chart**: Visual representation of your account balance over time
3. **Trade History**: Detailed list of all your executed trades

## Settings

The Settings tab allows you to configure your account and preferences:

### API Keys

1. Select an exchange
2. Enter your API key and secret
3. Click "Save API Keys" to store them securely

### Notification Settings

1. **Telegram Notifications**:
   - Enable/disable Telegram notifications
   - Configure bot token and chat ID
   
2. **Email Notifications**:
   - Enable/disable email notifications
   - Configure SMTP server, port, and credentials

### Account Settings

1. **Change Password**:
   - Enter current password
   - Enter and confirm new password
   
2. **Profile Settings**:
   - Update your full name and email

## Security Best Practices

1. **API Keys**: Use API keys with trading permissions only if you plan to use automated trading
2. **Password**: Use a strong, unique password for your account
3. **Session**: Log out when you're done using the application
4. **Verification**: Always verify trade details before execution

## Troubleshooting

### Common Issues

1. **Login Problems**:
   - Ensure your username and password are correct
   - Clear browser cache and try again
   
2. **Data Loading Issues**:
   - Click "Refresh Data" in the sidebar
   - Check your internet connection
   
3. **Trading Errors**:
   - Verify your API keys are correct and have appropriate permissions
   - Ensure you have sufficient balance for the trade

### Getting Help

If you encounter any issues or have questions, please contact support at:
support@example.com

## Updates and Maintenance

The application is regularly updated with new features and improvements. Check the changelog for the latest updates.

## Disclaimer

Trading cryptocurrencies involves significant risk. This application is provided for informational purposes only and should not be considered financial advice. Always do your own research before making investment decisions.
