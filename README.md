# Crypto Trading Bot - Web Application

This directory contains the web application version of the Cryptocurrency Trading Bot. The web application provides an interactive interface for users to access the trading bot from any device with a web browser.

## Features

- **User Authentication**: Secure login and registration system
- **Interactive Dashboard**: Real-time market data visualization with customizable charts
- **Strategy Selection**: Choose from multiple trading strategies
- **Automated Trading**: Set up automated trading with risk management
- **Performance Analytics**: Track and analyze trading performance
- **Alerts and Notifications**: Receive alerts for trading signals and price movements
- **API Key Management**: Securely store and manage exchange API keys
- **Multi-Exchange Support**: Connect to multiple cryptocurrency exchanges

## Directory Structure

```
crypto_trading_website/
├── api/                    # API key management
├── auth/                   # Authentication system
├── data/                   # Data management
├── database/               # Database configuration
├── strategies/             # Trading strategies
├── trading/                # Trading functionality
├── alerts/                 # Alert system
├── analytics/              # Performance analytics
├── visualization/          # Chart visualization
├── config.py               # Configuration settings
├── web_app.py              # Main web application
└── requirements.txt        # Python dependencies
```

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional):

Create a `.env` file in the root directory with the following variables:

```
DB_TYPE=sqlite  # or postgres
DB_PATH=/path/to/database.db  # for SQLite
DB_HOST=localhost  # for PostgreSQL
DB_PORT=5432  # for PostgreSQL
DB_NAME=crypto_trading  # for PostgreSQL
DB_USER=postgres  # for PostgreSQL
DB_PASSWORD=postgres  # for PostgreSQL
ENCRYPTION_KEY=your_encryption_key
ENCRYPTION_SALT=your_encryption_salt
```

## Running the Application

To run the web application locally:

```bash
streamlit run web_app.py
```

This will start the Streamlit server and open the application in your default web browser.

## Deployment

The application can be deployed to various cloud platforms:

### Streamlit Cloud

1. Push the code to a GitHub repository
2. Connect your Streamlit Cloud account to your GitHub repository
3. Deploy the application with `web_app.py` as the main file

### Heroku

1. Create a `Procfile` with the following content:
   ```
   web: streamlit run web_app.py --server.port $PORT
   ```
2. Deploy to Heroku using the Heroku CLI or GitHub integration

### AWS

1. Set up an EC2 instance
2. Install the required dependencies
3. Run the application using a process manager like Supervisor

## Usage

1. Register a new account or log in with existing credentials
2. Configure your exchange API keys in the Settings tab
3. Select a trading pair, timeframe, and strategy
4. Monitor the market in the Market View tab
5. Analyze strategy performance in the Strategy Analysis tab
6. Set up automated trading or place manual trades in the Trading tab
7. Track your performance in the Performance tab

## Security

- API keys are encrypted using Fernet symmetric encryption
- Passwords are hashed using bcrypt
- Database connections are secured
- Authentication is handled by Streamlit Authenticator

## Monitoring and Maintenance

The application includes logging for monitoring and troubleshooting:

- Logs are stored in the `logs` directory
- Different log levels are available (INFO, WARNING, ERROR)
- Performance metrics are tracked for system health

## Support

For support or feature requests, please contact the development team or open an issue on the GitHub repository.
