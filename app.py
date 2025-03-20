"""
Main application entry point for Cryptocurrency Trading Bot
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
import config
from data.data_manager import create_data_manager
from strategies.technical_indicators import create_technical_analyzer
from strategies.trading_strategies import create_strategy_manager
from trading.trading_bot import create_trading_bot
from alerts.notification_manager import create_notification_manager
from analytics.performance_analyzer import create_performance_analyzer
from visualization.dashboard import run_dashboard

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOGGING['level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGGING['file']),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('crypto_trading_bot')

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Cryptocurrency Trading Bot')
    
    parser.add_argument(
        '--mode',
        choices=['dashboard', 'trading', 'backtest'],
        default='dashboard',
        help='Operation mode (default: dashboard)'
    )
    
    parser.add_argument(
        '--trading-mode',
        choices=['auto', 'signal'],
        default='signal',
        help='Trading mode (default: signal)'
    )
    
    parser.add_argument(
        '--exchange',
        default=config.DEFAULT_EXCHANGE,
        help=f'Exchange to use (default: {config.DEFAULT_EXCHANGE})'
    )
    
    parser.add_argument(
        '--symbol',
        default=config.TRADING_PARAMS['default_symbols'][0],
        help=f'Trading symbol (default: {config.TRADING_PARAMS["default_symbols"][0]})'
    )
    
    parser.add_argument(
        '--strategy',
        default=config.TRADING_PARAMS['default_strategies'][0],
        help=f'Trading strategy (default: {config.TRADING_PARAMS["default_strategies"][0]})'
    )
    
    parser.add_argument(
        '--timeframe',
        default=config.TRADING_PARAMS['default_timeframes'][3],
        help=f'Timeframe (default: {config.TRADING_PARAMS["default_timeframes"][3]})'
    )
    
    return parser.parse_args()

def run_trading_bot(args):
    """Run trading bot"""
    logger.info(f"Starting trading bot in {args.trading_mode} mode")
    
    # Create trading bot
    bot = create_trading_bot(args.exchange, args.trading_mode)
    
    # Configure bot
    bot.set_strategy(args.strategy)
    bot.set_timeframe(args.timeframe)
    bot.add_symbol(args.symbol)
    
    try:
        # Start bot
        bot.start()
        
        # Keep running until interrupted
        logger.info("Trading bot is running. Press Ctrl+C to stop.")
        while True:
            pass
            
    except KeyboardInterrupt:
        logger.info("Stopping trading bot...")
    finally:
        # Stop bot
        bot.stop()
        logger.info("Trading bot stopped")

def run_backtest(args):
    """Run backtest"""
    logger.info(f"Starting backtest for {args.symbol} using {args.strategy} strategy")
    
    # Create data manager
    data_manager = create_data_manager(args.exchange)
    
    # Create strategy manager
    strategy_manager = create_strategy_manager()
    
    # Get strategy
    strategy = strategy_manager.get_strategy(args.strategy)
    
    if not strategy:
        logger.error(f"Strategy {args.strategy} not found")
        return
    
    # Create performance analyzer
    analyzer = create_performance_analyzer()
    
    try:
        # Fetch historical data
        data = data_manager.fetch_historical_data(args.symbol, args.timeframe, days_back=30)
        
        if data.empty:
            logger.error(f"No data available for {args.symbol}")
            return
        
        # Run backtest
        backtest_results = strategy.backtest(data)
        
        # Print backtest results
        logger.info(f"Backtest results for {args.symbol} using {args.strategy} strategy:")
        logger.info(f"Initial balance: ${backtest_results['initial_balance']:.2f}")
        logger.info(f"Final balance: ${backtest_results['final_balance']:.2f}")
        logger.info(f"Profit/Loss: ${backtest_results['profit_loss']:.2f} ({backtest_results['profit_loss_percent']:.2f}%)")
        logger.info(f"Total trades: {backtest_results['total_trades']}")
        logger.info(f"Win rate: {backtest_results['win_rate'] * 100:.2f}%")
        logger.info(f"Max drawdown: ${backtest_results['max_drawdown']:.2f} ({backtest_results['max_drawdown_percent']:.2f}%)")
        
        # Generate performance report
        report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports', 
                                f"{args.symbol.replace('/', '_')}_{args.strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        # Log trades for analysis
        for trade in backtest_results['trades']:
            if trade.get('type') == 'buy':
                # Log entry
                trade_id = analyzer.log_trade({
                    'symbol': args.symbol,
                    'strategy': args.strategy,
                    'side': 'buy',
                    'entry_time': trade.get('timestamp'),
                    'entry_price': trade.get('price'),
                    'amount': trade.get('amount'),
                    'status': 'open',
                    'stop_loss': trade.get('stop_loss'),
                    'take_profit': trade.get('take_profit')
                })
            elif trade.get('type') == 'sell' and trade.get('pnl') is not None:
                # Find corresponding buy trade
                buy_trades = analyzer.get_trades(
                    symbol=args.symbol,
                    strategy=args.strategy,
                    status='open',
                    limit=1
                )
                
                if buy_trades:
                    # Update trade with exit information
                    analyzer.update_trade(buy_trades[0]['id'], {
                        'exit_time': trade.get('timestamp'),
                        'exit_price': trade.get('price'),
                        'pnl': trade.get('pnl'),
                        'pnl_percent': trade.get('pnl_percent'),
                        'status': 'closed'
                    })
        
        # Generate report
        report = analyzer.generate_report(
            symbol=args.symbol,
            strategy=args.strategy,
            output_dir=report_dir
        )
        
        logger.info(f"Performance report generated: {report.get('report_path')}")
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
    finally:
        # Close data manager
        data_manager.close()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Run in selected mode
    if args.mode == 'dashboard':
        logger.info("Starting dashboard")
        run_dashboard()
    elif args.mode == 'trading':
        run_trading_bot(args)
    elif args.mode == 'backtest':
        run_backtest(args)
    else:
        logger.error(f"Invalid mode: {args.mode}")

if __name__ == "__main__":
    main()
