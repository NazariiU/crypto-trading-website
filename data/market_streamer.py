"""
Real-time Market Data Streaming Module for Cryptocurrency Trading Bot

This module handles real-time market data streaming using WebSocket connections
to cryptocurrency exchanges. It provides continuous updates on price changes,
order book updates, and trade executions.
"""

import os
import sys
import json
import time
import logging
import threading
import websocket
from typing import Dict, List, Callable, Any, Optional

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger('crypto_trading_bot.data.streaming')

class WebSocketManager:
    """Base class for managing WebSocket connections to exchanges"""
    
    def __init__(self, exchange_id: str):
        """
        Initialize WebSocket manager
        
        Args:
            exchange_id: Exchange identifier (e.g., 'binance', 'okx', 'bybit')
        """
        self.exchange_id = exchange_id
        self.ws = None
        self.is_connected = False
        self.callbacks = {}
        self.reconnect_count = 0
        self.max_reconnects = 5
        self.reconnect_delay = 5  # seconds
        
        # Get exchange-specific configuration
        self.exchange_config = config.EXCHANGE_CONFIGS.get(exchange_id, {})
        
        # Determine WebSocket URL based on exchange and testnet setting
        self.ws_url = self._get_websocket_url()
        
        logger.info(f"Initialized WebSocketManager for {exchange_id}")
    
    def _get_websocket_url(self) -> str:
        """Get WebSocket URL for the exchange"""
        # Default URLs for supported exchanges
        urls = {
            'binance': {
                'main': 'wss://stream.binance.com:9443/ws',
                'test': 'wss://testnet.binance.vision/ws'
            },
            'okx': {
                'main': 'wss://ws.okx.com:8443/ws/v5/public',
                'test': 'wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999'
            },
            'bybit': {
                'main': 'wss://stream.bybit.com/v5/public',
                'test': 'wss://stream-testnet.bybit.com/v5/public'
            }
        }
        
        # Get URL based on exchange and testnet setting
        if self.exchange_id in urls:
            use_testnet = self.exchange_config.get('testnet', False)
            url_key = 'test' if use_testnet else 'main'
            return urls[self.exchange_id][url_key]
        else:
            raise ValueError(f"Unsupported exchange for WebSocket: {self.exchange_id}")
    
    def connect(self):
        """Establish WebSocket connection"""
        websocket.enableTrace(False)
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # Start WebSocket connection in a separate thread
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        logger.info(f"Started WebSocket connection to {self.exchange_id}")
    
    def _on_open(self, ws):
        """Callback when WebSocket connection is opened"""
        self.is_connected = True
        self.reconnect_count = 0
        logger.info(f"WebSocket connection to {self.exchange_id} established")
        
        # Subscribe to channels
        self._subscribe_channels()
    
    def _on_message(self, ws, message):
        """Callback when WebSocket message is received"""
        try:
            data = json.loads(message)
            
            # Process message based on exchange-specific format
            processed_data = self._process_message(data)
            
            # Call registered callbacks with processed data
            if processed_data:
                event_type = processed_data.get('event_type')
                if event_type and event_type in self.callbacks:
                    for callback in self.callbacks[event_type]:
                        callback(processed_data)
        
        except json.JSONDecodeError:
            logger.error(f"Failed to decode WebSocket message: {message}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Callback when WebSocket error occurs"""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Callback when WebSocket connection is closed"""
        self.is_connected = False
        logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        
        # Attempt to reconnect
        if self.reconnect_count < self.max_reconnects:
            self.reconnect_count += 1
            reconnect_time = self.reconnect_delay * self.reconnect_count
            logger.info(f"Attempting to reconnect in {reconnect_time} seconds (attempt {self.reconnect_count}/{self.max_reconnects})")
            
            time.sleep(reconnect_time)
            self.connect()
    
    def _subscribe_channels(self):
        """Subscribe to WebSocket channels (to be implemented by subclasses)"""
        pass
    
    def _process_message(self, data: Dict) -> Optional[Dict]:
        """Process WebSocket message (to be implemented by subclasses)"""
        pass
    
    def register_callback(self, event_type: str, callback: Callable[[Dict], None]):
        """
        Register callback for specific event type
        
        Args:
            event_type: Event type (e.g., 'ticker', 'trade', 'orderbook')
            callback: Callback function to be called when event occurs
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
        logger.info(f"Registered callback for {event_type} events")
    
    def unregister_callback(self, event_type: str, callback: Callable[[Dict], None]):
        """
        Unregister callback for specific event type
        
        Args:
            event_type: Event type (e.g., 'ticker', 'trade', 'orderbook')
            callback: Callback function to unregister
        """
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
            logger.info(f"Unregistered callback for {event_type} events")
    
    def close(self):
        """Close WebSocket connection"""
        if self.ws:
            self.ws.close()
            logger.info(f"Closed WebSocket connection to {self.exchange_id}")


class BinanceWebSocketManager(WebSocketManager):
    """WebSocket manager for Binance exchange"""
    
    def __init__(self):
        """Initialize Binance WebSocket manager"""
        super().__init__('binance')
        self.subscribed_symbols = []
    
    def subscribe_ticker(self, symbol: str):
        """
        Subscribe to ticker updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
        """
        # Convert symbol format from CCXT to Binance (BTC/USDT -> btcusdt)
        formatted_symbol = symbol.replace('/', '').lower()
        
        if self.is_connected:
            # Subscribe to ticker stream
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{formatted_symbol}@ticker"],
                "id": int(time.time())
            }
            self.ws.send(json.dumps(subscribe_msg))
            
            # Add to subscribed symbols
            if symbol not in self.subscribed_symbols:
                self.subscribed_symbols.append(symbol)
            
            logger.info(f"Subscribed to Binance ticker updates for {symbol}")
    
    def subscribe_kline(self, symbol: str, interval: str = '1m'):
        """
        Subscribe to kline (candlestick) updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
        """
        # Convert symbol format from CCXT to Binance
        formatted_symbol = symbol.replace('/', '').lower()
        
        if self.is_connected:
            # Subscribe to kline stream
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{formatted_symbol}@kline_{interval}"],
                "id": int(time.time())
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to Binance kline updates for {symbol} ({interval})")
    
    def subscribe_trades(self, symbol: str):
        """
        Subscribe to trade updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
        """
        # Convert symbol format from CCXT to Binance
        formatted_symbol = symbol.replace('/', '').lower()
        
        if self.is_connected:
            # Subscribe to trade stream
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{formatted_symbol}@trade"],
                "id": int(time.time())
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to Binance trade updates for {symbol}")
    
    def subscribe_depth(self, symbol: str, level: str = '10'):
        """
        Subscribe to order book updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            level: Order book depth (e.g., '5', '10', '20')
        """
        # Convert symbol format from CCXT to Binance
        formatted_symbol = symbol.replace('/', '').lower()
        
        if self.is_connected:
            # Subscribe to depth stream
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{formatted_symbol}@depth{level}"],
                "id": int(time.time())
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to Binance order book updates for {symbol} (depth {level})")
    
    def _subscribe_channels(self):
        """Subscribe to channels for previously added symbols"""
        for symbol in self.subscribed_symbols:
            self.subscribe_ticker(symbol)
    
    def _process_message(self, data: Dict) -> Optional[Dict]:
        """Process Binance WebSocket message"""
        # Check if it's a ticker event
        if 'e' in data and data['e'] == 'ticker':
            return {
                'event_type': 'ticker',
                'exchange': 'binance',
                'symbol': data['s'],
                'timestamp': data['E'],
                'last_price': float(data['c']),
                'open_price': float(data['o']),
                'high_price': float(data['h']),
                'low_price': float(data['l']),
                'volume': float(data['v']),
                'quote_volume': float(data['q']),
                'change': float(data['p']),
                'change_percent': float(data['P'])
            }
        
        # Check if it's a kline event
        elif 'e' in data and data['e'] == 'kline':
            k = data['k']
            return {
                'event_type': 'kline',
                'exchange': 'binance',
                'symbol': data['s'],
                'timestamp': data['E'],
                'interval': k['i'],
                'open_time': k['t'],
                'close_time': k['T'],
                'open': float(k['o']),
                'high': float(k['h']),
                'low': float(k['l']),
                'close': float(k['c']),
                'volume': float(k['v']),
                'is_closed': k['x']
            }
        
        # Check if it's a trade event
        elif 'e' in data and data['e'] == 'trade':
            return {
                'event_type': 'trade',
                'exchange': 'binance',
                'symbol': data['s'],
                'timestamp': data['E'],
                'trade_id': data['t'],
                'price': float(data['p']),
                'quantity': float(data['q']),
                'buyer_order_id': data['b'],
                'seller_order_id': data['a'],
                'trade_time': data['T'],
                'is_buyer_maker': data['m']
            }
        
        # Check if it's a depth event
        elif 'lastUpdateId' in data:
            return {
                'event_type': 'orderbook',
                'exchange': 'binance',
                'symbol': data.get('s', ''),
                'timestamp': int(time.time() * 1000),
                'last_update_id': data['lastUpdateId'],
                'bids': [[float(price), float(qty)] for price, qty in data['bids']],
                'asks': [[float(price), float(qty)] for price, qty in data['asks']]
            }
        
        return None


class OKXWebSocketManager(WebSocketManager):
    """WebSocket manager for OKX exchange"""
    
    def __init__(self):
        """Initialize OKX WebSocket manager"""
        super().__init__('okx')
        self.subscribed_symbols = []
    
    def subscribe_ticker(self, symbol: str):
        """
        Subscribe to ticker updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
        """
        # Convert symbol format from CCXT to OKX (BTC/USDT -> BTC-USDT)
        formatted_symbol = symbol.replace('/', '-')
        
        if self.is_connected:
            # Subscribe to ticker stream
            subscribe_msg = {
                "op": "subscribe",
                "args": [{
                    "channel": "tickers",
                    "instId": formatted_symbol
                }]
            }
            self.ws.send(json.dumps(subscribe_msg))
            
            # Add to subscribed symbols
            if symbol not in self.subscribed_symbols:
                self.subscribed_symbols.append(symbol)
            
            logger.info(f"Subscribed to OKX ticker updates for {symbol}")
    
    def subscribe_kline(self, symbol: str, interval: str = '1m'):
        """
        Subscribe to kline (candlestick) updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
        """
        # Convert symbol format from CCXT to OKX
        formatted_symbol = symbol.replace('/', '-')
        
        # Convert interval format (1m -> 1M, 1h -> 1H)
        if interval.endswith('m'):
            okx_interval = interval.upper().replace('M', 'M')
        elif interval.endswith('h'):
            okx_interval = interval.upper().replace('H', 'H')
        elif interval.endswith('d'):
            okx_interval = interval.upper().replace('D', 'D')
        else:
            okx_interval = interval.upper()
        
        if self.is_connected:
            # Subscribe to kline stream
            subscribe_msg = {
                "op": "subscribe",
                "args": [{
                    "channel": "candle" + okx_interval,
                    "instId": formatted_symbol
                }]
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to OKX kline updates for {symbol} ({interval})")
    
    def subscribe_trades(self, symbol: str):
        """
        Subscribe to trade updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
        """
        # Convert symbol format from CCXT to OKX
        formatted_symbol = symbol.replace('/', '-')
        
        if self.is_connected:
            # Subscribe to trade stream
            subscribe_msg = {
                "op": "subscribe",
                "args": [{
                    "channel": "trades",
                    "instId": formatted_symbol
                }]
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to OKX trade updates for {symbol}")
    
    def subscribe_depth(self, symbol: str, depth: str = '5'):
        """
        Subscribe to order book updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            depth: Order book depth (e.g., '5', '10', '20')
        """
        # Convert symbol format from CCXT to OKX
        formatted_symbol = symbol.replace('/', '-')
        
        if self.is_connected:
            # Subscribe to depth stream
            subscribe_msg = {
                "op": "subscribe",
                "args": [{
                    "channel": "books" + depth,
                    "instId": formatted_symbol
                }]
            }
            self.ws.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to OKX order book updates for {symbol} (depth {depth})")
    
    def _subscribe_channels(self):
        """Subscribe to channels for previously added symbols"""
        for symbol in self.subscribed_symbols:
            self.subscribe_ticker(symbol)
    
    def _process_message(self, data: Dict) -> Optional[Dict]:
        """Process OKX WebSocket message"""
        # Check if it's a ticker event
        if 'data' in data and data.get('arg', {}).get('channel') == 'tickers':
            ticker_data = data['data'][0]
            return {
                'event_type': 'ticker',
                'exchange': 'okx',
                'symbol': ticker_data['instId'].replace('-', '/'),
                'timestamp': int(ticker_data['ts']),
                'last_price': float(ticker_data['last']),
                'open_price': float(ticker_data['open24h']),
                'high_price': float(ticker_data['high24h']),
                'low_price': float(ticker_data['low24h']),
                'volume': float(ticker_data['vol24h']),
                'quote_volume': float(ticker_data['volCcy24h']),
                'change': float(ticker_data['last']) - float(ticker_data['open24h']),
                'change_percent': float(ticker_data['change24h'])
            }
        
        # Check if it's a kline event
        elif 'data' in data and 'candle' in data.get('arg', {}).get('channel', ''):
            candle_data = data['data'][0]
            return {
                'event_type': 'kline',
                'exchange': 'okx',
                'symbol': data['arg']['instId'].replace('-', '/'),
                'timestamp': int(candle_data[0]),
                'interval': data['arg']['channel'].replace('candle', '').lower(),
                'open': float(candle_data[1]),
                'high': float(candle_data[2]),
                'low': float(candle_data[3]),
                'close': float(candle_data[4]),
                'volume': float(candle_data[5]),
                'is_closed': True
            }
        
        # Check if it's a trade event
        elif 'data' in data and data.get('arg', {}).get('channel') == 'trades':
            trade_data = data['data'][0]
            return {
                'event_type': 'trade',
                'exchange': 'okx',
                'symbol': data['arg']['instId'].replace('-', '/'),
                'timestamp': int(trade_data['ts']),
                'trade_id': trade_data['tradeId'],
                'price': float(trade_data['px']),
                'quantity': float(trade_data['sz']),
                'side': trade_data['side'],
                'trade_time': int(trade_data['ts'])
            }
        
        # Check if it's a depth event
        elif 'data' in data and 'books' in data.get('arg', {}).get('channel', ''):
            book_data = data['data'][0]
            return {
                'event_type': 'orderbook',
                'exchange': 'okx',
                'symbol': data['arg']['instId'].replace('-', '/'),
                'timestamp': int(book_data['ts']),
                'bids': [[float(price), float(qty)] for price, qty, _, _ in book_data.get('bids', [])],
                'asks': [[float(price), float(qty)] for price, qty, _, _ in book_data.get('asks', [])]
            }
        
        return None


class MarketDataStreamer:
    """Manager class for handling real-time market data streaming"""
    
    def __init__(self, exchange_id: str = None):
        """
        Initialize market data streamer
        
        Args:
            exchange_id: Exchange identifier (default from config)
        """
        self.exchange_id = exchange_id or config.DEFAULT_EXCHANGE
        self.ws_manager = self._create_ws_manager()
        self.active_streams = {}
        logger.info(f"Initialized MarketDataStreamer for {self.exchange_id}")
    
    def _create_ws_manager(self):
        """Create WebSocket manager for the specified exchange"""
        if self.exchange_id == 'binance':
            return BinanceWebSocketManager()
        elif self.exchange_id == 'okx':
            return OKXWebSocketManager()
        else:
            raise ValueError(f"Unsupported exchange for WebSocket streaming: {self.exchange_id}")
    
    def start(self):
        """Start market data streaming"""
        self.ws_manager.connect()
    
    def subscribe_ticker(self, symbol: str, callback: Callable[[Dict], None]):
        """
        Subscribe to ticker updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            callback: Callback function to be called when ticker updates are received
        """
        # Register callback
        self.ws_manager.register_callback('ticker', callback)
        
        # Subscribe to ticker updates
        self.ws_manager.subscribe_ticker(symbol)
        
        # Track active stream
        stream_key = f"ticker_{symbol}"
        if stream_key not in self.active_streams:
            self.active_streams[stream_key] = []
        self.active_streams[stream_key].append(callback)
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable[[Dict], None]):
        """
        Subscribe to kline (candlestick) updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            interval: Kline interval (e.g., '1m', '5m', '1h')
            callback: Callback function to be called when kline updates are received
        """
        # Register callback
        self.ws_manager.register_callback('kline', callback)
        
        # Subscribe to kline updates
        self.ws_manager.subscribe_kline(symbol, interval)
        
        # Track active stream
        stream_key = f"kline_{symbol}_{interval}"
        if stream_key not in self.active_streams:
            self.active_streams[stream_key] = []
        self.active_streams[stream_key].append(callback)
    
    def subscribe_trades(self, symbol: str, callback: Callable[[Dict], None]):
        """
        Subscribe to trade updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            callback: Callback function to be called when trade updates are received
        """
        # Register callback
        self.ws_manager.register_callback('trade', callback)
        
        # Subscribe to trade updates
        self.ws_manager.subscribe_trades(symbol)
        
        # Track active stream
        stream_key = f"trades_{symbol}"
        if stream_key not in self.active_streams:
            self.active_streams[stream_key] = []
        self.active_streams[stream_key].append(callback)
    
    def subscribe_orderbook(self, symbol: str, depth: str, callback: Callable[[Dict], None]):
        """
        Subscribe to order book updates for a symbol
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            depth: Order book depth (e.g., '5', '10', '20')
            callback: Callback function to be called when order book updates are received
        """
        # Register callback
        self.ws_manager.register_callback('orderbook', callback)
        
        # Subscribe to order book updates
        self.ws_manager.subscribe_depth(symbol, depth)
        
        # Track active stream
        stream_key = f"orderbook_{symbol}_{depth}"
        if stream_key not in self.active_streams:
            self.active_streams[stream_key] = []
        self.active_streams[stream_key].append(callback)
    
    def unsubscribe(self, stream_type: str, symbol: str, callback: Callable[[Dict], None] = None):
        """
        Unsubscribe from market data stream
        
        Args:
            stream_type: Stream type (e.g., 'ticker', 'kline', 'trades', 'orderbook')
            symbol: Trading symbol (e.g., 'BTC/USDT')
            callback: Callback function to unregister (if None, unregister all callbacks)
        """
        stream_key = f"{stream_type}_{symbol}"
        
        if stream_key in self.active_streams:
            if callback:
                # Unregister specific callback
                if callback in self.active_streams[stream_key]:
                    self.active_streams[stream_key].remove(callback)
                    self.ws_manager.unregister_callback(stream_type, callback)
            else:
                # Unregister all callbacks
                for cb in self.active_streams[stream_key]:
                    self.ws_manager.unregister_callback(stream_type, cb)
                self.active_streams[stream_key] = []
            
            logger.info(f"Unsubscribed from {stream_type} updates for {symbol}")
    
    def close(self):
        """Close all WebSocket connections"""
        if self.ws_manager:
            self.ws_manager.close()
            logger.info("Closed all WebSocket connections")


# Factory function to create market data streamer for specific exchange
def create_market_data_streamer(exchange_id: str = None) -> MarketDataStreamer:
    """
    Create a market data streamer for a specific exchange
    
    Args:
        exchange_id: Exchange identifier (default from config)
        
    Returns:
        MarketDataStreamer instance
    """
    return MarketDataStreamer(exchange_id)


# Example callback function for ticker updates
def ticker_callback(data: Dict):
    """Example callback function for ticker updates"""
    print(f"Ticker update for {data['symbol']}: {data['last_price']} ({data['change_percent']}%)")


# Test function to verify the module works correctly
def test_market_data_streaming():
    """Test market data streaming functionality"""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create market data streamer
    streamer = create_market_data_streamer('binance')
    
    # Start streaming
    streamer.start()
    
    # Wait for connection to establish
    time.sleep(2)
    
    # Subscribe to ticker updates for BTC/USDT
    streamer.subscribe_ticker('BTC/USDT', ticker_callback)
    
    try:
        # Keep running for 30 seconds
        print("Receiving market data for 30 seconds...")
        time.sleep(30)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Close connections
        streamer.close()


if __name__ == "__main__":
    # Run test
    test_market_data_streaming()
