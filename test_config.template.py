"""Template configuration settings for testing"""

# Test settings
TEST_CONFIG = {
    # Network settings
    'network': {
        'chain_id': 8453,  # Base Chain
        'rpc_url': 'https://mainnet.base.org',
        'ws_url': 'wss://mainnet.base.org',
        'block_time': 2,  # seconds
    },
    
    # Test wallet - REPLACE THESE VALUES
    'wallet': {
        'address': '<YOUR_WALLET_ADDRESS>',
        'private_key': '<YOUR_PRIVATE_KEY>'  # Never commit real private keys
    },
    
    # Test thresholds
    'thresholds': {
        'min_profit': 0.002,  # 0.2% minimum profit
        'max_slippage': 0.005,  # 0.5% maximum slippage
        'min_liquidity': 50000,  # Minimum liquidity in USD
        'max_gas': 500,  # Maximum gas in GWEI
    },
    
    # Test amounts
    'test_amounts': {
        'ETH': [0.1, 0.5, 1.0],
        'USDbC': [100, 500, 1000],
        'USDC': [100, 500, 1000],
    },
    
    # Telegram settings - REPLACE THESE VALUES
    'telegram': {
        'bot_token': '<YOUR_BOT_TOKEN>',
        'admin_ids': [123456789],  # Replace with your admin IDs
        'chat_id': -100123456789  # Replace with your chat ID
    },
    
    # Test timeouts
    'timeouts': {
        'transaction': 30,  # seconds
        'confirmation': 60,  # seconds
        'request': 10,  # seconds
    },
    
    # Test intervals
    'intervals': {
        'price_update': 5,  # seconds
        'opportunity_scan': 10,  # seconds
        'analytics_update': 60,  # seconds
    }
}

# Test data
TEST_TRADES = [
    {
        'pair': ('ETH', 'USDbC'),
        'amount': 1.0,
        'profit': 120.50,
        'gas_cost': 45.30,
        'success': True
    },
    {
        'pair': ('USDC', 'USDbC'),
        'amount': 1000,
        'profit': -20.15,
        'gas_cost': 35.20,
        'success': False
    }
]

# Test market data
TEST_MARKET_DATA = {
    'prices': {
        'ETH': 3500.00,
        'USDbC': 1.00,
        'USDC': 1.00
    },
    'liquidity': {
        'aerodrome': 1000000,
        'baseswap': 800000
    },
    'volume': {
        'aerodrome': 500000,
        'baseswap': 400000
    }
} 