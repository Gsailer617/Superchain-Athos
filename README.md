# Superchain Arbitrage Agent

An advanced AI-powered arbitrage bot for discovering and executing arbitrage opportunities across multiple DEXes on Base (Superchain).

## Features

### DEX Support
- Uniswap V3
- SushiSwap
- BaseSwap
- Aerodrome
- PancakeSwap
- SwapBased
- AlienBase
- Maverick
- SynthSwap
- Horizon DEX

### Core Features
- Real-time arbitrage opportunity detection
- Machine learning-based profit prediction
- Flash loan integration
- Risk management system
- Automated trading execution

### Enhanced LLM Integration
- Real-time opportunity analysis with HuggingFace models
- Market sentiment analysis and insights
- Performance metrics analysis
- Error analysis and recovery suggestions
- Trading strategy recommendations
- Dynamic market condition assessment

### Advanced Telegram Bot Features
- Real-time notifications with LLM-powered insights
- Interactive command system:
  - `/explain` - Get detailed analysis of latest opportunities
  - `/analyze` - Get performance metrics analysis
  - `/insight` - Get market condition insights
  - `/errors` - Get analysis of recent errors
  - `/performance` - View performance metrics
  - `/summary` - Get latest performance summary
  - `/status` - Check bot status
  - `/charts` - View performance charts
- Enhanced error reporting with recovery suggestions
- Performance tracking and visualization
- Rate-limited notifications to prevent spam
- Multi-admin support with role-based access

### Analytics & Monitoring
- Real-time performance tracking
- Interactive visualization dashboard
- Order book depth analysis
- Token performance analytics
- DEX performance metrics
- Gas price optimization
- Network health monitoring
- Risk assessment system

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/superchain-arbitrage-agent.git
cd superchain-arbitrage-agent
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your credentials:
```env
# Required Variables
TELEGRAM_BOT_TOKEN="your_bot_token"
TELEGRAM_CHAT_ID="your_chat_id"
TELEGRAM_ADMIN_IDS="comma,separated,admin,ids"
PRIVATE_KEY="your_wallet_private_key"
WEB3_PROVIDER_URI="your_web3_provider_uri"

# Optional Variables
LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
HF_API_KEY="your_huggingface_api_key"  # For LLM features
DEFILLAMA_API_KEY="your_defillama_api_key"
BASESCAN_API_KEY="your_basescan_api_key"
BASE_MAINNET_RPC="your_base_mainnet_rpc"
```

## Usage

1. Start the arbitrage agent:
```bash
python SuperchainArbitrageAgent.py
```

2. Access the visualization dashboard at `http://localhost:8050`

3. Interact with the Telegram bot:
- Start with `/startbot` command
- Use `/help` to see available commands
- Monitor notifications and insights
- View performance analytics
- Get LLM-powered analysis

## Configuration

### Risk Management
- Adjust risk parameters in `risk_settings`
- Configure slippage tolerance
- Set gas price limits
- Define profit thresholds
- Set position size limits

### DEX Settings
- Configure supported DEXes in `supported_dexes`
- Set fee tiers
- Define liquidity thresholds
- Configure price impact limits

### LLM Settings
- Configure model selection
- Adjust response parameters
- Set cache duration
- Define rate limits

### Telegram Bot Settings
- Configure notification preferences
- Set rate limits
- Define admin roles
- Customize alert thresholds

## Architecture

### Core Components
- Neural network model for opportunity prediction
- DEX data collection and analysis system
- Flash loan integration with multiple providers
- Real-time monitoring and execution engine

### Enhanced Features
- LLM-powered analysis system
- Advanced Telegram bot integration
- Interactive analytics dashboard
- Performance tracking system

### Safety Features
- Contract verification
- Multi-layer validation
- Risk assessment
- Slippage protection
- Gas optimization
- Error handling and recovery
- Rate limiting
- Duplicate transaction prevention

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Monitoring and Logging

### Logging Levels
- `DEBUG`: Detailed debugging information
- `INFO`: General operational information
- `WARNING`: Warning messages for potential issues
- `ERROR`: Error events that might still allow the program to run
- `CRITICAL`: Critical errors that may prevent proper operation

### Performance Monitoring
- Real-time trade tracking
- Gas usage optimization
- Error rate monitoring
- Success rate tracking
- Profit/loss tracking
- Network health monitoring

### Analytics Dashboard
- Real-time performance metrics
- Historical data analysis
- Token performance tracking
- DEX comparison tools
- Gas price trends
- Market sentiment analysis

## Support

For support, please open an issue in the repository or contact the development team through the provided channels.
