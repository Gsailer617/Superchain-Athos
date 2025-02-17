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

# Supported Blockchains

This project supports multiple blockchains for cross-chain arbitrage opportunities. Below are the details for each supported chain:

## Layer 1 Chains

### Ethereum
- **Chain ID**: 1
- **Block Time**: ~12 seconds
- **Gas Model**: EIP-1559
- **RPC Endpoint**: `ETHEREUM_RPC_URL` in .env
- **Documentation**: [Ethereum Docs](https://ethereum.org/en/developers/docs/)
- **Special Notes**: 
  - Base layer for most L2s
  - Highest security but also highest gas fees
  - MEV protection via Flashbots

### Polygon
- **Chain ID**: 137
- **Block Time**: ~2 seconds
- **Gas Model**: EIP-1559
- **RPC Endpoint**: `POLYGON_RPC_URL` in .env
- **Documentation**: [Polygon Docs](https://docs.polygon.technology/tools/)
- **Special Notes**:
  - High throughput, low fees
  - Requires monitoring MATIC token price for gas
  - Multiple bridge options (PoS, Plasma)

### BNB Chain
- **Chain ID**: 56
- **Block Time**: ~3 seconds
- **Gas Model**: Legacy
- **RPC Endpoint**: `BNB_RPC_URL` in .env
- **Documentation**: [BNB Chain Docs](https://docs.bnbchain.org/bnb-smart-chain/)
- **Special Notes**:
  - High throughput
  - Centralized validator set
  - Multiple RPC endpoints recommended

### Avalanche
- **Chain ID**: 43114
- **Block Time**: ~2 seconds
- **Gas Model**: EIP-1559
- **RPC Endpoint**: `AVALANCHE_RPC_URL` in .env
- **Documentation**: [Avalanche Docs](https://build.avax.network/docs)
- **Special Notes**:
  - C-Chain for EVM compatibility
  - Fast finality
  - Subnet support

### Gnosis
- **Chain ID**: 100
- **Block Time**: ~5 seconds
- **Gas Model**: EIP-1559
- **RPC Endpoint**: `GNOSIS_RPC_URL` in .env
- **Documentation**: [Gnosis Docs](https://docs.gnosischain.com/developers/Overview)
- **Special Notes**:
  - xDAI as native token
  - Stable token focus
  - OmniBridge support

## Layer 2 Chains

### Base
- **Chain ID**: 8453
- **Block Time**: ~2 seconds
- **Gas Model**: Optimistic
- **RPC Endpoint**: `BASE_RPC_URL` in .env
- **Documentation**: [Base Docs](https://docs.base.org/docs/)
- **Special Notes**:
  - Optimistic rollup
  - 7-day withdrawal period
  - Coinbase integration

### Arbitrum
- **Chain ID**: 42161
- **Block Time**: ~250ms
- **Gas Model**: Arbitrum-specific
- **RPC Endpoint**: `ARBITRUM_RPC_URL` in .env
- **Documentation**: [Arbitrum Docs](https://docs.arbitrum.io/welcome/get-started)
- **Special Notes**:
  - Nitro upgrade
  - Complex gas pricing
  - High throughput

### Optimism
- **Chain ID**: 10
- **Block Time**: ~2 seconds
- **Gas Model**: Optimistic
- **RPC Endpoint**: `OPTIMISM_RPC_URL` in .env
- **Documentation**: [Optimism Docs](https://docs.optimism.io/)
- **Special Notes**:
  - Bedrock upgrade
  - Similar to Ethereum
  - Strong infrastructure

### Linea
- **Chain ID**: 59144
- **Block Time**: ~12 seconds
- **Gas Model**: EIP-1559
- **RPC Endpoint**: `LINEA_RPC_URL` in .env
- **Documentation**: [Linea Docs](https://docs.linea.build/get-started)
- **Special Notes**:
  - ZK rollup
  - ConsenSys backing
  - ZK proof verification times

### Mantle
- **Chain ID**: 5000
- **Block Time**: ~2 seconds
- **Gas Model**: Optimistic
- **RPC Endpoint**: `MANTLE_RPC_URL` in .env
- **Documentation**: [Mantle Docs](https://docs.mantle.xyz/network)
- **Special Notes**:
  - Optimistic rollup
  - Data availability
  - MNT token incentives

### Mode
- **Chain ID**: 34443
- **Block Time**: ~2 seconds
- **Gas Model**: Mode-specific
- **RPC Endpoint**: `MODE_RPC_URL` in .env
- **Documentation**: [Mode Docs](https://docs.mode.network/)
- **Special Notes**:
  - Fixed priority fees
  - Sequencer infrastructure
  - Optimized for DeFi

### Sonic
- **Chain ID**: 8899
- **Block Time**: ~1 second
- **Gas Model**: Sonic-specific
- **RPC Endpoint**: `SONIC_RPC_URL` in .env
- **Documentation**: [Sonic Docs](https://docs.soniclabs.com/sonic/build-on-sonic/getting-started)
- **Special Notes**:
  - High performance
  - Low latency
  - Fixed gas fees

## Bridge Support

Each chain pair has specific bridge configurations and adapters:

- **Canonical Bridges**: Official bridges between L1-L2 pairs
- **Third-party Bridges**: Supported bridges include:
  - LayerZero
  - deBridge
  - Across
  - Superbridge
  - OmniBridge (Gnosis)
  - Plasma and PoS (Polygon)

## Configuration

All chain configurations are managed through environment variables. See `.env.example` for required variables.

### Required Environment Variables

```bash
# RPC URLs
ETHEREUM_RPC_URL=
BASE_RPC_URL=
POLYGON_RPC_URL=
ARBITRUM_RPC_URL=
OPTIMISM_RPC_URL=
BNB_RPC_URL=
LINEA_RPC_URL=
MANTLE_RPC_URL=
AVALANCHE_RPC_URL=
GNOSIS_RPC_URL=
MODE_RPC_URL=
SONIC_RPC_URL=

# Bridge Contract Addresses
MODE_L1_BRIDGE=
MODE_L2_BRIDGE=
MODE_MESSAGE_SERVICE=
SONIC_BRIDGE_ROUTER=
SONIC_TOKEN_BRIDGE=
SONIC_LIQUIDITY_POOL=
# ... other bridge addresses
```

## Gas Management

Each chain has specific gas management requirements:

- **EIP-1559 Chains**: Use base fee + priority fee
- **Legacy Chains**: Use gas price only
- **L2-specific**: Use custom gas models (Arbitrum, Optimism)
- **Fixed Fee Chains**: Use predefined fees (Mode, Sonic)

## Performance Considerations

- Monitor block times and confirmation requirements
- Account for bridge-specific delays
- Consider gas token prices (ETH, MATIC, BNB, etc.)
- Watch for network congestion
- Monitor bridge liquidity

## Adding New Chains

To add support for a new blockchain:

1. Add chain configuration to `src/config/chain_configurations.py`
2. Implement bridge adapter if needed
3. Update gas management for chain-specific models
4. Add chain-specific tests
5. Update environment variables
6. Test thoroughly on testnet before mainnet

## Troubleshooting

Common issues and solutions:

### RPC Connection Issues
- Verify RPC URL is correct and accessible
- Check for rate limiting
- Use fallback RPC providers

### Gas Estimation Failures
- Check chain-specific gas models
- Verify gas token balance
- Monitor network congestion

### Bridge Transfer Issues
- Verify bridge contract addresses
- Check bridge liquidity
- Monitor bridge status
- Verify message proofs

### Transaction Failures
- Check nonce management
- Verify gas settings
- Monitor chain state

## Resources

- [Ethereum Gas Tracker](https://etherscan.io/gastracker)
- [L2 Gas Fees](https://l2fees.info/)
- [Bridge Monitor](https://bridges.llama.fi/)
- [Chain Analytics](https://defillama.com/)

For more detailed information, refer to each chain's official documentation linked above.
