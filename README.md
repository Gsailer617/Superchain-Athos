# Flashing Base - DeFi Automation System

A comprehensive DeFi automation system for arbitrage opportunity discovery, yield farming management, and portfolio optimization.

## Overview

Flashing Base is a powerful DeFi automation platform that combines arbitrage opportunity discovery with yield farming management to maximize returns while managing risk. The system uses advanced algorithms to identify profitable trading opportunities across multiple chains and protocols, and provides tools for portfolio optimization and performance tracking.

## Features

### Arbitrage

- **Multi-DEX Arbitrage**: Identify price discrepancies across multiple decentralized exchanges
- **Flash Loan Arbitrage**: Execute arbitrage opportunities using flash loans for capital efficiency
- **Triangular Arbitrage**: Find profitable trading paths between multiple tokens
- **Cross-Chain Arbitrage**: Discover opportunities across different blockchain networks
- **Risk Assessment**: Evaluate opportunities based on risk factors and expected profitability
- **Automated Execution**: Execute trades automatically with configurable parameters

### Yield Farming

- **Opportunity Discovery**: Find the best yield farming opportunities across protocols
- **Position Management**: Create, monitor, and manage yield farming positions
- **Auto-Compounding**: Automatically compound rewards for maximum returns
- **Reward Harvesting**: Harvest rewards on optimal schedules
- **Risk Assessment**: Evaluate protocols and pools based on security and stability

### Portfolio Management

- **Portfolio Allocation**: Get recommendations for optimal allocation between arbitrage, yield farming, and reserves
- **Risk Profiles**: Choose from conservative, moderate, or aggressive risk profiles
- **Performance Tracking**: Monitor returns, profits, and performance metrics
- **Portfolio Optimization**: Rebalance portfolio based on changing market conditions
- **Multi-Chain Support**: Manage assets across multiple blockchain networks

## Architecture

The system is built with a modular architecture using the Command Query Responsibility Segregation (CQRS) pattern and Domain-Driven Design principles:

- **Core**: Foundational components including dependency injection, health monitoring, and configuration
- **CQRS**: Command and query handling, event processing, and bulkhead pattern implementation
- **Domain Models**: Rich domain models for tokens, arbitrage opportunities, and yield positions
- **Services**: Business logic for arbitrage, yield farming, and portfolio management
- **API**: RESTful API for integration with external systems
- **CLI**: Command-line interface for easy interaction

## Getting Started

### Prerequisites

- Python 3.9+
- Redis
- Ethereum node access (Infura, Alchemy, or self-hosted)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/flashing-base.git
   cd flashing-base
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   ```
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Initialize the system:
   ```
   python -m src.main
   ```

### Using the API

The system provides a RESTful API for integration with external systems:

```
# Start the API server
python -m src.api.defi_api
```

The API will be available at `http://localhost:8000`. See the API documentation for available endpoints.

### Using the CLI

The system provides a command-line interface for easy interaction:

```
# List DeFi opportunities
python -m src.cli.defi_cli opportunities list --wallet 0x123...

# Get portfolio allocation
python -m src.cli.defi_cli portfolio allocation --wallet 0x123... --value 10000

# Execute an arbitrage opportunity
python -m src.cli.defi_cli arbitrage execute --id arb-123 --wallet 0x123...

# Create a yield farming position
python -m src.cli.defi_cli yield create --id yield-123 --wallet 0x123... --deposit "0xtoken:100"
```

## Configuration

The system can be configured using environment variables or a configuration file:

- `CHAINS`: Comma-separated list of chain IDs to monitor (default: "1,56,137")
- `MIN_PROFIT_USD`: Minimum profit threshold for arbitrage opportunities in USD (default: 10)
- `MAX_GAS_PRICE_GWEI`: Maximum gas price to use for transactions in Gwei (default: 100)
- `RISK_PROFILE`: Default risk profile (default: "moderate")
- `AUTO_COMPOUND_THRESHOLD`: Threshold for auto-compounding rewards in USD (default: 50)
- `HARVEST_INTERVAL_HOURS`: Interval for harvesting rewards in hours (default: 24)

See the configuration documentation for all available options.

## Development

### Project Structure

```
flashing-base/
├── src/
│   ├── api/              # API endpoints
│   ├── cli/              # Command-line interface
│   ├── core/             # Core components
│   ├── cqrs/             # CQRS implementation
│   │   ├── commands/     # Command definitions and handlers
│   │   ├── queries/      # Query definitions and handlers
│   │   ├── events/       # Event definitions and handlers
│   │   ├── bulkhead/     # Bulkhead pattern implementation
│   │   └── handlers/     # Common handler interfaces
│   ├── services/         # Business logic services
│   └── main.py           # Application entry point
├── tests/                # Test suite
├── requirements.txt      # Dependencies
└── README.md             # This file
```

### Running Tests

```
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add my feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Web3.py](https://web3py.readthedocs.io/) - Python library for interacting with Ethereum
- [FastAPI](https://fastapi.tiangolo.com/) - Modern, fast web framework for building APIs
- [Click](https://click.palletsprojects.com/) - Python package for creating command-line interfaces
- [Rich](https://rich.readthedocs.io/) - Python library for rich text and beautiful formatting in the terminal
