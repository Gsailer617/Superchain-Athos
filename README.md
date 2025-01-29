# Flash Loan Arbitrage Bot for Base

This repository contains a flash loan arbitrage bot designed to execute profitable trading opportunities on the Base network using flash loans.

## Features

- Automated flash loan arbitrage execution
- Price discovery across multiple DEXes
- Profit calculation and gas optimization
- Real-time monitoring dashboard
- Configurable parameters for risk management

## Prerequisites

- Node.js (v14 or higher)
- Hardhat
- Base network RPC access
- MetaMask or another Web3 wallet

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
npm install
```

3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your configuration values
   ```bash
   cp .env.example .env
   ```

## Configuration

Create a `.env` file with the following variables (see `.env.example` for template):

- `BASE_RPC_URL`: Your Base network RPC URL
- `PRIVATE_KEY`: Your wallet private key (keep this secure!)
- `DEPLOYED_CONTRACT_ADDRESS`: Address of deployed FlashLoanArbitrage contract
- `POLLING_INTERVAL`: Bot polling interval in milliseconds
- `MIN_PROFIT_USD`: Minimum profit threshold in USD
- `GAS_PRICE_LIMIT`: Maximum gas price limit
- `BASESCAN_API_KEY`: Your Basescan API key

## Usage

1. Deploy the contracts:
```bash
npx hardhat run scripts/deploy.js --network base
```

2. Run the arbitrage bot:
```bash
npm run start:bot
```

3. Start the monitoring dashboard:
```bash
npm run start:dashboard
```

## Testing

```bash
npx hardhat test
```

## Security Considerations

- Never commit your `.env` file
- Keep your private keys secure
- Monitor gas prices and set appropriate limits
- Test thoroughly on testnet before mainnet deployment

## License

MIT

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request
