# Blockchain Reference Guide

This document provides key information about each supported blockchain, including performance characteristics, gas fees, and special considerations.

## Layer 1 Chains

### Ethereum
- **Block Time**: ~12 seconds
- **Finality**: ~12 blocks (~3 minutes)
- **Gas Model**: EIP-1559
- **Key Features**:
  - MEV-Boost for MEV protection
  - Flashbots for private transactions
  - Most liquid and secure network
- **Considerations**:
  - Highest gas fees during congestion
  - Base layer for most L2s
  - Best for high-value transactions

### Polygon
- **Block Time**: ~2 seconds
- **Finality**: 128 blocks (~4 minutes)
- **Gas Model**: EIP-1559
- **Key Features**:
  - PoS consensus
  - Plasma and PoS bridge options
  - High throughput
- **Considerations**:
  - Monitor MATIC gas token price
  - Use PoS bridge for better security
  - Higher confirmation requirements

### BNB Chain
- **Block Time**: ~3 seconds
- **Finality**: 15 blocks (~45 seconds)
- **Gas Model**: Legacy
- **Key Features**:
  - Parlia consensus (PoSA)
  - Cross-chain support
  - High throughput
- **Considerations**:
  - Centralized validator set
  - Use multiple RPC endpoints
  - Monitor BNB price for gas costs

### Avalanche
- **Block Time**: ~2 seconds
- **Finality**: 12 blocks (~24 seconds)
- **Gas Model**: EIP-1559
- **Key Features**:
  - Subnet support
  - Multiple chain types (X/P/C)
  - Fast finality
- **Considerations**:
  - Different chains for different purposes
  - C-Chain for EVM compatibility
  - AVAX required for gas

### Gnosis Chain
- **Block Time**: ~5 seconds
- **Finality**: 12 blocks (~1 minute)
- **Gas Model**: EIP-1559
- **Key Features**:
  - xDAI as native token
  - Omni bridge support
  - Stable token focus
- **Considerations**:
  - Lower gas costs in USD terms
  - Good for stable token operations
  - Limited liquidity compared to major chains

## Layer 2 Chains

### Base
- **Block Time**: ~2 seconds
- **Finality**: 5 blocks (~10 seconds)
- **Gas Model**: Optimistic
- **Key Features**:
  - Optimistic rollup
  - Fast withdrawals
  - Coinbase integration
- **Considerations**:
  - 7-day withdrawal period (without fast bridges)
  - Growing ecosystem
  - Shared security with Ethereum

### Arbitrum
- **Block Time**: ~250ms
- **Finality**: 64 blocks (~16 seconds)
- **Gas Model**: Arbitrum-specific
- **Key Features**:
  - Nitro upgrade
  - Calldata compression
  - Fast withdrawals
- **Considerations**:
  - Complex gas pricing
  - High throughput
  - Growing DeFi ecosystem

### Optimism
- **Block Time**: ~2 seconds
- **Finality**: 50 blocks (~100 seconds)
- **Gas Model**: Optimistic
- **Key Features**:
  - Bedrock upgrade
  - OP Stack compatibility
  - Fast withdrawals
- **Considerations**:
  - 7-day withdrawal period (without fast bridges)
  - Similar to Ethereum
  - Strong infrastructure

### Linea
- **Block Time**: ~12 seconds
- **Finality**: 10 blocks (~2 minutes)
- **Gas Model**: EIP-1559
- **Key Features**:
  - ZK rollup
  - ConsenSys backing
  - Native bridge
- **Considerations**:
  - Newer network
  - Growing ecosystem
  - ZK proof verification times

### Mantle
- **Block Time**: ~2 seconds
- **Finality**: 10 blocks (~20 seconds)
- **Gas Model**: Optimistic
- **Key Features**:
  - Optimistic rollup
  - Data availability
  - MNT token
- **Considerations**:
  - Newer network
  - Limited RPC options
  - Monitor ecosystem growth

## Cross-Chain Considerations

### Bridge Selection
- Use official bridges for maximum security
- Consider fast bridge options for better UX
- Monitor bridge liquidity and limits

### Gas Optimization
- Each chain has different gas models
- Monitor gas tokens (ETH, MATIC, BNB, etc.)
- Use appropriate gas limits per chain

### Transaction Monitoring
- Use chain-specific explorers
- Monitor multiple confirmations
- Consider reorg risks

### Security Considerations
- Different finality times per chain
- Bridge security varies
- Monitor network status

## Performance Monitoring

### Key Metrics
- Block times
- Gas prices
- Bridge liquidity
- Network congestion

### Health Checks
- RPC endpoint availability
- Bridge contract status
- Network upgrades

### Error Handling
- Chain-specific error codes
- Retry strategies
- Fallback endpoints 