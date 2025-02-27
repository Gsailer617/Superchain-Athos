# Gas Optimization Module

This module provides advanced gas optimization capabilities for Ethereum transactions, helping to minimize transaction costs while ensuring timely execution.

## Components

### GasManager

The `GasManager` class is responsible for managing gas price optimization and estimation. It provides methods for:

- Optimizing gas settings based on transaction parameters and network conditions
- Estimating gas costs for transactions
- Tracking network congestion levels
- Bumping gas prices for stuck transactions
- Gathering historical gas price statistics

### AsyncGasOptimizer

The `AsyncGasOptimizer` class provides asynchronous gas optimization capabilities, including:

- Caching optimization results for efficiency
- Mode-specific optimizations (economy, normal, performance, urgent)
- Tracking optimization history
- Adjusting gas parameters based on urgency levels and network conditions

### EnhancedGasOptimizer

The `EnhancedGasOptimizer` class extends the gas optimization capabilities with:

- Full EIP-1559 support (maxFeePerGas and maxPriorityFeePerGas)
- Advanced wait time estimation based on network congestion
- External gas price API integration
- Comprehensive optimization statistics
- Gas savings calculation
- Retry mechanisms for network operations

### GasExecutionIntegrator

The `GasExecutionIntegrator` class connects gas optimization with the execution module, providing:

- Seamless integration between gas optimization and transaction execution
- Transaction execution with optimized gas settings
- Cost estimation with different optimization modes
- Gas savings calculation for executed transactions

## Usage

### Basic Usage

```python
from web3 import Web3
from src.gas import GasManager, EnhancedGasOptimizer

# Connect to Ethereum node
web3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_API_KEY'))

# Create gas manager
gas_manager = GasManager(web3)

# Create enhanced gas optimizer
optimizer = EnhancedGasOptimizer(web3, gas_manager=gas_manager)

# Optimize gas settings for a transaction
async def optimize_transaction():
    tx_params = {
        'from': '0xYourAddress',
        'to': '0xTargetAddress',
        'value': web3.to_wei(0.1, 'ether'),
        'data': '0x'
    }
    
    # Optimize gas settings
    optimized_params = await optimizer.optimize_gas_settings(tx_params)
    
    # Merge optimized settings with original params
    tx_params.update(optimized_params)
    
    # Send transaction
    tx_hash = web3.eth.send_transaction(tx_params)
    
    return tx_hash
```

### Integration with Execution Module

```python
from web3 import Web3
from src.gas import GasExecutionIntegrator

# Connect to Ethereum node
web3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/YOUR_API_KEY'))

# Configuration
config = {
    'gas': {
        'use_eip1559': True,
        'max_priority_fee': 2_000_000_000,  # 2 Gwei
        'max_fee_per_gas': 100_000_000_000,  # 100 Gwei
    },
    'execution': {
        'simulate_transactions': True,
        'retry_count': 3
    }
}

# Create integrator
integrator = GasExecutionIntegrator(web3, config)

# Execute transaction with optimized gas
async def execute_transaction():
    tx_params = {
        'from': '0xYourAddress',
        'to': '0xTargetAddress',
        'value': web3.to_wei(0.1, 'ether'),
        'data': '0x'
    }
    
    # Execute with optimized gas
    result = await integrator.execute_with_optimized_gas(
        tx_params,
        optimization_mode='normal',
        wait_for_receipt=True
    )
    
    return result
```

## Optimization Modes

The gas optimization module supports different optimization modes:

- **economy**: Prioritizes lower gas costs over speed (0.8x multiplier)
- **normal**: Balanced approach between cost and speed (1.0x multiplier)
- **performance**: Prioritizes faster confirmation over cost (1.3x multiplier)
- **urgent**: Maximizes confirmation speed (1.5x multiplier)

## Advanced Features

### Network Congestion Tracking

The module tracks network congestion levels to adjust gas prices accordingly:

```python
# Get current network congestion
congestion = await optimizer.get_network_congestion()
print(f"Network congestion: {congestion:.2f}")  # 0.0 to 1.0
```

### Gas Price Statistics

Get statistics about gas optimizations:

```python
# Get optimization statistics
stats = optimizer.get_optimization_stats()
print(f"Average gas price: {web3.from_wei(stats['avg_gas_price'], 'gwei'):.2f} Gwei")
print(f"Average priority fee: {web3.from_wei(stats['avg_priority_fee'], 'gwei'):.2f} Gwei")
print(f"Total optimizations: {stats['total_optimizations']}")
```

### External Gas Price APIs

The module can use external APIs for gas price data:

```python
# Configure external API
config = {
    'gas': {
        'use_external_api': True,
        'etherscan_api_key': 'YOUR_API_KEY'
    }
}

# Get external gas prices
prices = await optimizer.get_external_gas_prices()
print(f"Slow: {web3.from_wei(prices['slow'], 'gwei'):.2f} Gwei")
print(f"Normal: {web3.from_wei(prices['normal'], 'gwei'):.2f} Gwei")
print(f"Fast: {web3.from_wei(prices['fast'], 'gwei'):.2f} Gwei")
```

## Example

See the `examples/gas_execution_example.py` file for a complete example of using the gas optimization module with the execution module. 