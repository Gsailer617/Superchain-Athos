# Blockchain Integration Guide

This guide provides detailed information for developers on how to maintain existing blockchain integrations and add new ones to the project.

## Table of Contents
1. [Understanding the Architecture](#understanding-the-architecture)
2. [Maintaining Existing Integrations](#maintaining-existing-integrations)
3. [Adding New Blockchains](#adding-new-blockchains)
4. [Testing and Validation](#testing-and-validation)
5. [Troubleshooting](#troubleshooting)

## Understanding the Architecture

### Core Components

1. **Chain Configuration (`src/config/chain_configurations.py`)**
   - Defines chain-specific parameters
   - Manages RPC endpoints and network settings
   - Handles gas models and fee calculations

2. **Bridge Adapters (`src/core/`)**
   - Implements chain-specific bridge logic
   - Manages message verification and state monitoring
   - Handles transaction preparation and recovery

3. **Chain Connector (`src/core/chain_connector.py`)**
   - Manages Web3 connections
   - Handles connection pooling and caching
   - Implements retry logic and error handling

4. **Gas Manager (`src/core/gas_manager.py`)**
   - Implements chain-specific gas models
   - Handles fee estimation and optimization
   - Manages priority fee calculations

### Key Interfaces

1. **BridgeAdapter Interface**
```python
class BridgeAdapter:
    def validate_transfer(self, source_chain: str, target_chain: str, token: str, amount: float) -> bool:
        """Validate if transfer is possible"""
        pass

    def estimate_fees(self, source_chain: str, target_chain: str, token: str, amount: float) -> Dict[str, float]:
        """Estimate transfer fees"""
        pass

    def prepare_transfer(self, source_chain: str, target_chain: str, token: str, amount: float, recipient: str) -> TxParams:
        """Prepare transfer transaction"""
        pass
```

2. **ChainConfig Interface**
```python
@dataclass
class ChainConfig:
    name: str
    chain_id: int
    rpc_url: str
    ws_url: Optional[str]
    block_time: int
    gas_fee_model: str
    explorer_url: Optional[str]
```

## Maintaining Existing Integrations

### Regular Maintenance Tasks

1. **Update RPC Endpoints**
   - Regularly verify RPC endpoint health
   - Maintain a list of fallback endpoints
   - Monitor rate limits and usage

2. **Bridge Contract Updates**
   - Monitor bridge contract upgrades
   - Update contract addresses when needed
   - Verify new contract ABIs

3. **Gas Model Updates**
   - Monitor changes in gas models
   - Update fee calculations as needed
   - Validate gas estimation accuracy

### Version Management

1. **Dependency Updates**
   - Keep Web3.py version up to date
   - Monitor chain-specific library updates
   - Test compatibility with new versions

2. **Contract Version Tracking**
   - Track bridge contract versions
   - Document upgrade procedures
   - Maintain ABI version history

### Performance Monitoring

1. **Connection Health**
   - Monitor RPC response times
   - Track connection failures
   - Analyze request patterns

2. **Bridge Performance**
   - Monitor bridge liquidity
   - Track transfer success rates
   - Analyze gas usage patterns

## Adding New Blockchains

### Step-by-Step Process

1. **Gather Chain Information**
```python
# Example chain configuration
new_chain_config = ChainConfig(
    name="NewChain",
    chain_id=12345,
    rpc_url="https://rpc.newchain.network",
    block_time=2,
    gas_fee_model="EIP-1559",
    explorer_url="https://explorer.newchain.network"
)
```

2. **Implement Bridge Adapter**
```python
class NewChainBridgeAdapter(BridgeAdapter):
    def __init__(self, config: BridgeConfig, web3: Web3):
        super().__init__(config, web3)
        self._initialize_protocol()
    
    def _initialize_protocol(self) -> None:
        # Initialize chain-specific contracts
        pass
```

3. **Update Gas Management**
```python
def _estimate_new_chain_fees(self, web3: Web3, chain_spec: ChainSpec) -> Tuple[Wei, Wei]:
    # Implement chain-specific fee estimation
    pass
```

4. **Add Configuration**
```python
# Add to chain_configurations.py
CHAIN_CONFIGS.update({
    "newchain": new_chain_config
})
```

5. **Implement Tests**
```python
@pytest.mark.asyncio
async def test_new_chain_integration(self):
    # Test new chain functionality
    pass
```

### Required Documentation

1. **Chain Specification**
   - Document chain ID and network details
   - List official RPC endpoints
   - Document gas model specifics

2. **Bridge Details**
   - Document bridge contracts
   - Explain message verification
   - List supported tokens

3. **Integration Notes**
   - Document special considerations
   - List known limitations
   - Provide troubleshooting tips

## Testing and Validation

### Test Categories

1. **Unit Tests**
   - Test chain-specific logic
   - Validate gas calculations
   - Test bridge adapter methods

2. **Integration Tests**
   - Test cross-chain transfers
   - Validate bridge operations
   - Test error recovery

3. **Performance Tests**
   - Test connection handling
   - Validate timeout behavior
   - Test concurrent operations

### Test Environment

1. **Local Testing**
```bash
# Start test chain
ganache-cli --port 8545 --chainId 12345 --deterministic
```

2. **Testnet Validation**
```python
# Configure testnet
testnet_config = ChainConfig(
    name="newchain-testnet",
    chain_id=12345,
    rpc_url="https://testnet.newchain.network"
)
```

## Troubleshooting

### Common Issues

1. **RPC Connection Issues**
```python
# Example retry logic
def get_web3_with_retry(chain: str, max_retries: int = 3) -> Web3:
    for i in range(max_retries):
        try:
            return get_web3(chain)
        except ConnectionError:
            if i == max_retries - 1:
                raise
            time.sleep(1 * (i + 1))
```

2. **Gas Estimation Failures**
```python
# Example fallback gas estimation
def estimate_gas_with_fallback(tx: TxParams) -> int:
    try:
        return web3.eth.estimate_gas(tx)
    except Exception:
        return 300000  # Default gas limit
```

3. **Bridge Transfer Issues**
```python
# Example transfer recovery
async def recover_failed_transfer(tx_hash: str) -> Optional[str]:
    # Implement recovery logic
    pass
```

### Debugging Tools

1. **Transaction Tracer**
```python
def trace_transaction(tx_hash: str) -> Dict[str, Any]:
    # Implement transaction tracing
    pass
```

2. **State Validator**
```python
def validate_bridge_state(chain: str) -> bool:
    # Implement state validation
    pass
```

### Logging and Monitoring

1. **Configure Logging**
```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
```

2. **Monitor Metrics**
```python
def update_metrics(chain: str, success: bool) -> None:
    # Update chain-specific metrics
    pass
```

## Best Practices

1. **Code Organization**
   - Keep chain-specific logic isolated
   - Use consistent naming conventions
   - Maintain clear documentation

2. **Error Handling**
   - Implement proper error recovery
   - Use descriptive error messages
   - Log relevant debug information

3. **Performance Optimization**
   - Cache frequently used data
   - Implement connection pooling
   - Use appropriate timeouts

4. **Security Considerations**
   - Validate all inputs
   - Implement proper access controls
   - Monitor for suspicious activity

## Resources

- [Web3.py Documentation](https://web3py.readthedocs.io/)
- [EIP-1559 Specification](https://eips.ethereum.org/EIPS/eip-1559)
- [Gas Price Oracle](https://docs.ethgasstation.info/)
- [Bridge Security Best Practices](https://ethereum.org/en/bridges/) 