"""Enhanced trade metrics for comprehensive trade history tracking"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
import json

@dataclass
class GasMetrics:
    """Gas usage metrics for a trade"""
    gas_used: int
    gas_price: int  # in wei
    max_fee_per_gas: Optional[int] = None  # for EIP-1559
    max_priority_fee_per_gas: Optional[int] = None  # for EIP-1559
    effective_gas_price: Optional[int] = None  # actual gas price paid
    gas_cost_wei: int = 0  # total gas cost in wei
    gas_cost_eth: float = 0.0  # total gas cost in ETH
    gas_cost_usd: float = 0.0  # total gas cost in USD
    optimization_mode: str = "normal"  # economy, normal, performance, urgent
    optimization_savings: float = 0.0  # estimated savings percentage
    network_congestion: float = 0.0  # network congestion at time of transaction
    
    def __post_init__(self):
        """Calculate derived fields if not provided"""
        if self.gas_cost_wei == 0 and self.gas_used > 0:
            # Calculate gas cost in wei
            if self.effective_gas_price:
                self.gas_cost_wei = self.gas_used * self.effective_gas_price
            else:
                self.gas_cost_wei = self.gas_used * self.gas_price
                
        # Calculate ETH cost if not provided
        if self.gas_cost_eth == 0 and self.gas_cost_wei > 0:
            self.gas_cost_eth = self.gas_cost_wei / 1e18

@dataclass
class ExecutionMetrics:
    """Execution metrics for a trade"""
    tx_hash: str
    block_number: Optional[int] = None
    status: bool = False  # True if successful
    chain_id: int = 1  # Ethereum mainnet by default
    nonce: Optional[int] = None
    execution_time: float = 0.0  # seconds
    confirmation_time: float = 0.0  # seconds
    confirmation_blocks: int = 0
    retry_count: int = 0
    simulated: bool = False  # whether transaction was simulated before execution
    simulation_success: bool = False  # result of simulation
    error: Optional[str] = None  # error message if failed
    
@dataclass
class TokenMetrics:
    """Token metrics for a trade"""
    token_in: str
    token_out: str
    amount_in: float
    amount_out: float
    token_in_symbol: str
    token_out_symbol: str
    token_in_decimals: int
    token_out_decimals: int
    token_in_price_usd: float = 0.0
    token_out_price_usd: float = 0.0
    slippage: float = 0.0  # actual slippage percentage
    price_impact: float = 0.0  # price impact percentage
    
@dataclass
class EnhancedTradeMetrics:
    """Enhanced metrics for a single trade with comprehensive data"""
    # Basic trade info
    timestamp: datetime
    strategy: str
    token_pair: str
    dex: str
    profit: float  # in USD
    success: bool
    
    # Detailed metrics
    gas: GasMetrics
    execution: ExecutionMetrics
    tokens: TokenMetrics
    
    # Additional data
    route: List[Dict[str, Any]] = field(default_factory=list)  # trading route details
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            # Basic info
            'timestamp': self.timestamp,
            'strategy': self.strategy,
            'token_pair': self.token_pair,
            'dex': self.dex,
            'profit': self.profit,
            'success': self.success,
            
            # Gas metrics
            'gas_used': self.gas.gas_used,
            'gas_price': self.gas.gas_price,
            'max_fee_per_gas': self.gas.max_fee_per_gas,
            'max_priority_fee_per_gas': self.gas.max_priority_fee_per_gas,
            'effective_gas_price': self.gas.effective_gas_price,
            'gas_cost_wei': self.gas.gas_cost_wei,
            'gas_cost_eth': self.gas.gas_cost_eth,
            'gas_cost_usd': self.gas.gas_cost_usd,
            'optimization_mode': self.gas.optimization_mode,
            'optimization_savings': self.gas.optimization_savings,
            'network_congestion': self.gas.network_congestion,
            
            # Execution metrics
            'tx_hash': self.execution.tx_hash,
            'block_number': self.execution.block_number,
            'status': self.execution.status,
            'chain_id': self.execution.chain_id,
            'nonce': self.execution.nonce,
            'execution_time': self.execution.execution_time,
            'confirmation_time': self.execution.confirmation_time,
            'confirmation_blocks': self.execution.confirmation_blocks,
            'retry_count': self.execution.retry_count,
            'simulated': self.execution.simulated,
            'simulation_success': self.execution.simulation_success,
            'error': self.execution.error,
            
            # Token metrics
            'token_in': self.tokens.token_in,
            'token_out': self.tokens.token_out,
            'amount_in': self.tokens.amount_in,
            'amount_out': self.tokens.amount_out,
            'token_in_symbol': self.tokens.token_in_symbol,
            'token_out_symbol': self.tokens.token_out_symbol,
            'token_in_decimals': self.tokens.token_in_decimals,
            'token_out_decimals': self.tokens.token_out_decimals,
            'token_in_price_usd': self.tokens.token_in_price_usd,
            'token_out_price_usd': self.tokens.token_out_price_usd,
            'slippage': self.tokens.slippage,
            'price_impact': self.tokens.price_impact,
            
            # Additional data
            'route': self.route,
            'additional_data': self.additional_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnhancedTradeMetrics':
        """Create from dictionary"""
        # Extract gas metrics
        gas = GasMetrics(
            gas_used=data.get('gas_used', 0),
            gas_price=data.get('gas_price', 0),
            max_fee_per_gas=data.get('max_fee_per_gas'),
            max_priority_fee_per_gas=data.get('max_priority_fee_per_gas'),
            effective_gas_price=data.get('effective_gas_price'),
            gas_cost_wei=data.get('gas_cost_wei', 0),
            gas_cost_eth=data.get('gas_cost_eth', 0.0),
            gas_cost_usd=data.get('gas_cost_usd', 0.0),
            optimization_mode=data.get('optimization_mode', 'normal'),
            optimization_savings=data.get('optimization_savings', 0.0),
            network_congestion=data.get('network_congestion', 0.0)
        )
        
        # Extract execution metrics
        execution = ExecutionMetrics(
            tx_hash=data.get('tx_hash', ''),
            block_number=data.get('block_number'),
            status=data.get('status', False),
            chain_id=data.get('chain_id', 1),
            nonce=data.get('nonce'),
            execution_time=data.get('execution_time', 0.0),
            confirmation_time=data.get('confirmation_time', 0.0),
            confirmation_blocks=data.get('confirmation_blocks', 0),
            retry_count=data.get('retry_count', 0),
            simulated=data.get('simulated', False),
            simulation_success=data.get('simulation_success', False),
            error=data.get('error')
        )
        
        # Extract token metrics
        tokens = TokenMetrics(
            token_in=data.get('token_in', ''),
            token_out=data.get('token_out', ''),
            amount_in=data.get('amount_in', 0.0),
            amount_out=data.get('amount_out', 0.0),
            token_in_symbol=data.get('token_in_symbol', ''),
            token_out_symbol=data.get('token_out_symbol', ''),
            token_in_decimals=data.get('token_in_decimals', 18),
            token_out_decimals=data.get('token_out_decimals', 18),
            token_in_price_usd=data.get('token_in_price_usd', 0.0),
            token_out_price_usd=data.get('token_out_price_usd', 0.0),
            slippage=data.get('slippage', 0.0),
            price_impact=data.get('price_impact', 0.0)
        )
        
        return cls(
            timestamp=data.get('timestamp', datetime.now()),
            strategy=data.get('strategy', ''),
            token_pair=data.get('token_pair', ''),
            dex=data.get('dex', ''),
            profit=data.get('profit', 0.0),
            success=data.get('success', False),
            gas=gas,
            execution=execution,
            tokens=tokens,
            route=data.get('route', []),
            additional_data=data.get('additional_data', {})
        )
    
    @classmethod
    def from_execution_result(cls, result: Dict[str, Any], strategy: str = '') -> 'EnhancedTradeMetrics':
        """Create from execution result dictionary"""
        # Extract receipt data
        receipt = result.get('receipt', {})
        
        # Extract gas optimization data
        gas_optimization = result.get('gas_optimization', {})
        
        # Extract token data
        token_data = result.get('token_data', {})
        
        # Create gas metrics
        gas = GasMetrics(
            gas_used=receipt.get('gasUsed', 0),
            gas_price=receipt.get('effectiveGasPrice', 0),
            effective_gas_price=receipt.get('effectiveGasPrice'),
            max_fee_per_gas=result.get('tx_params', {}).get('maxFeePerGas'),
            max_priority_fee_per_gas=result.get('tx_params', {}).get('maxPriorityFeePerGas'),
            gas_cost_wei=receipt.get('gasUsed', 0) * receipt.get('effectiveGasPrice', 0),
            optimization_mode=gas_optimization.get('mode', 'normal'),
            optimization_savings=gas_optimization.get('estimated_savings', 0.0),
            network_congestion=gas_optimization.get('network_congestion', 0.0)
        )
        
        # Create execution metrics
        execution = ExecutionMetrics(
            tx_hash=receipt.get('transactionHash', result.get('tx_hash', '')),
            block_number=receipt.get('blockNumber'),
            status=receipt.get('status', 0) == 1,
            chain_id=result.get('chain_id', 1),
            nonce=result.get('tx_params', {}).get('nonce'),
            execution_time=result.get('execution_time', 0.0),
            confirmation_time=result.get('confirmation_time', 0.0),
            confirmation_blocks=result.get('confirmation_blocks', 0),
            retry_count=result.get('retry_count', 0),
            simulated=result.get('simulated', False),
            simulation_success=result.get('simulation_success', False),
            error=result.get('error')
        )
        
        # Create token metrics
        tokens = TokenMetrics(
            token_in=token_data.get('token_in', ''),
            token_out=token_data.get('token_out', ''),
            amount_in=token_data.get('amount_in', 0.0),
            amount_out=token_data.get('amount_out', 0.0),
            token_in_symbol=token_data.get('token_in_symbol', ''),
            token_out_symbol=token_data.get('token_out_symbol', ''),
            token_in_decimals=token_data.get('token_in_decimals', 18),
            token_out_decimals=token_data.get('token_out_decimals', 18),
            token_in_price_usd=token_data.get('token_in_price_usd', 0.0),
            token_out_price_usd=token_data.get('token_out_price_usd', 0.0),
            slippage=token_data.get('slippage', 0.0),
            price_impact=token_data.get('price_impact', 0.0)
        )
        
        # Calculate profit
        profit = token_data.get('profit_usd', 0.0)
        if profit == 0.0 and tokens.token_out_price_usd > 0:
            # Try to calculate profit
            value_out = tokens.amount_out * tokens.token_out_price_usd
            value_in = tokens.amount_in * tokens.token_in_price_usd
            profit = value_out - value_in - gas.gas_cost_usd
        
        return cls(
            timestamp=datetime.now(),
            strategy=strategy,
            token_pair=f"{tokens.token_in_symbol}/{tokens.token_out_symbol}",
            dex=result.get('dex', ''),
            profit=profit,
            success=execution.status,
            gas=gas,
            execution=execution,
            tokens=tokens,
            route=result.get('route', []),
            additional_data=result.get('additional_data', {})
        ) 