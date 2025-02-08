from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from decimal import Decimal
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradeOpportunity:
    """Represents a potential trading opportunity"""
    token_pair: Tuple[str, str]  # (token0_address, token1_address)
    amount: Decimal
    expected_profit: Decimal
    confidence: float
    risk_score: float
    gas_estimate: int
    path: List[str]  # List of DEX addresses in the arbitrage path
    timestamp: Optional[float] = None
    execution_params: Optional[dict] = None
    gas_price: Optional[int] = None
    max_slippage: float = 0.01  # 1% default max slippage
    min_profit_threshold: Decimal = Decimal('0.001')  # Minimum profit in ETH
    
    def __post_init__(self):
        # Convert numeric inputs to appropriate types
        self.amount = Decimal(str(self.amount))
        self.expected_profit = Decimal(str(self.expected_profit))
        self.timestamp = self.timestamp or datetime.now().timestamp()
        
        # Validate the opportunity
        self._validate()
        
    def _validate(self):
        """Validate the trade opportunity"""
        if not all(isinstance(addr, str) and len(addr) == 42 and addr.startswith('0x') 
                  for addr in self.token_pair):
            raise ValueError("Invalid token addresses")
            
        if self.amount <= 0:
            raise ValueError("Trade amount must be positive")
            
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
            
        if self.risk_score < 0 or self.risk_score > 1:
            raise ValueError("Risk score must be between 0 and 1")
            
        if not self.path:
            raise ValueError("Trade path cannot be empty")
            
    def calculate_gas_cost(self, gas_price: Optional[int] = None) -> Decimal:
        """Calculate the gas cost in ETH"""
        gas_price = gas_price or self.gas_price
        if not gas_price:
            raise ValueError("Gas price not provided")
            
        return Decimal(str(self.gas_estimate * gas_price)) / Decimal('1000000000000000000')
        
    def calculate_net_profit(self, gas_price: Optional[int] = None) -> Decimal:
        """Calculate net profit after gas costs"""
        try:
            gas_cost = self.calculate_gas_cost(gas_price)
            return self.expected_profit - gas_cost
        except Exception as e:
            logger.error(f"Error calculating net profit: {str(e)}")
            return Decimal('0')
            
    def estimate_slippage(self, liquidity_data: Dict[str, Decimal]) -> float:
        """Estimate potential slippage based on liquidity data"""
        try:
            min_liquidity = min(
                liquidity_data.get(dex, Decimal('0'))
                for dex in self.path
            )
            if min_liquidity == 0:
                return float('inf')
                
            # Simple slippage estimation based on trade size vs liquidity
            return float(self.amount / min_liquidity)
        except Exception as e:
            logger.error(f"Error estimating slippage: {str(e)}")
            return float('inf')
            
    def validate_execution_conditions(self, 
                                   current_gas_price: int,
                                   liquidity_data: Dict[str, Decimal]) -> Tuple[bool, str]:
        """Validate if the trade can be executed under current conditions"""
        try:
            # Check if trade is still profitable with current gas price
            net_profit = self.calculate_net_profit(current_gas_price)
            if net_profit < self.min_profit_threshold:
                return False, f"Net profit {net_profit} below threshold {self.min_profit_threshold}"
                
            # Check slippage
            estimated_slippage = self.estimate_slippage(liquidity_data)
            if estimated_slippage > self.max_slippage:
                return False, f"Estimated slippage {estimated_slippage} above maximum {self.max_slippage}"
                
            # Check if trade is expired
            if (datetime.now().timestamp() - self.timestamp) > 60:  # 1 minute timeout
                return False, "Trade opportunity expired"
                
            return True, "Trade conditions valid"
            
        except Exception as e:
            logger.error(f"Error validating execution conditions: {str(e)}")
            return False, str(e)
        
    def to_dict(self) -> dict:
        """Convert the trade opportunity to a dictionary"""
        return {
            'token_pair': self.token_pair,
            'amount': float(self.amount),
            'expected_profit': float(self.expected_profit),
            'confidence': self.confidence,
            'risk_score': self.risk_score,
            'gas_estimate': self.gas_estimate,
            'path': self.path,
            'timestamp': self.timestamp,
            'execution_params': self.execution_params,
            'gas_price': self.gas_price,
            'net_profit': float(self.calculate_net_profit()) if self.gas_price else None
        }
        
    @property
    def is_profitable(self) -> bool:
        """Check if the trade is expected to be profitable after gas costs"""
        try:
            return (self.calculate_net_profit() > self.min_profit_threshold and 
                   self.confidence > 0.5)
        except Exception:
            return False
        
    @property
    def risk_adjusted_profit(self) -> float:
        """Calculate risk-adjusted profit expectation"""
        try:
            net_profit = float(self.calculate_net_profit())
            return net_profit * (1 - self.risk_score) * self.confidence
        except Exception:
            return 0.0 