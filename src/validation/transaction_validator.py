"""
Transaction validation system for arbitrage operations
"""

from typing import Dict, Any, Tuple, List, Optional, Union
from web3 import Web3
import logging
from src.core.types import MarketValidationResult, OpportunityType, FlashLoanOpportunityType

logger = logging.getLogger(__name__)

class TransactionValidator:
    """Validates transactions before execution"""
    
    def __init__(self, web3: Web3, config: Dict[str, Any]):
        """Initialize transaction validator with configuration
        
        Args:
            web3: Web3 instance for blockchain interaction
            config: Configuration dictionary with validation thresholds
        """
        self.web3 = web3
        self.config = config
        
        # Default thresholds if not in config
        self.max_price_movement = config.get('max_price_movement', 0.02)  # 2%
        self.min_liquidity_ratio = config.get('min_liquidity_ratio', 0.8)  # 80%
        self.max_gas_increase = config.get('max_gas_increase', 1.5)  # 50%
        self.max_slippage = config.get('max_slippage', 0.01)  # 1%
        
    async def validate_transaction(
        self,
        tx_params: Dict[str, Any],
        opportunity: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Validate transaction parameters
        
        Args:
            tx_params: Transaction parameters to validate
            opportunity: Optional opportunity data for additional validation
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Basic transaction parameter validation
        if not self._validate_basic_params(tx_params):
            errors.append("Invalid basic transaction parameters")
            
        # Gas price validation
        if not await self._validate_gas_price(tx_params.get('gasPrice')):
            errors.append("Gas price too high")
            
        # Value validation
        if not self._validate_value(tx_params.get('value', 0)):
            errors.append("Invalid transaction value")
            
        # Opportunity-specific validation
        if opportunity:
            market_validation = await self.validate_market_conditions(opportunity)
            if not market_validation.is_valid:
                errors.append(f"Market validation failed: {market_validation.reason}")
                
        return len(errors) == 0, errors
        
    def _validate_basic_params(self, tx_params: Dict[str, Any]) -> bool:
        """Validate basic transaction parameters"""
        try:
            required_fields = ['from', 'to', 'gasPrice']
            if not all(field in tx_params for field in required_fields):
                return False
                
            # Validate addresses
            if not Web3.is_address(tx_params['from']) or not Web3.is_address(tx_params['to']):
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error in basic parameter validation: {str(e)}")
            return False
            
    async def _validate_gas_price(self, gas_price: int) -> bool:
        """Validate gas price is within acceptable range"""
        try:
            current_gas_price = await self.web3.eth.gas_price
            max_acceptable = int(current_gas_price * self.max_gas_increase)
            return gas_price <= max_acceptable
        except Exception as e:
            logger.error(f"Error in gas price validation: {str(e)}")
            return False
            
    def _validate_value(self, value: int) -> bool:
        """Validate transaction value"""
        try:
            # Must be non-negative
            if value < 0:
                return False
                
            # Check against max transaction value if configured
            max_value = self.config.get('max_transaction_value')
            if max_value and value > max_value:
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error in value validation: {str(e)}")
            return False
            
    async def validate_market_conditions(
        self,
        opportunity: Union[OpportunityType, FlashLoanOpportunityType]
    ) -> MarketValidationResult:
        """Validate current market conditions before execution
        
        Args:
            opportunity: Trading opportunity to validate
            
        Returns:
            MarketValidationResult with validation status and details
        """
        try:
            # Get current price
            current_price = await self._get_current_price(opportunity['token_pair'])
            entry_price = opportunity.get('entry_price', current_price)
            
            # Calculate price movement
            price_change = abs(current_price - entry_price) / entry_price
            if price_change > self.max_price_movement:
                return MarketValidationResult(
                    is_valid=False,
                    reason=f"Price moved {price_change*100:.2f}% since opportunity detection",
                    current_price=current_price,
                    price_change=price_change
                )
                
            # Validate liquidity
            current_liquidity = await self._get_current_liquidity(opportunity['token_pair'])
            initial_liquidity = opportunity.get('initial_liquidity', current_liquidity)
            liquidity_ratio = current_liquidity / initial_liquidity
            
            if liquidity_ratio < self.min_liquidity_ratio:
                return MarketValidationResult(
                    is_valid=False,
                    reason=f"Liquidity decreased to {liquidity_ratio*100:.2f}% of initial",
                    current_price=current_price,
                    price_change=price_change
                )
                
            return MarketValidationResult(
                is_valid=True,
                reason="All validations passed",
                current_price=current_price,
                price_change=price_change
            )
            
        except Exception as e:
            logger.error(f"Error in market condition validation: {str(e)}")
            return MarketValidationResult(
                is_valid=False,
                reason=f"Validation error: {str(e)}",
                current_price=0,
                price_change=0
            )
            
    async def _get_current_price(self, token_pair: Tuple[str, str]) -> float:
        """Get current price for token pair"""
        # Implementation would depend on price feed integration
        return 0.0
        
    async def _get_current_liquidity(self, token_pair: Tuple[str, str]) -> float:
        """Get current liquidity for token pair"""
        # Implementation would depend on liquidity source
        return 0.0 