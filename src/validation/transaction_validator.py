"""
Transaction validation system for arbitrage operations
"""

from typing import Dict, Any, Tuple, List, Optional, Union, Callable
from web3 import Web3
import logging
import json
import os
from enum import Enum
from functools import wraps
from src.core.types import MarketValidationResult, OpportunityType, FlashLoanOpportunityType
from src.utils.unified_metrics import MetricsManager

logger = logging.getLogger(__name__)

class TransactionType(Enum):
    """Enum for different transaction types"""
    STANDARD = "standard"
    BRIDGE = "bridge"
    SWAP = "swap"
    FLASH_LOAN = "flash_loan"
    CONTRACT_INTERACTION = "contract_interaction"

class ValidationLevel(Enum):
    """Validation levels for different contexts"""
    BASIC = "basic"  # Basic parameter validation
    STANDARD = "standard"  # Standard validation including gas and value
    STRICT = "strict"  # Strict validation including market conditions

def validation_metrics(func):
    """Decorator to record validation metrics"""
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        tx_type = kwargs.get('tx_type', TransactionType.STANDARD)
        metrics = MetricsManager.get_instance()
        
        try:
            start_time = metrics.current_time_ms()
            result = await func(self, *args, **kwargs)
            end_time = metrics.current_time_ms()
            
            # Record validation time
            metrics.update_metric(
                "transaction_validation_time", 
                end_time - start_time,
                {"transaction_type": tx_type.value}
            )
            
            # Record validation result
            metrics.update_metric(
                "transaction_validation_result",
                1 if result[0] else 0,
                {"transaction_type": tx_type.value}
            )
            
            # Record error count if any
            if not result[0]:
                metrics.update_metric(
                    "transaction_validation_errors",
                    len(result[1]),
                    {"transaction_type": tx_type.value}
                )
            
            return result
        except Exception as e:
            logger.error(f"Error in validation metrics: {str(e)}")
            metrics.update_metric(
                "transaction_validation_exceptions",
                1,
                {"transaction_type": tx_type.value, "error": str(e)}
            )
            raise
    
    return wrapper

class TransactionValidator:
    """Validates transactions before execution"""
    
    # Schema directory relative to this file
    SCHEMA_DIR = os.path.join(os.path.dirname(__file__), "schemas")
    
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
        
        # Load JSON schemas for different transaction types
        self.schemas = self._load_schemas()
        
        # Register validators for different transaction types
        self.validators = {
            TransactionType.STANDARD: self._validate_standard_tx,
            TransactionType.BRIDGE: self._validate_bridge_tx,
            TransactionType.SWAP: self._validate_swap_tx,
            TransactionType.FLASH_LOAN: self._validate_flash_loan_tx,
            TransactionType.CONTRACT_INTERACTION: self._validate_contract_interaction
        }
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_metrics(self):
        """Initialize metrics for transaction validation"""
        metrics = MetricsManager.get_instance()
        
        # Register validation metrics
        metrics.register_metric(
            "transaction_validation_time",
            "Time taken to validate transactions in ms",
            "gauge"
        )
        
        metrics.register_metric(
            "transaction_validation_result",
            "Result of transaction validation (1=success, 0=failure)",
            "counter"
        )
        
        metrics.register_metric(
            "transaction_validation_errors",
            "Number of validation errors",
            "counter"
        )
        
        metrics.register_metric(
            "transaction_validation_exceptions",
            "Number of exceptions during validation",
            "counter"
        )
    
    def _load_schemas(self) -> Dict[TransactionType, Dict]:
        """Load JSON schemas for transaction validation"""
        schemas = {}
        
        # Create schemas directory if it doesn't exist
        os.makedirs(self.SCHEMA_DIR, exist_ok=True)
        
        # Define schemas for different transaction types
        # These could be loaded from files in a production system
        schemas[TransactionType.STANDARD] = {
            "type": "object",
            "required": ["from", "to", "gasPrice"],
            "properties": {
                "from": {"type": "string"},
                "to": {"type": "string"},
                "gasPrice": {"type": "integer", "minimum": 0},
                "value": {"type": "integer", "minimum": 0},
                "data": {"type": "string"}
            }
        }
        
        schemas[TransactionType.BRIDGE] = {
            "type": "object",
            "required": ["source_chain", "target_chain", "token", "amount", "recipient"],
            "properties": {
                "source_chain": {"type": "string"},
                "target_chain": {"type": "string"},
                "token": {"type": "string"},
                "amount": {"type": "string", "pattern": "^[0-9]+$"},
                "recipient": {"type": "string"},
                "max_gas": {"type": "integer", "minimum": 0}
            }
        }
        
        schemas[TransactionType.SWAP] = {
            "type": "object",
            "required": ["token_in", "token_out", "amount_in", "min_amount_out"],
            "properties": {
                "token_in": {"type": "string"},
                "token_out": {"type": "string"},
                "amount_in": {"type": "string", "pattern": "^[0-9]+$"},
                "min_amount_out": {"type": "string", "pattern": "^[0-9]+$"},
                "deadline": {"type": "integer", "minimum": 0}
            }
        }
        
        schemas[TransactionType.FLASH_LOAN] = {
            "type": "object",
            "required": ["token", "amount", "actions"],
            "properties": {
                "token": {"type": "string"},
                "amount": {"type": "string", "pattern": "^[0-9]+$"},
                "actions": {
                    "type": "array",
                    "items": {"type": "object"}
                }
            }
        }
        
        # Save schemas to files for reference
        for tx_type, schema in schemas.items():
            schema_path = os.path.join(self.SCHEMA_DIR, f"{tx_type.value}.json")
            with open(schema_path, 'w') as f:
                json.dump(schema, f, indent=2)
        
        return schemas
    
    @validation_metrics
    async def validate_transaction(
        self,
        tx_params: Dict[str, Any],
        tx_type: TransactionType = TransactionType.STANDARD,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        opportunity: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[str]]:
        """Validate transaction parameters
        
        Args:
            tx_params: Transaction parameters to validate
            tx_type: Type of transaction to validate
            validation_level: Level of validation to perform
            opportunity: Optional opportunity data for additional validation
            
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Structure validation (JSON schema)
        if not self._validate_structure(tx_params, tx_type):
            errors.append(f"Invalid {tx_type.value} transaction structure")
        
        # Type-specific validation
        if tx_type in self.validators:
            type_valid, type_errors = await self.validators[tx_type](tx_params)
            if not type_valid:
                errors.extend(type_errors)
        
        # Standard validation if requested
        if validation_level in [ValidationLevel.STANDARD, ValidationLevel.STRICT]:
            # Gas price validation for standard transactions
            if tx_type == TransactionType.STANDARD and 'gasPrice' in tx_params:
                if not await self._validate_gas_price(tx_params.get('gasPrice')):
                    errors.append("Gas price too high")
                
            # Value validation
            if 'value' in tx_params and not self._validate_value(tx_params.get('value', 0)):
                errors.append("Invalid transaction value")
        
        # Strict validation including market conditions
        if validation_level == ValidationLevel.STRICT and opportunity:
            market_validation = await self.validate_market_conditions(opportunity)
            if not market_validation.is_valid:
                errors.append(f"Market validation failed: {market_validation.reason}")
        
        return len(errors) == 0, errors
    
    def _validate_structure(self, tx_params: Dict[str, Any], tx_type: TransactionType) -> bool:
        """Validate transaction structure against JSON schema"""
        try:
            import jsonschema
            
            if tx_type not in self.schemas:
                logger.warning(f"No schema defined for transaction type {tx_type.value}")
                return True
            
            jsonschema.validate(instance=tx_params, schema=self.schemas[tx_type])
            return True
        except ImportError:
            logger.warning("jsonschema package not installed, skipping schema validation")
            return True
        except Exception as e:
            logger.error(f"Schema validation error for {tx_type.value}: {str(e)}")
            return False
    
    async def _validate_standard_tx(self, tx_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate standard transaction parameters"""
        errors = []
        
        try:
            # Validate addresses
            if not Web3.is_address(tx_params.get('from', '')) or not Web3.is_address(tx_params.get('to', '')):
                errors.append("Invalid 'from' or 'to' address")
            
            # Validate gas limit if provided
            if 'gas' in tx_params:
                try:
                    # Estimate gas for the transaction
                    estimated_gas = await self.web3.eth.estimate_gas({
                        'from': tx_params['from'],
                        'to': tx_params['to'],
                        'value': tx_params.get('value', 0),
                        'data': tx_params.get('data', '0x')
                    })
                    
                    # Check if provided gas is sufficient
                    if tx_params['gas'] < estimated_gas:
                        errors.append(f"Gas limit too low. Provided: {tx_params['gas']}, Estimated: {estimated_gas}")
                except Exception as e:
                    errors.append(f"Gas estimation failed: {str(e)}")
            
            return len(errors) == 0, errors
        except Exception as e:
            logger.error(f"Error in standard transaction validation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    async def _validate_bridge_tx(self, tx_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate bridge transaction parameters"""
        errors = []
        
        try:
            # Check if bridge exists for the chain pair
            source_chain = tx_params.get('source_chain')
            target_chain = tx_params.get('target_chain')
            
            # This would be replaced with actual bridge availability check
            bridge_available = self._check_bridge_availability(source_chain, target_chain)
            if not bridge_available:
                errors.append(f"No bridge available from {source_chain} to {target_chain}")
            
            # Validate amount
            amount = tx_params.get('amount', '0')
            if int(amount) <= 0:
                errors.append("Bridge amount must be greater than 0")
            
            # Validate recipient address
            recipient = tx_params.get('recipient')
            if not Web3.is_address(recipient):
                errors.append("Invalid recipient address")
            
            return len(errors) == 0, errors
        except Exception as e:
            logger.error(f"Error in bridge transaction validation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def _check_bridge_availability(self, source_chain: str, target_chain: str) -> bool:
        """Check if a bridge is available between chains"""
        # This would be implemented with actual bridge registry
        # For now, return True as a placeholder
        return True
    
    async def _validate_swap_tx(self, tx_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate swap transaction parameters"""
        errors = []
        
        try:
            # Validate token addresses
            token_in = tx_params.get('token_in')
            token_out = tx_params.get('token_out')
            
            if not Web3.is_address(token_in) or not Web3.is_address(token_out):
                errors.append("Invalid token address")
            
            # Validate amounts
            amount_in = int(tx_params.get('amount_in', '0'))
            min_amount_out = int(tx_params.get('min_amount_out', '0'))
            
            if amount_in <= 0:
                errors.append("Input amount must be greater than 0")
            
            if min_amount_out <= 0:
                errors.append("Minimum output amount must be greater than 0")
            
            # Validate slippage
            if 'expected_amount_out' in tx_params:
                expected_out = int(tx_params.get('expected_amount_out'))
                slippage = (expected_out - min_amount_out) / expected_out
                
                if slippage > self.max_slippage:
                    errors.append(f"Slippage too high: {slippage*100:.2f}% > {self.max_slippage*100:.2f}%")
            
            # Validate deadline if provided
            if 'deadline' in tx_params:
                current_block = await self.web3.eth.block_number
                current_timestamp = (await self.web3.eth.get_block(current_block))['timestamp']
                
                if tx_params['deadline'] <= current_timestamp:
                    errors.append("Transaction deadline has passed")
            
            return len(errors) == 0, errors
        except Exception as e:
            logger.error(f"Error in swap transaction validation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    async def _validate_flash_loan_tx(self, tx_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate flash loan transaction parameters"""
        errors = []
        
        try:
            # Validate token address
            token = tx_params.get('token')
            if not Web3.is_address(token):
                errors.append("Invalid token address")
            
            # Validate amount
            amount = int(tx_params.get('amount', '0'))
            if amount <= 0:
                errors.append("Flash loan amount must be greater than 0")
            
            # Validate actions
            actions = tx_params.get('actions', [])
            if not actions:
                errors.append("Flash loan must include at least one action")
            
            # Validate each action
            for i, action in enumerate(actions):
                if 'type' not in action:
                    errors.append(f"Action {i} missing type")
                
                # Validate action parameters based on type
                action_type = action.get('type')
                if action_type == 'swap':
                    swap_valid, swap_errors = await self._validate_swap_tx(action)
                    if not swap_valid:
                        errors.extend([f"Action {i} swap: {err}" for err in swap_errors])
            
            return len(errors) == 0, errors
        except Exception as e:
            logger.error(f"Error in flash loan transaction validation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    async def _validate_contract_interaction(self, tx_params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate contract interaction parameters"""
        errors = []
        
        try:
            # Validate contract address
            contract_address = tx_params.get('contract_address')
            if not Web3.is_address(contract_address):
                errors.append("Invalid contract address")
            
            # Validate function signature
            function_signature = tx_params.get('function_signature')
            if not function_signature:
                errors.append("Missing function signature")
            
            # Validate parameters
            params = tx_params.get('params', [])
            
            # This would be replaced with actual ABI validation
            # For now, just check if params is a list
            if not isinstance(params, list):
                errors.append("Parameters must be a list")
            
            return len(errors) == 0, errors
        except Exception as e:
            logger.error(f"Error in contract interaction validation: {str(e)}")
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
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