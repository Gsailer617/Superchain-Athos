"""Security manager for handling security validations and checks."""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from web3 import Web3
import json
from pathlib import Path

from src.core.exceptions import SecurityError
from src.core.types import TokenAddress, ChainId, ProtocolId

@dataclass
class SecurityConfig:
    """Security configuration settings."""
    max_slippage: float = 0.01  # 1%
    max_price_impact: float = 0.05  # 5%
    min_liquidity: float = 100000.0  # $100k
    max_position_size: float = 1000000.0  # $1M
    max_gas_price: int = 500  # gwei
    min_timelock: int = 86400  # 24 hours
    max_flash_loan_ratio: float = 0.75  # 75%
    required_audits: bool = True
    required_timelock: bool = True
    blacklisted_addresses: Set[str] = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.blacklisted_addresses is None:
            self.blacklisted_addresses = set()

@dataclass
class SecurityValidation:
    """Result of a security validation check."""
    is_valid: bool
    reason: Optional[str] = None
    risk_score: float = 0.0
    warnings: List[str] = None
    
    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.warnings is None:
            self.warnings = []

@dataclass
class SecurityMetrics:
    """Security-related metrics."""
    total_validations: int = 0
    failed_validations: int = 0
    average_risk_score: float = 0.0
    high_risk_operations: int = 0
    blocked_operations: int = 0

class SecurityManager:
    """Manager for security validations and checks."""
    
    def __init__(self, config: SecurityConfig):
        """Initialize the security manager.
        
        Args:
            config: Security configuration settings
        """
        self.config = config
        self.metrics = SecurityMetrics()
        
    async def validate_strategy(self, strategy: Dict[str, Any]) -> SecurityValidation:
        """Validate a trading strategy for security concerns.
        
        Args:
            strategy: Strategy to validate
            
        Returns:
            SecurityValidation result
            
        Raises:
            SecurityError: If validation fails
        """
        try:
            validation = SecurityValidation(is_valid=True)
            
            # Check position size
            position_size = float(strategy.get('position_size', 0))
            if position_size > self.config.max_position_size:
                validation.is_valid = False
                validation.reason = f"Position size {position_size} exceeds maximum {self.config.max_position_size}"
                validation.risk_score = 1.0
                return validation
            
            # Check slippage
            slippage = float(strategy.get('max_slippage', 0))
            if slippage > self.config.max_slippage:
                validation.warnings.append(
                    f"High slippage tolerance: {slippage*100}%"
                )
                validation.risk_score += 0.3
            
            # Check price impact
            price_impact = float(strategy.get('price_impact', 0))
            if price_impact > self.config.max_price_impact:
                validation.is_valid = False
                validation.reason = f"Price impact {price_impact*100}% exceeds maximum {self.config.max_price_impact*100}%"
                validation.risk_score = 1.0
                return validation
            
            # Check liquidity
            liquidity = float(strategy.get('liquidity', 0))
            if liquidity < self.config.min_liquidity:
                validation.warnings.append(
                    f"Low liquidity: ${liquidity:,.2f}"
                )
                validation.risk_score += 0.3
            
            # Check gas price
            gas_price = int(strategy.get('gas_price', 0))
            if gas_price > self.config.max_gas_price:
                validation.warnings.append(
                    f"High gas price: {gas_price} gwei"
                )
                validation.risk_score += 0.2
            
            # Update metrics
            self.metrics.total_validations += 1
            if not validation.is_valid:
                self.metrics.failed_validations += 1
            if validation.risk_score >= 0.7:
                self.metrics.high_risk_operations += 1
            
            return validation
            
        except Exception as e:
            raise SecurityError(f"Strategy validation failed: {str(e)}")
            
    async def validate_protocol(self, protocol_id: ProtocolId) -> SecurityValidation:
        """Validate a protocol for security concerns.
        
        Args:
            protocol_id: Protocol to validate
            
        Returns:
            SecurityValidation result
            
        Raises:
            SecurityError: If validation fails
        """
        try:
            validation = SecurityValidation(is_valid=True)
            
            # Load protocol security data
            security_data = await self._load_protocol_security(protocol_id)
            
            # Check required audits
            if self.config.required_audits and not security_data.get('audited', False):
                validation.is_valid = False
                validation.reason = "Protocol not audited"
                validation.risk_score = 1.0
                return validation
            
            # Check timelock
            if self.config.required_timelock:
                timelock = int(security_data.get('timelock', 0))
                if timelock < self.config.min_timelock:
                    validation.warnings.append(
                        f"Short timelock period: {timelock/3600:.1f} hours"
                    )
                    validation.risk_score += 0.4
            
            # Check TVL trend
            tvl_change = float(security_data.get('tvl_change_24h', 0))
            if tvl_change < -0.2:  # 20% TVL drop
                validation.warnings.append(
                    f"Significant TVL decrease: {tvl_change*100:.1f}%"
                )
                validation.risk_score += 0.3
            
            # Update metrics
            self.metrics.total_validations += 1
            if not validation.is_valid:
                self.metrics.failed_validations += 1
            if validation.risk_score >= 0.7:
                self.metrics.high_risk_operations += 1
            
            return validation
            
        except Exception as e:
            raise SecurityError(f"Protocol validation failed: {str(e)}")
            
    async def validate_flash_loan(self, strategy: Dict[str, Any]) -> SecurityValidation:
        """Validate a flash loan strategy for security concerns.
        
        Args:
            strategy: Flash loan strategy to validate
            
        Returns:
            SecurityValidation result
            
        Raises:
            SecurityError: If validation fails
        """
        try:
            validation = SecurityValidation(is_valid=True)
            
            # Check loan ratio
            loan_amount = float(strategy.get('loan_amount', 0))
            collateral = float(strategy.get('collateral', 0))
            if collateral > 0:
                ratio = loan_amount / collateral
                if ratio > self.config.max_flash_loan_ratio:
                    validation.is_valid = False
                    validation.reason = f"Loan ratio {ratio:.2f} exceeds maximum {self.config.max_flash_loan_ratio}"
                    validation.risk_score = 1.0
                    return validation
            
            # Check protocol security
            protocol_id = strategy.get('protocol_id')
            if protocol_id:
                protocol_validation = await self.validate_protocol(protocol_id)
                if not protocol_validation.is_valid:
                    validation.is_valid = False
                    validation.reason = f"Protocol validation failed: {protocol_validation.reason}"
                    validation.risk_score = protocol_validation.risk_score
                    return validation
                validation.warnings.extend(protocol_validation.warnings)
                validation.risk_score = max(validation.risk_score, protocol_validation.risk_score)
            
            # Update metrics
            self.metrics.total_validations += 1
            if not validation.is_valid:
                self.metrics.failed_validations += 1
            if validation.risk_score >= 0.7:
                self.metrics.high_risk_operations += 1
            
            return validation
            
        except Exception as e:
            raise SecurityError(f"Flash loan validation failed: {str(e)}")
            
    async def validate_lending_protocol(self, protocol_id: ProtocolId) -> SecurityValidation:
        """Validate a lending protocol for security concerns.
        
        Args:
            protocol_id: Protocol to validate
            
        Returns:
            SecurityValidation result
            
        Raises:
            SecurityError: If validation fails
        """
        try:
            # Start with basic protocol validation
            validation = await self.validate_protocol(protocol_id)
            if not validation.is_valid:
                return validation
            
            # Load lending-specific security data
            security_data = await self._load_protocol_security(protocol_id)
            
            # Check liquidation ratio
            min_ratio = float(security_data.get('min_liquidation_ratio', 1.5))
            if min_ratio < 1.2:  # Minimum safe ratio
                validation.warnings.append(
                    f"Low liquidation ratio: {min_ratio:.2f}"
                )
                validation.risk_score += 0.3
            
            # Check oracle security
            oracle_score = float(security_data.get('oracle_security_score', 1.0))
            if oracle_score < 0.8:
                validation.warnings.append(
                    f"Oracle security concerns: {oracle_score:.2f}/1.0"
                )
                validation.risk_score += 0.4
            
            # Update metrics
            if validation.risk_score >= 0.7:
                self.metrics.high_risk_operations += 1
            
            return validation
            
        except Exception as e:
            raise SecurityError(f"Lending protocol validation failed: {str(e)}")
            
    async def validate_liquidity_pool(
        self,
        pool_address: str,
        chain_id: ChainId
    ) -> SecurityValidation:
        """Validate a liquidity pool for security concerns.
        
        Args:
            pool_address: Address of the pool to validate
            chain_id: Chain ID where the pool exists
            
        Returns:
            SecurityValidation result
            
        Raises:
            SecurityError: If validation fails
        """
        try:
            validation = SecurityValidation(is_valid=True)
            
            # Check if pool is blacklisted
            if pool_address in self.config.blacklisted_addresses:
                validation.is_valid = False
                validation.reason = "Pool address is blacklisted"
                validation.risk_score = 1.0
                return validation
            
            # Load pool security data
            security_data = await self._load_pool_security(pool_address, chain_id)
            
            # Check liquidity
            liquidity = float(security_data.get('liquidity_usd', 0))
            if liquidity < self.config.min_liquidity:
                validation.warnings.append(
                    f"Low liquidity: ${liquidity:,.2f}"
                )
                validation.risk_score += 0.3
            
            # Check volume
            volume = float(security_data.get('volume_24h_usd', 0))
            if volume < liquidity * 0.1:  # Less than 10% daily turnover
                validation.warnings.append(
                    f"Low trading volume: ${volume:,.2f}"
                )
                validation.risk_score += 0.2
            
            # Check smart contract security
            contract_score = float(security_data.get('contract_security_score', 1.0))
            if contract_score < 0.8:
                validation.warnings.append(
                    f"Contract security concerns: {contract_score:.2f}/1.0"
                )
                validation.risk_score += 0.4
            
            # Update metrics
            self.metrics.total_validations += 1
            if not validation.is_valid:
                self.metrics.failed_validations += 1
            if validation.risk_score >= 0.7:
                self.metrics.high_risk_operations += 1
            
            return validation
            
        except Exception as e:
            raise SecurityError(f"Liquidity pool validation failed: {str(e)}")
            
    async def _load_protocol_security(self, protocol_id: ProtocolId) -> Dict[str, Any]:
        """Load security data for a protocol.
        
        Args:
            protocol_id: Protocol to load data for
            
        Returns:
            Protocol security data
            
        Raises:
            SecurityError: If data loading fails
        """
        try:
            config_path = Path(__file__).parent / 'data' / 'protocol_security.json'
            with open(config_path) as f:
                security_data = json.load(f)
            return security_data.get(protocol_id, {})
        except Exception as e:
            raise SecurityError(f"Failed to load protocol security data: {str(e)}")
            
    async def _load_pool_security(
        self,
        pool_address: str,
        chain_id: ChainId
    ) -> Dict[str, Any]:
        """Load security data for a liquidity pool.
        
        Args:
            pool_address: Pool address to load data for
            chain_id: Chain ID where the pool exists
            
        Returns:
            Pool security data
            
        Raises:
            SecurityError: If data loading fails
        """
        try:
            config_path = Path(__file__).parent / 'data' / 'pool_security.json'
            with open(config_path) as f:
                security_data = json.load(f)
            return security_data.get(f"{chain_id}:{pool_address.lower()}", {})
        except Exception as e:
            raise SecurityError(f"Failed to load pool security data: {str(e)}")
            
    def get_metrics(self) -> SecurityMetrics:
        """Get current security metrics.
        
        Returns:
            Current security metrics
        """
        if self.metrics.total_validations > 0:
            self.metrics.average_risk_score = (
                self.metrics.failed_validations / self.metrics.total_validations
            )
        return self.metrics 