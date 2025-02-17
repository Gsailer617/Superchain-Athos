from typing import Dict, Any, Optional, Protocol, Type, List, Union
from abc import ABC, abstractmethod
import logging
from web3 import Web3
from web3.types import TxParams, Wei
from eth_typing import HexAddress
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

@dataclass
class ProtocolConfig:
    """Protocol-specific configuration"""
    name: str
    version: str
    supported_chains: List[str]
    fee_tier_options: List[int]
    default_fee_tier: int
    slippage_tolerance: float = 0.005  # 0.5%
    deadline_seconds: int = 1200  # 20 minutes
    # Added protocol-specific configurations
    min_liquidity_usd: float = 100000  # Minimum pool liquidity in USD
    max_price_impact: float = 0.01  # Maximum price impact (1%)
    min_pool_utilization: float = 0.05  # Minimum pool utilization (5%)
    health_check_interval: int = 300  # Health check interval in seconds

class PoolHealth(Enum):
    """Pool health status"""
    HEALTHY = "healthy"
    LOW_LIQUIDITY = "low_liquidity"
    HIGH_VOLATILITY = "high_volatility"
    MANIPULATED = "manipulated"
    INACTIVE = "inactive"

@dataclass
class PoolMetrics:
    """Pool performance metrics"""
    liquidity_usd: float
    volume_24h: float
    fees_24h: float
    price_impact: float
    volatility_24h: float
    utilization: float
    last_trade_timestamp: float
    last_error: Optional[str] = None

class ProtocolAdapter(ABC):
    """Base class for protocol-specific adapters"""
    
    def __init__(self, config: ProtocolConfig, web3: Web3):
        self.config = config
        self.web3 = web3
        self.pool_metrics: Dict[str, PoolMetrics] = {}
        self.last_health_check: Dict[str, float] = {}
    
    @abstractmethod
    def validate_pool(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> bool:
        """Validate if pool exists and is active"""
        pass
    
    @abstractmethod
    def get_pool_address(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> str:
        """Get pool address for token pair"""
        pass
    
    @abstractmethod
    def estimate_output(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        fee_tier: Optional[int] = None
    ) -> int:
        """Estimate output amount for swap"""
        pass
    
    @abstractmethod
    def prepare_swap(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        min_amount_out: int,
        recipient: str,
        fee_tier: Optional[int] = None
    ) -> TxParams:
        """Prepare swap transaction"""
        pass
    
    @abstractmethod
    def check_pool_health(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> PoolHealth:
        """Check pool health status"""
        pass
    
    @abstractmethod
    def optimize_swap_params(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        fee_tier: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize swap parameters"""
        pass
    
    @abstractmethod
    def monitor_pool_state(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> Dict[str, Any]:
        """Monitor pool state"""
        pass
    
    @abstractmethod
    def validate_price_impact(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        amount_out: int,
        fee_tier: Optional[int] = None
    ) -> bool:
        """Validate price impact of swap"""
        pass

class UniswapV3Adapter(ProtocolAdapter):
    """Adapter for Uniswap V3"""
    
    def __init__(self, config: ProtocolConfig, web3: Web3):
        super().__init__(config, web3)
        self.router_addresses = {
            'ethereum': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'polygon': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
            'base': '0x2626664c2603336E57B271c5C0b26F421741e481'
        }
        self.factory_addresses = {
            'ethereum': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
            'polygon': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
            'base': '0x33128a8fC17869897dcE68Ed026d694621f6FDfD'
        }
        self.quoter_addresses = {
            'ethereum': '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6',
            'polygon': '0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6',
            'base': '0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a'
        }
    
    def validate_pool(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> bool:
        """Validate Uniswap V3 pool"""
        if chain not in self.config.supported_chains:
            return False
        if fee_tier and fee_tier not in self.config.fee_tier_options:
            return False
        # Implementation would check if pool exists and has liquidity
        return True
    
    def get_pool_address(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> str:
        """Get Uniswap V3 pool address"""
        # Implementation would compute pool address
        return "0x1234..."
    
    def estimate_output(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        fee_tier: Optional[int] = None
    ) -> int:
        """Estimate Uniswap V3 output"""
        # Implementation would calculate actual output
        return int(amount_in * 0.997)  # Example with 0.3% fee
    
    def prepare_swap(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        min_amount_out: int,
        recipient: str,
        fee_tier: Optional[int] = None
    ) -> TxParams:
        """Prepare Uniswap V3 swap"""
        return {
            'to': self.router_addresses[chain],
            'data': self._encode_swap_data(
                token_in,
                token_out,
                amount_in,
                min_amount_out,
                recipient,
                fee_tier or self.config.default_fee_tier
            ),
            'value': Wei(0)
        }
    
    def _encode_swap_data(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        min_amount_out: int,
        recipient: str,
        fee_tier: int
    ) -> bytes:
        """Encode Uniswap V3 swap data"""
        # Implementation would encode actual swap data
        return b""
    
    def check_pool_health(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> PoolHealth:
        """Check Uniswap V3 pool health"""
        try:
            pool_address = self.get_pool_address(chain, token0, token1, fee_tier)
            metrics = self._get_pool_metrics(pool_address)
            
            # Check liquidity
            if metrics.liquidity_usd < self.config.min_liquidity_usd:
                return PoolHealth.LOW_LIQUIDITY
            
            # Check volatility
            if metrics.volatility_24h > 0.1:  # 10% daily volatility
                return PoolHealth.HIGH_VOLATILITY
            
            # Check manipulation
            if self._detect_manipulation(metrics):
                return PoolHealth.MANIPULATED
            
            # Check activity
            if time.time() - metrics.last_trade_timestamp > 3600:  # 1 hour
                return PoolHealth.INACTIVE
            
            return PoolHealth.HEALTHY
            
        except Exception as e:
            logger.error(f"Error checking pool health: {str(e)}")
            return PoolHealth.INACTIVE
    
    def optimize_swap_params(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        fee_tier: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize Uniswap V3 swap parameters"""
        try:
            # Get pool metrics
            pool_address = self.get_pool_address(chain, token_in, token_out, fee_tier)
            metrics = self._get_pool_metrics(pool_address)
            
            # Optimize fee tier
            optimal_fee = self._find_optimal_fee_tier(
                chain,
                token_in,
                token_out,
                amount_in,
                metrics
            )
            
            # Calculate optimal slippage
            optimal_slippage = self._calculate_optimal_slippage(metrics)
            
            # Determine deadline
            deadline = self._calculate_deadline(metrics)
            
            return {
                'fee_tier': optimal_fee,
                'slippage_tolerance': optimal_slippage,
                'deadline': deadline,
                'split_routes': self._should_split_route(amount_in, metrics)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing swap params: {str(e)}")
            return {
                'fee_tier': fee_tier or self.config.default_fee_tier,
                'slippage_tolerance': self.config.slippage_tolerance,
                'deadline': int(time.time() + self.config.deadline_seconds),
                'split_routes': False
            }
    
    def monitor_pool_state(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> Dict[str, Any]:
        """Monitor Uniswap V3 pool state"""
        try:
            pool_address = self.get_pool_address(chain, token0, token1, fee_tier)
            metrics = self._get_pool_metrics(pool_address)
            
            # Update metrics cache
            self.pool_metrics[pool_address] = metrics
            self.last_health_check[pool_address] = time.time()
            
            return {
                'liquidity': metrics.liquidity_usd,
                'volume': metrics.volume_24h,
                'fees': metrics.fees_24h,
                'utilization': metrics.utilization,
                'price_impact': metrics.price_impact,
                'volatility': metrics.volatility_24h,
                'health': self.check_pool_health(chain, token0, token1, fee_tier)
            }
            
        except Exception as e:
            logger.error(f"Error monitoring pool state: {str(e)}")
            return {}
    
    def validate_price_impact(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        amount_out: int,
        fee_tier: Optional[int] = None
    ) -> bool:
        """Validate Uniswap V3 price impact"""
        try:
            # Get pool metrics
            pool_address = self.get_pool_address(chain, token_in, token_out, fee_tier)
            metrics = self._get_pool_metrics(pool_address)
            
            # Calculate price impact
            price_impact = self._calculate_price_impact(
                amount_in,
                amount_out,
                metrics
            )
            
            # Update metrics
            metrics.price_impact = price_impact
            self.pool_metrics[pool_address] = metrics
            
            return price_impact <= self.config.max_price_impact
            
        except Exception as e:
            logger.error(f"Error validating price impact: {str(e)}")
            return False
    
    def _get_pool_metrics(self, pool_address: str) -> PoolMetrics:
        """Get Uniswap V3 pool metrics"""
        # Implementation would get actual metrics from pool
        return PoolMetrics(
            liquidity_usd=1000000,
            volume_24h=100000,
            fees_24h=1000,
            price_impact=0.001,
            volatility_24h=0.05,
            utilization=0.1,
            last_trade_timestamp=time.time()
        )
    
    def _detect_manipulation(self, metrics: PoolMetrics) -> bool:
        """Detect pool manipulation"""
        # Implementation would use actual manipulation detection logic
        return False
    
    def _find_optimal_fee_tier(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        metrics: PoolMetrics
    ) -> int:
        """Find optimal fee tier"""
        # Implementation would calculate optimal fee tier
        return self.config.default_fee_tier
    
    def _calculate_optimal_slippage(self, metrics: PoolMetrics) -> float:
        """Calculate optimal slippage tolerance"""
        # Implementation would calculate optimal slippage
        return self.config.slippage_tolerance
    
    def _calculate_deadline(self, metrics: PoolMetrics) -> int:
        """Calculate optimal deadline"""
        # Implementation would calculate optimal deadline
        return int(time.time() + self.config.deadline_seconds)
    
    def _should_split_route(self, amount_in: int, metrics: PoolMetrics) -> bool:
        """Determine if route should be split"""
        # Implementation would determine if route should be split
        return False
    
    def _calculate_price_impact(
        self,
        amount_in: int,
        amount_out: int,
        metrics: PoolMetrics
    ) -> float:
        """Calculate price impact"""
        # Implementation would calculate actual price impact
        return 0.001

class CurveAdapter(ProtocolAdapter):
    """Adapter for Curve"""
    
    def __init__(self, config: ProtocolConfig, web3: Web3):
        super().__init__(config, web3)
        self.registry_addresses = {
            'ethereum': '0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5',
            'polygon': '0x47bB542B9dE58b970bA50c9dae444DDB4c16751a',
            'base': '0x4a4962275DF8C60a9Bde9FA09397a505c5B0d0B7'
        }
    
    def validate_pool(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> bool:
        """Validate Curve pool"""
        if chain not in self.config.supported_chains:
            return False
        # Implementation would check if pool exists
        return True
    
    def get_pool_address(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> str:
        """Get Curve pool address"""
        # Implementation would get pool from registry
        return "0x5678..."
    
    def estimate_output(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        fee_tier: Optional[int] = None
    ) -> int:
        """Estimate Curve output"""
        # Implementation would calculate actual output
        return int(amount_in * 0.999)  # Example with 0.1% fee
    
    def prepare_swap(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        min_amount_out: int,
        recipient: str,
        fee_tier: Optional[int] = None
    ) -> TxParams:
        """Prepare Curve swap"""
        pool_address = self.get_pool_address(chain, token_in, token_out)
        return {
            'to': pool_address,
            'data': self._encode_swap_data(
                token_in,
                token_out,
                amount_in,
                min_amount_out,
                recipient
            ),
            'value': Wei(0)
        }
    
    def _encode_swap_data(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        min_amount_out: int,
        recipient: str
    ) -> bytes:
        """Encode Curve swap data"""
        # Implementation would encode actual swap data
        return b""
    
    def check_pool_health(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> PoolHealth:
        """Check Curve pool health"""
        try:
            pool_address = self.get_pool_address(chain, token0, token1)
            metrics = self._get_pool_metrics(pool_address)
            
            # Check liquidity
            if metrics.liquidity_usd < self.config.min_liquidity_usd:
                return PoolHealth.LOW_LIQUIDITY
            
            # Check volatility
            if metrics.volatility_24h > 0.05:  # 5% daily volatility
                return PoolHealth.HIGH_VOLATILITY
            
            # Check manipulation
            if self._detect_manipulation(metrics):
                return PoolHealth.MANIPULATED
            
            # Check activity
            if time.time() - metrics.last_trade_timestamp > 3600:  # 1 hour
                return PoolHealth.INACTIVE
            
            return PoolHealth.HEALTHY
            
        except Exception as e:
            logger.error(f"Error checking pool health: {str(e)}")
            return PoolHealth.INACTIVE
    
    def optimize_swap_params(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        fee_tier: Optional[int] = None
    ) -> Dict[str, Any]:
        """Optimize Curve swap parameters"""
        try:
            # Get pool metrics
            pool_address = self.get_pool_address(chain, token_in, token_out)
            metrics = self._get_pool_metrics(pool_address)
            
            # Calculate optimal slippage
            optimal_slippage = self._calculate_optimal_slippage(metrics)
            
            # Determine deadline
            deadline = self._calculate_deadline(metrics)
            
            return {
                'slippage_tolerance': optimal_slippage,
                'deadline': deadline,
                'use_underlying': self._should_use_underlying(metrics)
            }
            
        except Exception as e:
            logger.error(f"Error optimizing swap params: {str(e)}")
            return {
                'slippage_tolerance': self.config.slippage_tolerance,
                'deadline': int(time.time() + self.config.deadline_seconds),
                'use_underlying': False
            }
    
    def monitor_pool_state(
        self,
        chain: str,
        token0: str,
        token1: str,
        fee_tier: Optional[int] = None
    ) -> Dict[str, Any]:
        """Monitor Curve pool state"""
        try:
            pool_address = self.get_pool_address(chain, token0, token1)
            metrics = self._get_pool_metrics(pool_address)
            
            # Update metrics cache
            self.pool_metrics[pool_address] = metrics
            self.last_health_check[pool_address] = time.time()
            
            return {
                'liquidity': metrics.liquidity_usd,
                'volume': metrics.volume_24h,
                'fees': metrics.fees_24h,
                'utilization': metrics.utilization,
                'price_impact': metrics.price_impact,
                'volatility': metrics.volatility_24h,
                'health': self.check_pool_health(chain, token0, token1)
            }
            
        except Exception as e:
            logger.error(f"Error monitoring pool state: {str(e)}")
            return {}
    
    def validate_price_impact(
        self,
        chain: str,
        token_in: str,
        token_out: str,
        amount_in: int,
        amount_out: int,
        fee_tier: Optional[int] = None
    ) -> bool:
        """Validate Curve price impact"""
        try:
            # Get pool metrics
            pool_address = self.get_pool_address(chain, token_in, token_out)
            metrics = self._get_pool_metrics(pool_address)
            
            # Calculate price impact
            price_impact = self._calculate_price_impact(
                amount_in,
                amount_out,
                metrics
            )
            
            # Update metrics
            metrics.price_impact = price_impact
            self.pool_metrics[pool_address] = metrics
            
            return price_impact <= self.config.max_price_impact
            
        except Exception as e:
            logger.error(f"Error validating price impact: {str(e)}")
            return False
    
    def _get_pool_metrics(self, pool_address: str) -> PoolMetrics:
        """Get Curve pool metrics"""
        # Implementation would get actual metrics from pool
        return PoolMetrics(
            liquidity_usd=1000000,
            volume_24h=100000,
            fees_24h=1000,
            price_impact=0.001,
            volatility_24h=0.05,
            utilization=0.1,
            last_trade_timestamp=time.time()
        )
    
    def _detect_manipulation(self, metrics: PoolMetrics) -> bool:
        """Detect pool manipulation"""
        # Implementation would use actual manipulation detection logic
        return False
    
    def _calculate_optimal_slippage(self, metrics: PoolMetrics) -> float:
        """Calculate optimal slippage tolerance"""
        # Implementation would calculate optimal slippage
        return self.config.slippage_tolerance
    
    def _calculate_deadline(self, metrics: PoolMetrics) -> int:
        """Calculate optimal deadline"""
        # Implementation would calculate optimal deadline
        return int(time.time() + self.config.deadline_seconds)
    
    def _should_use_underlying(self, metrics: PoolMetrics) -> bool:
        """Determine if underlying tokens should be used"""
        # Implementation would determine if underlying tokens should be used
        return False
    
    def _calculate_price_impact(
        self,
        amount_in: int,
        amount_out: int,
        metrics: PoolMetrics
    ) -> float:
        """Calculate price impact"""
        # Implementation would calculate actual price impact
        return 0.001

class ProtocolAdapterFactory:
    """Factory for creating protocol-specific adapters"""
    
    _adapters: Dict[str, Type[ProtocolAdapter]] = {
        'uniswap_v3': UniswapV3Adapter,
        'curve': CurveAdapter
    }
    
    @classmethod
    def get_adapter(cls, protocol: str, config: ProtocolConfig, web3: Web3) -> ProtocolAdapter:
        """Get appropriate adapter for protocol"""
        adapter_class = cls._adapters.get(protocol.lower())
        if not adapter_class:
            raise ValueError(f"No adapter available for protocol: {protocol}")
        return adapter_class(config, web3)
    
    @classmethod
    def register_adapter(cls, protocol: str, adapter: Type[ProtocolAdapter]) -> None:
        """Register new protocol adapter"""
        cls._adapters[protocol.lower()] = adapter 