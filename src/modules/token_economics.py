"""
Token Economics Module

This module provides token economic analysis capabilities:
- Token validation
- Economic analysis
- Market cap verification
- Liquidity analysis
- Holder distribution analysis
- Price impact estimation
- Volume analysis
- Supply distribution
- Risk assessment
- Integration with monitoring
"""

import asyncio
from typing import Dict, List, Optional, Tuple, NamedTuple, cast, Any
import structlog
from web3 import Web3
from web3.types import BlockData, EventData, LogReceipt
from prometheus_client import Histogram, Gauge, Counter
from ..utils.cache import cache
from ..utils.rate_limiter import rate_limiter, Priority
import aiohttp
from eth_typing import ChecksumAddress
from eth_utils.address import to_checksum_address
from datetime import datetime, timedelta

logger = structlog.get_logger(__name__)

class TokenMetrics(NamedTuple):
    """Token metrics data structure"""
    market_cap: float
    total_supply: int
    circulating_supply: int
    holders_count: int
    liquidity_usd: float
    volume_24h: float
    price_usd: float
    market_dominance: float
    volatility_24h: float
    fully_diluted_valuation: float

class HolderMetrics(NamedTuple):
    """Holder metrics data structure"""
    address: ChecksumAddress
    balance: int
    percentage: float
    last_transfer: datetime
    transfer_count: int
    avg_holding_time: float

class RiskMetrics(NamedTuple):
    """Risk assessment metrics"""
    liquidity_risk: float  # 0-1, higher is riskier
    concentration_risk: float  # 0-1, higher is riskier
    volatility_risk: float  # 0-1, higher is riskier
    volume_risk: float  # 0-1, higher is riskier
    overall_risk: float  # 0-1, higher is riskier

class TokenEconomics:
    def __init__(self, web3: Web3):
        self.web3 = web3
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        self._token_metrics = Gauge(
            'token_metrics',
            'Token economic metrics',
            ['token_address', 'metric']
        )
        self._token_analysis_time = Histogram(
            'token_analysis_seconds',
            'Time spent on token analysis',
            ['analysis_type']
        )
        self._risk_metrics = Gauge(
            'token_risk_metrics',
            'Token risk assessment metrics',
            ['token_address', 'risk_type']
        )
        self._holder_metrics = Gauge(
            'token_holder_metrics',
            'Token holder metrics',
            ['token_address', 'holder_type']
        )

    @cache.memoize(ttl=300)  # Cache for 5 minutes
    async def get_token_metrics(self, 
                              token_address: str,
                              chain_id: int = 1) -> TokenMetrics:
        """Get comprehensive token metrics"""
        start_time = asyncio.get_event_loop().time()
        try:
            # Use high priority for important metrics
            await rate_limiter.acquire("defillama", Priority.HIGH)
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://api.defillama.com/v2/token/{chain_id}/{token_address}"
                ) as response:
                    data = await response.json()
                
                metrics = TokenMetrics(
                    market_cap=float(data.get('market_cap', 0)),
                    total_supply=int(data.get('total_supply', 0)),
                    circulating_supply=int(data.get('circulating_supply', 0)),
                    holders_count=int(data.get('holders', 0)),
                    liquidity_usd=float(data.get('liquidity', 0)),
                    volume_24h=float(data.get('volume24h', 0)),
                    price_usd=float(data.get('price', 0)),
                    market_dominance=float(data.get('market_dominance', 0)),
                    volatility_24h=float(data.get('volatility24h', 0)),
                    fully_diluted_valuation=float(data.get('fdv', 0))
                )
                
                # Update Prometheus metrics
                for field, value in metrics._asdict().items():
                    self._token_metrics.labels(
                        token_address=token_address,
                        metric=field
                    ).set(float(value))
                
                return metrics
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self._token_analysis_time.labels(
                analysis_type="metrics"
            ).observe(duration)

    async def validate_token(self, 
                           token_address: str,
                           chain_id: int = 1) -> Tuple[bool, str]:
        """Validate token based on economic metrics"""
        metrics = await self.get_token_metrics(token_address, chain_id)
        risk = await self.assess_risk(token_address, chain_id)
        
        # Define validation criteria
        validations = [
            (metrics.market_cap >= 1000000, "Market cap too low"),
            (metrics.liquidity_usd >= 100000, "Insufficient liquidity"),
            (metrics.holders_count >= 1000, "Too few holders"),
            (metrics.volume_24h >= 50000, "Insufficient trading volume"),
            (risk.overall_risk <= 0.7, "Risk assessment too high")
        ]
        
        # Check all criteria
        for is_valid, reason in validations:
            if not is_valid:
                return False, reason
        
        return True, "Token passes all validation criteria"

    @cache.memoize(ttl=3600)
    async def analyze_holder_distribution(self,
                                       token_address: str,
                                       chain_id: int = 1) -> Dict[ChecksumAddress, HolderMetrics]:
        """Analyze token holder distribution"""
        start_time = asyncio.get_event_loop().time()
        try:
            # Rate limit API calls
            await rate_limiter.acquire("etherscan", Priority.MEDIUM)
            
            # Get top holders
            checksum_address = to_checksum_address(token_address)
            transfer_event_abi = {
                'anonymous': False,
                'inputs': [
                    {'indexed': True, 'name': 'from', 'type': 'address'},
                    {'indexed': True, 'name': 'to', 'type': 'address'},
                    {'indexed': False, 'name': 'value', 'type': 'uint256'}
                ],
                'name': 'Transfer',
                'type': 'event'
            }
            
            contract = self.web3.eth.contract(
                address=checksum_address,
                abi=[transfer_event_abi]
            )
            
            # Get block number asynchronously
            current_block = await asyncio.to_thread(
                lambda: self.web3.eth.block_number
            )
            
            # Analyze transfer events
            event_filter = contract.events.Transfer.create_filter(
                fromBlock=current_block - 1000000
            )
            
            # Get logs asynchronously
            logs = await asyncio.to_thread(
                lambda: event_filter.get_all_entries()
            )
            
            # Process holder data
            holder_data: Dict[ChecksumAddress, Dict] = {}
            total_supply = 0
            
            for log in logs:
                event_data = cast(EventData, log)
                args = event_data['args']
                if not args:
                    continue
                    
                from_addr = to_checksum_address(args['from'])
                to_addr = to_checksum_address(args['to'])
                value = int(args['value'])
                timestamp = datetime.fromtimestamp(
                    self.web3.eth.get_block(log['blockNumber'])['timestamp']
                )
                
                # Update sender data
                if from_addr not in holder_data:
                    holder_data[from_addr] = {
                        'balance': 0,
                        'transfers': 0,
                        'first_transfer': timestamp,
                        'last_transfer': timestamp
                    }
                holder_data[from_addr]['balance'] -= value
                holder_data[from_addr]['transfers'] += 1
                holder_data[from_addr]['last_transfer'] = timestamp
                
                # Update receiver data
                if to_addr not in holder_data:
                    holder_data[to_addr] = {
                        'balance': 0,
                        'transfers': 0,
                        'first_transfer': timestamp,
                        'last_transfer': timestamp
                    }
                holder_data[to_addr]['balance'] += value
                holder_data[to_addr]['transfers'] += 1
                holder_data[to_addr]['last_transfer'] = timestamp
                
                total_supply = max(total_supply, holder_data[to_addr]['balance'])
            
            # Convert to HolderMetrics
            result: Dict[ChecksumAddress, HolderMetrics] = {}
            for addr, data in holder_data.items():
                if data['balance'] <= 0:
                    continue
                    
                holding_time = (data['last_transfer'] - data['first_transfer']).total_seconds()
                result[addr] = HolderMetrics(
                    address=addr,
                    balance=data['balance'],
                    percentage=(data['balance'] / total_supply * 100) if total_supply > 0 else 0,
                    last_transfer=data['last_transfer'],
                    transfer_count=data['transfers'],
                    avg_holding_time=holding_time / data['transfers'] if data['transfers'] > 0 else 0
                )
                
                # Update holder metrics
                self._holder_metrics.labels(
                    token_address=token_address,
                    holder_type='balance'
                ).set(result[addr].balance)
                
            return result
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self._token_analysis_time.labels(
                analysis_type="holder_distribution"
            ).observe(duration)

    async def assess_risk(self,
                         token_address: str,
                         chain_id: int = 1) -> RiskMetrics:
        """Assess token risk metrics"""
        metrics = await self.get_token_metrics(token_address, chain_id)
        holders = await self.analyze_holder_distribution(token_address, chain_id)
        
        # Calculate liquidity risk
        liquidity_risk = max(0, min(1, 1000000 / metrics.liquidity_usd if metrics.liquidity_usd > 0 else 1))
        
        # Calculate concentration risk
        top_holder_percentage = sum(
            holder.percentage for holder in sorted(
                holders.values(),
                key=lambda x: x.percentage,
                reverse=True
            )[:5]  # Top 5 holders
        )
        concentration_risk = top_holder_percentage / 100
        
        # Calculate volatility risk
        volatility_risk = min(1, metrics.volatility_24h / 100)
        
        # Calculate volume risk
        volume_risk = max(0, min(1, 1000000 / metrics.volume_24h if metrics.volume_24h > 0 else 1))
        
        # Calculate overall risk
        overall_risk = (liquidity_risk + concentration_risk + volatility_risk + volume_risk) / 4
        
        risk_metrics = RiskMetrics(
            liquidity_risk=liquidity_risk,
            concentration_risk=concentration_risk,
            volatility_risk=volatility_risk,
            volume_risk=volume_risk,
            overall_risk=overall_risk
        )
        
        # Update risk metrics
        for field, value in risk_metrics._asdict().items():
            self._risk_metrics.labels(
                token_address=token_address,
                risk_type=field
            ).set(value)
        
        return risk_metrics

    async def estimate_price_impact(self,
                                  token_address: str,
                                  amount_usd: float,
                                  chain_id: int = 1) -> float:
        """Estimate price impact of a trade"""
        start_time = asyncio.get_event_loop().time()
        try:
            metrics = await self.get_token_metrics(token_address, chain_id)
            
            # Enhanced price impact estimation considering volume
            base_impact = (amount_usd / metrics.liquidity_usd) * 100 if metrics.liquidity_usd > 0 else 100.0
            volume_factor = min(1, amount_usd / metrics.volume_24h) if metrics.volume_24h > 0 else 1
            
            impact = base_impact * (1 + volume_factor)
            return min(impact, 100.0)  # Cap at 100%
        finally:
            duration = asyncio.get_event_loop().time() - start_time
            self._token_analysis_time.labels(
                analysis_type="price_impact"
            ).observe(duration)

# Global token economics analyzer instance
token_economics = None  # Initialize in application startup 