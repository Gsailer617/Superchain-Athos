"""
Token Discovery Domain Model

This module defines the domain model for token discovery and arbitrage:
- Token aggregate for managing token state
- Events for token lifecycle
- Event handlers for token-related events
"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from uuid import uuid4

from pydantic import BaseModel, Field

from .base import Event, EventHandler
from .aggregates import Aggregate, AggregateState
from ..decorators import handles_event
from ...core.dependency_injector import container

# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------

class TokenType(str, Enum):
    """Token types"""
    ERC20 = "erc20"
    ERC721 = "erc721"
    ERC1155 = "erc1155"

class TokenStatus(str, Enum):
    """Token validation status"""
    DISCOVERED = "discovered"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    BLACKLISTED = "blacklisted"

class LiquiditySource(str, Enum):
    """Liquidity sources"""
    UNISWAP_V2 = "uniswap_v2"
    UNISWAP_V3 = "uniswap_v3"
    SUSHISWAP = "sushiswap"
    PANCAKESWAP = "pancakeswap"
    BALANCER = "balancer"
    CURVE = "curve"
    DODO = "dodo"
    CUSTOM = "custom"

class SecurityRisk(str, Enum):
    """Security risk types"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SecurityCheck(BaseModel):
    """Security check result"""
    name: str
    passed: bool
    risk_level: SecurityRisk = SecurityRisk.NONE
    details: Dict[str, Any] = Field(default_factory=dict)

class LiquidityPool(BaseModel):
    """Liquidity pool information"""
    address: str
    source: LiquiditySource
    pair_with: str  # Token address paired with
    total_liquidity_usd: Decimal
    token_reserves: Decimal
    pair_reserves: Decimal
    last_updated: datetime = Field(default_factory=datetime.now)

class PricePoint(BaseModel):
    """Price point information"""
    timestamp: datetime
    price_usd: Decimal
    source: str
    volume_24h: Optional[Decimal] = None

# -----------------------------------------------------------------------------
# Aggregate State
# -----------------------------------------------------------------------------

class TokenState(AggregateState):
    """State for token aggregate"""
    chain_id: int
    address: str
    name: Optional[str] = None
    symbol: Optional[str] = None
    decimals: Optional[int] = None
    token_type: TokenType = TokenType.ERC20
    status: TokenStatus = TokenStatus.DISCOVERED
    discovery_source: str = "unknown"
    discovery_time: datetime = Field(default_factory=datetime.now)
    validation_time: Optional[datetime] = None
    total_supply: Optional[Decimal] = None
    circulating_supply: Optional[Decimal] = None
    owner_address: Optional[str] = None
    creator_address: Optional[str] = None
    contract_verified: bool = False
    security_checks: List[SecurityCheck] = Field(default_factory=list)
    liquidity_pools: List[LiquidityPool] = Field(default_factory=list)
    price_history: List[PricePoint] = Field(default_factory=list)
    current_price_usd: Optional[Decimal] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_honeypot: bool = False
    is_mintable: bool = False
    is_burnable: bool = False
    is_pausable: bool = False
    has_blacklist: bool = False
    has_whitelist: bool = False
    has_anti_whale: bool = False
    max_transaction_amount: Optional[Decimal] = None
    max_wallet_amount: Optional[Decimal] = None
    buy_tax: Optional[Decimal] = None
    sell_tax: Optional[Decimal] = None
    transfer_tax: Optional[Decimal] = None

# -----------------------------------------------------------------------------
# Token Aggregate
# -----------------------------------------------------------------------------

class TokenAggregate(Aggregate[TokenState]):
    """Aggregate for managing token state"""
    
    @classmethod
    async def create_token(
        cls,
        chain_id: int,
        address: str,
        discovery_source: str,
        token_type: TokenType = TokenType.ERC20
    ) -> "TokenAggregate":
        """Create a new token
        
        Args:
            chain_id: Chain ID
            address: Token address
            discovery_source: Source of discovery
            token_type: Token type
            
        Returns:
            Token aggregate
        """
        # Create aggregate
        aggregate = await cls.create(
            chain_id=chain_id,
            address=address,
            status=TokenStatus.DISCOVERED,
            discovery_source=discovery_source,
            token_type=token_type,
            discovery_time=datetime.now()
        )
        
        # Record creation event
        aggregate.record_event(
            event_type="TOKEN_DISCOVERED",
            payload={
                "chain_id": chain_id,
                "address": address,
                "discovery_source": discovery_source,
                "token_type": token_type
            }
        )
        
        return aggregate
    
    def update_basic_info(
        self,
        name: str,
        symbol: str,
        decimals: int,
        total_supply: Optional[Decimal] = None
    ) -> None:
        """Update basic token information
        
        Args:
            name: Token name
            symbol: Token symbol
            decimals: Token decimals
            total_supply: Total supply
        """
        self.record_event(
            event_type="TOKEN_INFO_UPDATED",
            payload={
                "name": name,
                "symbol": symbol,
                "decimals": decimals,
                "total_supply": str(total_supply) if total_supply else None
            }
        )
    
    def update_contract_info(
        self,
        owner_address: Optional[str] = None,
        creator_address: Optional[str] = None,
        contract_verified: Optional[bool] = None,
        is_mintable: Optional[bool] = None,
        is_burnable: Optional[bool] = None,
        is_pausable: Optional[bool] = None,
        has_blacklist: Optional[bool] = None,
        has_whitelist: Optional[bool] = None,
        has_anti_whale: Optional[bool] = None,
        max_transaction_amount: Optional[Decimal] = None,
        max_wallet_amount: Optional[Decimal] = None,
        buy_tax: Optional[Decimal] = None,
        sell_tax: Optional[Decimal] = None,
        transfer_tax: Optional[Decimal] = None
    ) -> None:
        """Update contract information
        
        Args:
            owner_address: Contract owner address
            creator_address: Contract creator address
            contract_verified: Whether contract is verified
            is_mintable: Whether token is mintable
            is_burnable: Whether token is burnable
            is_pausable: Whether token is pausable
            has_blacklist: Whether token has blacklist
            has_whitelist: Whether token has whitelist
            has_anti_whale: Whether token has anti-whale
            max_transaction_amount: Maximum transaction amount
            max_wallet_amount: Maximum wallet amount
            buy_tax: Buy tax percentage
            sell_tax: Sell tax percentage
            transfer_tax: Transfer tax percentage
        """
        payload = {}
        
        if owner_address is not None:
            payload["owner_address"] = owner_address
        if creator_address is not None:
            payload["creator_address"] = creator_address
        if contract_verified is not None:
            payload["contract_verified"] = contract_verified
        if is_mintable is not None:
            payload["is_mintable"] = is_mintable
        if is_burnable is not None:
            payload["is_burnable"] = is_burnable
        if is_pausable is not None:
            payload["is_pausable"] = is_pausable
        if has_blacklist is not None:
            payload["has_blacklist"] = has_blacklist
        if has_whitelist is not None:
            payload["has_whitelist"] = has_whitelist
        if has_anti_whale is not None:
            payload["has_anti_whale"] = has_anti_whale
        if max_transaction_amount is not None:
            payload["max_transaction_amount"] = str(max_transaction_amount)
        if max_wallet_amount is not None:
            payload["max_wallet_amount"] = str(max_wallet_amount)
        if buy_tax is not None:
            payload["buy_tax"] = str(buy_tax)
        if sell_tax is not None:
            payload["sell_tax"] = str(sell_tax)
        if transfer_tax is not None:
            payload["transfer_tax"] = str(transfer_tax)
        
        if payload:
            self.record_event(
                event_type="CONTRACT_INFO_UPDATED",
                payload=payload
            )
    
    def add_security_check(self, security_check: SecurityCheck) -> None:
        """Add security check result
        
        Args:
            security_check: Security check result
        """
        self.record_event(
            event_type="SECURITY_CHECK_ADDED",
            payload=security_check.dict()
        )
    
    def add_liquidity_pool(self, liquidity_pool: LiquidityPool) -> None:
        """Add liquidity pool
        
        Args:
            liquidity_pool: Liquidity pool information
        """
        payload = liquidity_pool.dict()
        # Convert Decimal to string for serialization
        payload["total_liquidity_usd"] = str(liquidity_pool.total_liquidity_usd)
        payload["token_reserves"] = str(liquidity_pool.token_reserves)
        payload["pair_reserves"] = str(liquidity_pool.pair_reserves)
        payload["last_updated"] = liquidity_pool.last_updated.isoformat()
        
        self.record_event(
            event_type="LIQUIDITY_POOL_ADDED",
            payload=payload
        )
    
    def update_price(
        self,
        price_usd: Decimal,
        source: str,
        volume_24h: Optional[Decimal] = None
    ) -> None:
        """Update token price
        
        Args:
            price_usd: Price in USD
            source: Price source
            volume_24h: 24-hour volume
        """
        self.record_event(
            event_type="PRICE_UPDATED",
            payload={
                "price_usd": str(price_usd),
                "source": source,
                "volume_24h": str(volume_24h) if volume_24h else None,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def start_validation(self) -> None:
        """Start token validation process"""
        if self.state.status != TokenStatus.DISCOVERED:
            raise ValueError(f"Cannot start validation from status {self.state.status}")
        
        self.record_event(
            event_type="VALIDATION_STARTED",
            payload={
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def mark_as_validated(self) -> None:
        """Mark token as validated"""
        if self.state.status not in [TokenStatus.DISCOVERED, TokenStatus.VALIDATING]:
            raise ValueError(f"Cannot validate from status {self.state.status}")
        
        self.record_event(
            event_type="TOKEN_VALIDATED",
            payload={
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def mark_as_rejected(self, reason: str) -> None:
        """Mark token as rejected
        
        Args:
            reason: Rejection reason
        """
        self.record_event(
            event_type="TOKEN_REJECTED",
            payload={
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def mark_as_blacklisted(self, reason: str) -> None:
        """Mark token as blacklisted
        
        Args:
            reason: Blacklist reason
        """
        self.record_event(
            event_type="TOKEN_BLACKLISTED",
            payload={
                "reason": reason,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def mark_as_honeypot(self, details: Dict[str, Any]) -> None:
        """Mark token as honeypot
        
        Args:
            details: Honeypot details
        """
        self.record_event(
            event_type="HONEYPOT_DETECTED",
            payload={
                "details": details,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def add_tags(self, tags: List[str]) -> None:
        """Add tags to token
        
        Args:
            tags: Tags to add
        """
        self.record_event(
            event_type="TAGS_ADDED",
            payload={
                "tags": tags
            }
        )
    
    # Event handlers
    
    def apply_token_discovered(self, event: Event) -> None:
        """Apply TOKEN_DISCOVERED event"""
        # Most state is already initialized in constructor
        pass
    
    def apply_token_info_updated(self, event: Event) -> None:
        """Apply TOKEN_INFO_UPDATED event"""
        payload = event.payload
        self.state.name = payload.get("name")
        self.state.symbol = payload.get("symbol")
        self.state.decimals = payload.get("decimals")
        
        total_supply = payload.get("total_supply")
        if total_supply:
            self.state.total_supply = Decimal(total_supply)
    
    def apply_contract_info_updated(self, event: Event) -> None:
        """Apply CONTRACT_INFO_UPDATED event"""
        payload = event.payload
        
        if "owner_address" in payload:
            self.state.owner_address = payload["owner_address"]
        if "creator_address" in payload:
            self.state.creator_address = payload["creator_address"]
        if "contract_verified" in payload:
            self.state.contract_verified = payload["contract_verified"]
        if "is_mintable" in payload:
            self.state.is_mintable = payload["is_mintable"]
        if "is_burnable" in payload:
            self.state.is_burnable = payload["is_burnable"]
        if "is_pausable" in payload:
            self.state.is_pausable = payload["is_pausable"]
        if "has_blacklist" in payload:
            self.state.has_blacklist = payload["has_blacklist"]
        if "has_whitelist" in payload:
            self.state.has_whitelist = payload["has_whitelist"]
        if "has_anti_whale" in payload:
            self.state.has_anti_whale = payload["has_anti_whale"]
        
        if "max_transaction_amount" in payload:
            self.state.max_transaction_amount = Decimal(payload["max_transaction_amount"])
        if "max_wallet_amount" in payload:
            self.state.max_wallet_amount = Decimal(payload["max_wallet_amount"])
        if "buy_tax" in payload:
            self.state.buy_tax = Decimal(payload["buy_tax"])
        if "sell_tax" in payload:
            self.state.sell_tax = Decimal(payload["sell_tax"])
        if "transfer_tax" in payload:
            self.state.transfer_tax = Decimal(payload["transfer_tax"])
    
    def apply_security_check_added(self, event: Event) -> None:
        """Apply SECURITY_CHECK_ADDED event"""
        security_check = SecurityCheck(**event.payload)
        self.state.security_checks.append(security_check)
    
    def apply_liquidity_pool_added(self, event: Event) -> None:
        """Apply LIQUIDITY_POOL_ADDED event"""
        payload = event.payload.copy()
        
        # Convert string to Decimal
        payload["total_liquidity_usd"] = Decimal(payload["total_liquidity_usd"])
        payload["token_reserves"] = Decimal(payload["token_reserves"])
        payload["pair_reserves"] = Decimal(payload["pair_reserves"])
        
        # Convert string to datetime
        payload["last_updated"] = datetime.fromisoformat(payload["last_updated"])
        
        liquidity_pool = LiquidityPool(**payload)
        
        # Check if pool already exists
        for i, pool in enumerate(self.state.liquidity_pools):
            if pool.address == liquidity_pool.address:
                # Update existing pool
                self.state.liquidity_pools[i] = liquidity_pool
                return
        
        # Add new pool
        self.state.liquidity_pools.append(liquidity_pool)
    
    def apply_price_updated(self, event: Event) -> None:
        """Apply PRICE_UPDATED event"""
        payload = event.payload
        
        price_point = PricePoint(
            timestamp=datetime.fromisoformat(payload["timestamp"]),
            price_usd=Decimal(payload["price_usd"]),
            source=payload["source"],
            volume_24h=Decimal(payload["volume_24h"]) if payload.get("volume_24h") else None
        )
        
        # Update current price
        self.state.current_price_usd = price_point.price_usd
        
        # Add to price history
        self.state.price_history.append(price_point)
        
        # Keep only last 100 price points
        if len(self.state.price_history) > 100:
            self.state.price_history = self.state.price_history[-100:]
    
    def apply_validation_started(self, event: Event) -> None:
        """Apply VALIDATION_STARTED event"""
        self.state.status = TokenStatus.VALIDATING
    
    def apply_token_validated(self, event: Event) -> None:
        """Apply TOKEN_VALIDATED event"""
        self.state.status = TokenStatus.VALIDATED
        self.state.validation_time = datetime.fromisoformat(event.payload["timestamp"])
    
    def apply_token_rejected(self, event: Event) -> None:
        """Apply TOKEN_REJECTED event"""
        self.state.status = TokenStatus.REJECTED
        self.state.metadata["rejection_reason"] = event.payload["reason"]
        self.state.metadata["rejection_time"] = event.payload["timestamp"]
    
    def apply_token_blacklisted(self, event: Event) -> None:
        """Apply TOKEN_BLACKLISTED event"""
        self.state.status = TokenStatus.BLACKLISTED
        self.state.metadata["blacklist_reason"] = event.payload["reason"]
        self.state.metadata["blacklist_time"] = event.payload["timestamp"]
    
    def apply_honeypot_detected(self, event: Event) -> None:
        """Apply HONEYPOT_DETECTED event"""
        self.state.is_honeypot = True
        self.state.metadata["honeypot_details"] = event.payload["details"]
        self.state.metadata["honeypot_detection_time"] = event.payload["timestamp"]
    
    def apply_tags_added(self, event: Event) -> None:
        """Apply TAGS_ADDED event"""
        new_tags = event.payload["tags"]
        
        # Add new tags without duplicates
        for tag in new_tags:
            if tag not in self.state.tags:
                self.state.tags.append(tag)

# -----------------------------------------------------------------------------
# Event Handlers
# -----------------------------------------------------------------------------

@handles_event("TOKEN_DISCOVERED")
class TokenDiscoveredHandler(EventHandler):
    """Handler for TOKEN_DISCOVERED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event"""
        from ...services.token_service import token_service
        
        # Extract data from event
        chain_id = event.payload["chain_id"]
        token_address = event.payload["address"]
        
        # Queue token for basic info retrieval
        await token_service.queue_token_info_retrieval(chain_id, token_address)
        
        # Log discovery
        print(f"Token discovered: {token_address} on chain {chain_id}")

@handles_event("VALIDATION_STARTED")
class ValidationStartedHandler(EventHandler):
    """Handler for VALIDATION_STARTED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event"""
        from ...services.validation_service import validation_service
        
        # Get token from repository
        from ..aggregates import aggregate_factory
        repository = aggregate_factory(TokenAggregate)
        
        token = await repository.get_by_id(event.aggregate_id)
        if not token:
            print(f"Token not found: {event.aggregate_id}")
            return
        
        # Queue security checks
        await validation_service.queue_security_checks(
            token.state.chain_id,
            token.state.address
        )
        
        # Queue liquidity checks
        await validation_service.queue_liquidity_checks(
            token.state.chain_id,
            token.state.address
        )
        
        print(f"Validation started for token: {token.state.address}")

@handles_event("TOKEN_VALIDATED")
class TokenValidatedHandler(EventHandler):
    """Handler for TOKEN_VALIDATED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event"""
        from ...services.notification_service import notification_service
        
        # Get token from repository
        from ..aggregates import aggregate_factory
        repository = aggregate_factory(TokenAggregate)
        
        token = await repository.get_by_id(event.aggregate_id)
        if not token:
            print(f"Token not found: {event.aggregate_id}")
            return
        
        # Send notification
        await notification_service.send_token_validated_notification(
            token.state.chain_id,
            token.state.address,
            token.state.symbol or "Unknown",
            token.state.current_price_usd
        )
        
        print(f"Token validated: {token.state.address} ({token.state.symbol})")

@handles_event("HONEYPOT_DETECTED")
class HoneypotDetectedHandler(EventHandler):
    """Handler for HONEYPOT_DETECTED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event"""
        from ...services.notification_service import notification_service
        
        # Get token from repository
        from ..aggregates import aggregate_factory
        repository = aggregate_factory(TokenAggregate)
        
        token = await repository.get_by_id(event.aggregate_id)
        if not token:
            print(f"Token not found: {event.aggregate_id}")
            return
        
        # Send notification
        await notification_service.send_honeypot_notification(
            token.state.chain_id,
            token.state.address,
            token.state.symbol or "Unknown",
            event.payload["details"]
        )
        
        print(f"Honeypot detected: {token.state.address} ({token.state.symbol})") 