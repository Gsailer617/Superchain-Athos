"""
Arbitrage Domain Model

This module defines the domain model for arbitrage opportunities:
- Arbitrage opportunity aggregate for managing arbitrage state
- Events for arbitrage lifecycle
- Event handlers for arbitrage-related events
"""

import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

from .base import Event, EventHandler
from .aggregates import Aggregate, AggregateState
from .token_discovery import TokenAggregate, LiquiditySource
from ..decorators import handles_event
from ...core.dependency_injector import container

# -----------------------------------------------------------------------------
# Domain Models
# -----------------------------------------------------------------------------

class ArbitrageStatus(str, Enum):
    """Arbitrage opportunity status"""
    IDENTIFIED = "identified"
    ANALYZING = "analyzing"
    READY = "ready"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

class ArbitrageType(str, Enum):
    """Arbitrage types"""
    SIMPLE_DEX = "simple_dex"  # Between two DEXes
    TRIANGULAR = "triangular"  # Three-token cycle
    CROSS_CHAIN = "cross_chain"  # Between chains
    FLASH_LOAN = "flash_loan"  # Using flash loans
    SANDWICH = "sandwich"  # Sandwich attack
    CUSTOM = "custom"  # Custom strategy

class ArbitrageRisk(str, Enum):
    """Risk levels for arbitrage opportunities"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class ExecutionPriority(str, Enum):
    """Execution priority levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    VERY_HIGH = "very_high"

class RouteStep(BaseModel):
    """Step in an arbitrage route"""
    step_index: int
    dex_name: str
    source_token: str
    target_token: str
    source_amount: Decimal
    expected_target_amount: Decimal
    min_target_amount: Decimal
    pool_address: str
    chain_id: int
    execution_data: Dict[str, Any] = Field(default_factory=dict)

class GasEstimate(BaseModel):
    """Gas estimate for transaction"""
    gas_limit: int
    gas_price: Decimal  # In wei
    total_cost_eth: Decimal
    total_cost_usd: Decimal
    estimated_at: datetime = Field(default_factory=datetime.now)

class SimulationResult(BaseModel):
    """Result of transaction simulation"""
    success: bool
    profit_amount: Decimal
    profit_token: str
    profit_usd: Decimal
    gas_used: int
    error_message: Optional[str] = None
    execution_trace: Optional[Dict[str, Any]] = None
    simulated_at: datetime = Field(default_factory=datetime.now)

class FlashLoanDetails(BaseModel):
    """Flash loan details"""
    provider: str  # Aave, dYdX, etc.
    token_address: str
    amount: Decimal
    fee_amount: Decimal
    fee_percentage: Decimal

# -----------------------------------------------------------------------------
# Aggregate State
# -----------------------------------------------------------------------------

class ArbitrageState(AggregateState):
    """State for arbitrage opportunity aggregate"""
    opportunity_id: str
    arbitrage_type: ArbitrageType
    status: ArbitrageStatus = ArbitrageStatus.IDENTIFIED
    chain_ids: List[int]  # Can span multiple chains
    tokens_involved: List[str]  # Token addresses
    entry_token: str  # Token to start with
    entry_amount: Decimal  # Amount to start with
    expected_profit_token: str  # Token to receive profit in
    expected_profit_amount: Decimal  # Expected profit amount
    expected_profit_usd: Decimal  # Expected profit in USD
    expected_profit_percentage: Decimal  # Expected profit percentage
    risk_level: ArbitrageRisk = ArbitrageRisk.MEDIUM
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    route_steps: List[RouteStep] = Field(default_factory=list)
    gas_estimate: Optional[GasEstimate] = None
    simulation_result: Optional[SimulationResult] = None
    flash_loan_details: Optional[FlashLoanDetails] = None
    execution_tx_hash: Optional[str] = None
    execution_timestamp: Optional[datetime] = None
    execution_result: Optional[Dict[str, Any]] = None
    actual_profit_amount: Optional[Decimal] = None
    actual_profit_usd: Optional[Decimal] = None
    expiration_time: datetime  # When opportunity expires
    discovery_source: str = "unknown"  # How this opportunity was found
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# -----------------------------------------------------------------------------
# Arbitrage Aggregate
# -----------------------------------------------------------------------------

class ArbitrageAggregate(Aggregate[ArbitrageState]):
    """Aggregate for managing arbitrage opportunities"""
    
    @classmethod
    async def create_opportunity(
        cls,
        arbitrage_type: ArbitrageType,
        chain_ids: List[int],
        tokens_involved: List[str],
        entry_token: str,
        entry_amount: Decimal,
        expected_profit_token: str,
        expected_profit_amount: Decimal,
        expected_profit_usd: Decimal,
        expected_profit_percentage: Decimal,
        discovery_source: str,
        expiration_time: Optional[datetime] = None,
        risk_level: ArbitrageRisk = ArbitrageRisk.MEDIUM,
        priority: ExecutionPriority = ExecutionPriority.NORMAL
    ) -> "ArbitrageAggregate":
        """Create a new arbitrage opportunity
        
        Args:
            arbitrage_type: Type of arbitrage
            chain_ids: Chain IDs involved
            tokens_involved: Token addresses involved
            entry_token: Token to start with
            entry_amount: Amount to start with
            expected_profit_token: Token to receive profit in
            expected_profit_amount: Expected profit amount
            expected_profit_usd: Expected profit in USD
            expected_profit_percentage: Expected profit percentage
            discovery_source: How this opportunity was found
            expiration_time: When opportunity expires
            risk_level: Risk level
            priority: Execution priority
            
        Returns:
            Arbitrage opportunity aggregate
        """
        # Generate opportunity ID
        opportunity_id = str(uuid4())
        
        # Set default expiration time if not provided
        if not expiration_time:
            expiration_time = datetime.now() + timedelta(minutes=5)
        
        # Create aggregate
        aggregate = await cls.create(
            opportunity_id=opportunity_id,
            arbitrage_type=arbitrage_type,
            status=ArbitrageStatus.IDENTIFIED,
            chain_ids=chain_ids,
            tokens_involved=tokens_involved,
            entry_token=entry_token,
            entry_amount=entry_amount,
            expected_profit_token=expected_profit_token,
            expected_profit_amount=expected_profit_amount,
            expected_profit_usd=expected_profit_usd,
            expected_profit_percentage=expected_profit_percentage,
            risk_level=risk_level,
            priority=priority,
            discovery_source=discovery_source,
            expiration_time=expiration_time
        )
        
        # Record creation event
        aggregate.record_event(
            event_type="ARBITRAGE_OPPORTUNITY_IDENTIFIED",
            payload={
                "opportunity_id": opportunity_id,
                "arbitrage_type": arbitrage_type,
                "chain_ids": chain_ids,
                "tokens_involved": tokens_involved,
                "entry_token": entry_token,
                "entry_amount": str(entry_amount),
                "expected_profit_token": expected_profit_token,
                "expected_profit_amount": str(expected_profit_amount),
                "expected_profit_usd": str(expected_profit_usd),
                "expected_profit_percentage": str(expected_profit_percentage),
                "risk_level": risk_level,
                "priority": priority,
                "discovery_source": discovery_source,
                "expiration_time": expiration_time.isoformat()
            }
        )
        
        return aggregate
    
    def add_route_step(self, route_step: RouteStep) -> None:
        """Add a step to the arbitrage route
        
        Args:
            route_step: Route step to add
        """
        payload = route_step.dict()
        
        # Convert Decimal to string for serialization
        payload["source_amount"] = str(route_step.source_amount)
        payload["expected_target_amount"] = str(route_step.expected_target_amount)
        payload["min_target_amount"] = str(route_step.min_target_amount)
        
        self.record_event(
            event_type="ROUTE_STEP_ADDED",
            payload=payload
        )
    
    def set_flash_loan_details(self, flash_loan_details: FlashLoanDetails) -> None:
        """Set flash loan details
        
        Args:
            flash_loan_details: Flash loan details
        """
        payload = flash_loan_details.dict()
        
        # Convert Decimal to string for serialization
        payload["amount"] = str(flash_loan_details.amount)
        payload["fee_amount"] = str(flash_loan_details.fee_amount)
        payload["fee_percentage"] = str(flash_loan_details.fee_percentage)
        
        self.record_event(
            event_type="FLASH_LOAN_DETAILS_SET",
            payload=payload
        )
    
    def set_gas_estimate(self, gas_estimate: GasEstimate) -> None:
        """Set gas estimate
        
        Args:
            gas_estimate: Gas estimate
        """
        payload = gas_estimate.dict()
        
        # Convert Decimal to string for serialization
        payload["gas_price"] = str(gas_estimate.gas_price)
        payload["total_cost_eth"] = str(gas_estimate.total_cost_eth)
        payload["total_cost_usd"] = str(gas_estimate.total_cost_usd)
        payload["estimated_at"] = gas_estimate.estimated_at.isoformat()
        
        self.record_event(
            event_type="GAS_ESTIMATE_SET",
            payload=payload
        )
    
    def set_simulation_result(self, simulation_result: SimulationResult) -> None:
        """Set simulation result
        
        Args:
            simulation_result: Simulation result
        """
        payload = simulation_result.dict()
        
        # Convert Decimal to string for serialization
        payload["profit_amount"] = str(simulation_result.profit_amount)
        payload["profit_usd"] = str(simulation_result.profit_usd)
        payload["simulated_at"] = simulation_result.simulated_at.isoformat()
        
        self.record_event(
            event_type="SIMULATION_COMPLETED",
            payload=payload
        )
    
    def mark_as_ready(self) -> None:
        """Mark arbitrage opportunity as ready for execution"""
        if self.state.status not in [ArbitrageStatus.IDENTIFIED, ArbitrageStatus.ANALYZING]:
            raise ValueError(f"Cannot mark as ready from status {self.state.status}")
        
        self.record_event(
            event_type="ARBITRAGE_READY",
            payload={
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def start_execution(self) -> None:
        """Start arbitrage execution"""
        if self.state.status != ArbitrageStatus.READY:
            raise ValueError(f"Cannot start execution from status {self.state.status}")
        
        self.record_event(
            event_type="EXECUTION_STARTED",
            payload={
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def complete_execution(
        self,
        tx_hash: str,
        actual_profit_amount: Decimal,
        actual_profit_usd: Decimal,
        execution_result: Dict[str, Any]
    ) -> None:
        """Complete arbitrage execution
        
        Args:
            tx_hash: Transaction hash
            actual_profit_amount: Actual profit amount
            actual_profit_usd: Actual profit in USD
            execution_result: Execution result details
        """
        if self.state.status != ArbitrageStatus.EXECUTING:
            raise ValueError(f"Cannot complete execution from status {self.state.status}")
        
        self.record_event(
            event_type="EXECUTION_COMPLETED",
            payload={
                "tx_hash": tx_hash,
                "actual_profit_amount": str(actual_profit_amount),
                "actual_profit_usd": str(actual_profit_usd),
                "execution_result": execution_result,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def fail_execution(self, reason: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Fail arbitrage execution
        
        Args:
            reason: Failure reason
            details: Failure details
        """
        self.record_event(
            event_type="EXECUTION_FAILED",
            payload={
                "reason": reason,
                "details": details or {},
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def expire_opportunity(self) -> None:
        """Mark arbitrage opportunity as expired"""
        if self.state.status in [ArbitrageStatus.COMPLETED, ArbitrageStatus.FAILED, ArbitrageStatus.EXPIRED]:
            return
        
        self.record_event(
            event_type="OPPORTUNITY_EXPIRED",
            payload={
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def update_priority(self, priority: ExecutionPriority) -> None:
        """Update execution priority
        
        Args:
            priority: New execution priority
        """
        self.record_event(
            event_type="PRIORITY_UPDATED",
            payload={
                "priority": priority
            }
        )
    
    def add_tags(self, tags: List[str]) -> None:
        """Add tags to arbitrage opportunity
        
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
    
    def apply_arbitrage_opportunity_identified(self, event: Event) -> None:
        """Apply ARBITRAGE_OPPORTUNITY_IDENTIFIED event"""
        # Most state is already initialized in constructor
        pass
    
    def apply_route_step_added(self, event: Event) -> None:
        """Apply ROUTE_STEP_ADDED event"""
        payload = event.payload.copy()
        
        # Convert string to Decimal
        payload["source_amount"] = Decimal(payload["source_amount"])
        payload["expected_target_amount"] = Decimal(payload["expected_target_amount"])
        payload["min_target_amount"] = Decimal(payload["min_target_amount"])
        
        route_step = RouteStep(**payload)
        
        # Check if step already exists
        for i, step in enumerate(self.state.route_steps):
            if step.step_index == route_step.step_index:
                # Update existing step
                self.state.route_steps[i] = route_step
                return
        
        # Add new step
        self.state.route_steps.append(route_step)
        
        # Sort steps by index
        self.state.route_steps.sort(key=lambda s: s.step_index)
    
    def apply_flash_loan_details_set(self, event: Event) -> None:
        """Apply FLASH_LOAN_DETAILS_SET event"""
        payload = event.payload.copy()
        
        # Convert string to Decimal
        payload["amount"] = Decimal(payload["amount"])
        payload["fee_amount"] = Decimal(payload["fee_amount"])
        payload["fee_percentage"] = Decimal(payload["fee_percentage"])
        
        self.state.flash_loan_details = FlashLoanDetails(**payload)
    
    def apply_gas_estimate_set(self, event: Event) -> None:
        """Apply GAS_ESTIMATE_SET event"""
        payload = event.payload.copy()
        
        # Convert string to Decimal
        payload["gas_price"] = Decimal(payload["gas_price"])
        payload["total_cost_eth"] = Decimal(payload["total_cost_eth"])
        payload["total_cost_usd"] = Decimal(payload["total_cost_usd"])
        
        # Convert string to datetime
        payload["estimated_at"] = datetime.fromisoformat(payload["estimated_at"])
        
        self.state.gas_estimate = GasEstimate(**payload)
    
    def apply_simulation_completed(self, event: Event) -> None:
        """Apply SIMULATION_COMPLETED event"""
        payload = event.payload.copy()
        
        # Convert string to Decimal
        payload["profit_amount"] = Decimal(payload["profit_amount"])
        payload["profit_usd"] = Decimal(payload["profit_usd"])
        
        # Convert string to datetime
        payload["simulated_at"] = datetime.fromisoformat(payload["simulated_at"])
        
        self.state.simulation_result = SimulationResult(**payload)
        
        # Update status to analyzing
        if self.state.status == ArbitrageStatus.IDENTIFIED:
            self.state.status = ArbitrageStatus.ANALYZING
    
    def apply_arbitrage_ready(self, event: Event) -> None:
        """Apply ARBITRAGE_READY event"""
        self.state.status = ArbitrageStatus.READY
    
    def apply_execution_started(self, event: Event) -> None:
        """Apply EXECUTION_STARTED event"""
        self.state.status = ArbitrageStatus.EXECUTING
        self.state.execution_timestamp = datetime.fromisoformat(event.payload["timestamp"])
    
    def apply_execution_completed(self, event: Event) -> None:
        """Apply EXECUTION_COMPLETED event"""
        self.state.status = ArbitrageStatus.COMPLETED
        self.state.execution_tx_hash = event.payload["tx_hash"]
        self.state.actual_profit_amount = Decimal(event.payload["actual_profit_amount"])
        self.state.actual_profit_usd = Decimal(event.payload["actual_profit_usd"])
        self.state.execution_result = event.payload["execution_result"]
    
    def apply_execution_failed(self, event: Event) -> None:
        """Apply EXECUTION_FAILED event"""
        self.state.status = ArbitrageStatus.FAILED
        self.state.metadata["failure_reason"] = event.payload["reason"]
        self.state.metadata["failure_details"] = event.payload["details"]
        self.state.metadata["failure_time"] = event.payload["timestamp"]
    
    def apply_opportunity_expired(self, event: Event) -> None:
        """Apply OPPORTUNITY_EXPIRED event"""
        self.state.status = ArbitrageStatus.EXPIRED
    
    def apply_priority_updated(self, event: Event) -> None:
        """Apply PRIORITY_UPDATED event"""
        self.state.priority = event.payload["priority"]
    
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

@handles_event("ARBITRAGE_OPPORTUNITY_IDENTIFIED")
class ArbitrageIdentifiedHandler(EventHandler):
    """Handler for ARBITRAGE_OPPORTUNITY_IDENTIFIED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event"""
        from ...services.arbitrage_service import arbitrage_service
        
        # Extract data from event
        opportunity_id = event.payload["opportunity_id"]
        
        # Queue opportunity for analysis
        await arbitrage_service.queue_opportunity_analysis(opportunity_id)
        
        # Log discovery
        print(f"Arbitrage opportunity identified: {opportunity_id}")

@handles_event("SIMULATION_COMPLETED")
class SimulationCompletedHandler(EventHandler):
    """Handler for SIMULATION_COMPLETED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event"""
        from ...services.arbitrage_service import arbitrage_service
        
        # Get arbitrage from repository
        from ..aggregates import aggregate_factory
        repository = aggregate_factory(ArbitrageAggregate)
        
        arbitrage = await repository.get_by_id(event.aggregate_id)
        if not arbitrage:
            print(f"Arbitrage not found: {event.aggregate_id}")
            return
        
        # Check if simulation was successful
        simulation_result = arbitrage.state.simulation_result
        if not simulation_result or not simulation_result.success:
            # Simulation failed, mark opportunity as failed
            arbitrage.fail_execution(
                reason="Simulation failed",
                details={
                    "error": simulation_result.error_message if simulation_result else "Unknown error"
                }
            )
            await repository.save(arbitrage)
            return
        
        # Check if opportunity is still profitable after gas costs
        gas_estimate = arbitrage.state.gas_estimate
        if gas_estimate and simulation_result.profit_usd <= gas_estimate.total_cost_usd:
            # Not profitable after gas costs
            arbitrage.fail_execution(
                reason="Not profitable after gas costs",
                details={
                    "profit_usd": str(simulation_result.profit_usd),
                    "gas_cost_usd": str(gas_estimate.total_cost_usd)
                }
            )
            await repository.save(arbitrage)
            return
        
        # Mark as ready for execution
        arbitrage.mark_as_ready()
        await repository.save(arbitrage)
        
        # Queue for execution if high priority
        if arbitrage.state.priority in [ExecutionPriority.HIGH, ExecutionPriority.VERY_HIGH]:
            await arbitrage_service.queue_opportunity_execution(event.aggregate_id)

@handles_event("EXECUTION_COMPLETED")
class ExecutionCompletedHandler(EventHandler):
    """Handler for EXECUTION_COMPLETED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event"""
        from ...services.notification_service import notification_service
        
        # Get arbitrage from repository
        from ..aggregates import aggregate_factory
        repository = aggregate_factory(ArbitrageAggregate)
        
        arbitrage = await repository.get_by_id(event.aggregate_id)
        if not arbitrage:
            print(f"Arbitrage not found: {event.aggregate_id}")
            return
        
        # Send notification
        await notification_service.send_arbitrage_completed_notification(
            arbitrage.state.opportunity_id,
            arbitrage.state.arbitrage_type,
            arbitrage.state.actual_profit_usd,
            arbitrage.state.execution_tx_hash
        )
        
        print(f"Arbitrage completed: {arbitrage.state.opportunity_id} with profit ${arbitrage.state.actual_profit_usd}")

@handles_event("EXECUTION_FAILED")
class ExecutionFailedHandler(EventHandler):
    """Handler for EXECUTION_FAILED event"""
    
    async def handle(self, event: Event) -> None:
        """Handle event"""
        from ...services.notification_service import notification_service
        
        # Get arbitrage from repository
        from ..aggregates import aggregate_factory
        repository = aggregate_factory(ArbitrageAggregate)
        
        arbitrage = await repository.get_by_id(event.aggregate_id)
        if not arbitrage:
            print(f"Arbitrage not found: {event.aggregate_id}")
            return
        
        # Send notification
        await notification_service.send_arbitrage_failed_notification(
            arbitrage.state.opportunity_id,
            arbitrage.state.arbitrage_type,
            event.payload["reason"]
        )
        
        print(f"Arbitrage failed: {arbitrage.state.opportunity_id} - {event.payload['reason']}")

# Create a background task to expire old opportunities
async def expire_old_opportunities():
    """Background task to expire old opportunities"""
    while True:
        try:
            # Get all active opportunities
            from ..aggregates import aggregate_factory
            repository = aggregate_factory(ArbitrageAggregate)
            
            # This would normally use a query to find expired opportunities
            # For now, we'll just log a message
            print("Checking for expired arbitrage opportunities...")
            
            # Sleep for 1 minute
            await asyncio.sleep(60)
            
        except Exception as e:
            print(f"Error in expire_old_opportunities: {str(e)}")
            await asyncio.sleep(10)  # Sleep on error

# Start background task
def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(expire_old_opportunities()) 