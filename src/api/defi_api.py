"""
DeFi API

This module provides a FastAPI interface for DeFi services:
- Arbitrage opportunity discovery and execution
- Yield farming position management
- Portfolio optimization
- Performance tracking
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union

from fastapi import FastAPI, HTTPException, Query, Path, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import uvicorn

from ..services.defi_service import defi_service
from ..services.arbitrage_service import arbitrage_service
from ..services.yield_farming_service import yield_farming_service
from ..core.dependency_injector import container

# Create FastAPI app
app = FastAPI(
    title="DeFi API",
    description="API for DeFi operations including arbitrage and yield farming",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Pydantic Models
# -----------------------------------------------------------------------------

class DecimalStr(str):
    """String representation of a decimal value"""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if isinstance(v, str):
            return Decimal(v)
        return v

class ArbitrageOpportunity(BaseModel):
    """Arbitrage opportunity model"""
    id: str
    chain_id: int
    type: str
    expected_profit_usd: DecimalStr
    risk_level: str
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v)
        }

class YieldOpportunity(BaseModel):
    """Yield opportunity model"""
    id: str
    chain_id: int
    protocol: str
    pool_name: str
    apy: DecimalStr
    tvl_usd: DecimalStr
    risk_level: str
    tokens: List[str]
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v)
        }

class OpportunitiesResponse(BaseModel):
    """Response model for opportunities"""
    arbitrage: List[ArbitrageOpportunity]
    yield_farming: List[YieldOpportunity]
    risk_profile: str
    max_risk_level: str
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v)
        }

class PortfolioAllocationResponse(BaseModel):
    """Response model for portfolio allocation"""
    wallet_address: str
    total_value_usd: DecimalStr
    arbitrage_allocation_usd: DecimalStr
    yield_allocation_usd: DecimalStr
    reserve_allocation_usd: DecimalStr
    arbitrage_opportunities: List[ArbitrageOpportunity]
    yield_opportunities: List[YieldOpportunity]
    risk_score: int
    expected_monthly_return: DecimalStr
    timestamp: datetime
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }

class PerformanceMetricsResponse(BaseModel):
    """Response model for performance metrics"""
    wallet_address: str
    start_date: datetime
    end_date: datetime
    starting_balance_usd: DecimalStr
    ending_balance_usd: DecimalStr
    total_profit_usd: DecimalStr
    total_profit_percentage: DecimalStr
    annualized_return: DecimalStr
    arbitrage_profit_usd: DecimalStr
    yield_farming_profit_usd: DecimalStr
    successful_arbitrages: int
    failed_arbitrages: int
    active_yield_positions: int
    closed_yield_positions: int
    best_performing_strategy: str
    worst_performing_strategy: str
    risk_adjusted_return: DecimalStr
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }

class DepositAmount(BaseModel):
    """Deposit amount model"""
    token_address: str
    amount: DecimalStr
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v)
        }

class CreateYieldPositionRequest(BaseModel):
    """Request model for creating a yield position"""
    opportunity_id: str
    wallet_address: str
    deposit_amounts: List[DepositAmount]
    gas_price_gwei: Optional[DecimalStr] = None
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v)
        }

class YieldPositionResponse(BaseModel):
    """Response model for yield position"""
    id: str
    opportunity_id: str
    wallet_address: str
    deposit_amounts: Dict[str, DecimalStr]
    deposit_usd_value: DecimalStr
    entry_timestamp: datetime
    last_harvest_timestamp: Optional[datetime] = None
    harvested_rewards: Dict[str, DecimalStr]
    current_value_usd: DecimalStr
    profit_usd: DecimalStr
    apy_realized: DecimalStr
    status: str
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v),
            datetime: lambda v: v.isoformat()
        }

class OptimizePortfolioRequest(BaseModel):
    """Request model for portfolio optimization"""
    wallet_address: str
    total_value_usd: DecimalStr
    risk_profile: str = "moderate"
    chain_ids: Optional[List[int]] = None
    rebalance: bool = False
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v)
        }

class OptimizePortfolioResponse(BaseModel):
    """Response model for portfolio optimization"""
    wallet_address: str
    risk_profile: str
    total_value_usd: DecimalStr
    target_yield_allocation: DecimalStr
    current_yield_allocation: DecimalStr
    rebalance_needed: bool
    rebalance_performed: bool
    expected_monthly_return: DecimalStr
    arbitrage_opportunities: int
    yield_opportunities: int
    
    class Config:
        json_encoders = {
            Decimal: lambda v: str(v)
        }

# -----------------------------------------------------------------------------
# API Routes
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "DeFi API",
        "version": "1.0.0",
        "description": "API for DeFi operations including arbitrage and yield farming"
    }

@app.get("/opportunities", response_model=OpportunitiesResponse)
async def get_opportunities(
    wallet_address: str = Query(..., description="Wallet address"),
    chain_ids: str = Query("1,56,137", description="Comma-separated list of chain IDs"),
    risk_profile: str = Query("moderate", description="Risk profile (conservative, moderate, aggressive)"),
    max_opportunities: int = Query(10, description="Maximum number of opportunities to return"),
    include_arbitrage: bool = Query(True, description="Whether to include arbitrage opportunities"),
    include_yield: bool = Query(True, description="Whether to include yield farming opportunities")
):
    """Get DeFi opportunities for a wallet"""
    try:
        # Parse chain IDs
        chain_id_list = [int(chain_id) for chain_id in chain_ids.split(',')]
        
        # Get opportunities
        opportunities = await defi_service.get_defi_opportunities(
            wallet_address=wallet_address,
            chain_ids=chain_id_list,
            risk_profile=risk_profile,
            max_opportunities=max_opportunities,
            include_arbitrage=include_arbitrage,
            include_yield=include_yield
        )
        
        # Convert to response model
        return OpportunitiesResponse(
            arbitrage=[ArbitrageOpportunity(**opp) for opp in opportunities["arbitrage"]],
            yield_farming=[YieldOpportunity(**opp) for opp in opportunities["yield_farming"]],
            risk_profile=opportunities["risk_profile"],
            max_risk_level=opportunities["max_risk_level"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/allocation", response_model=PortfolioAllocationResponse)
async def get_portfolio_allocation(
    wallet_address: str = Query(..., description="Wallet address"),
    total_value_usd: str = Query(..., description="Total portfolio value in USD"),
    risk_profile: str = Query("moderate", description="Risk profile (conservative, moderate, aggressive)"),
    chain_ids: Optional[str] = Query(None, description="Comma-separated list of chain IDs")
):
    """Get portfolio allocation recommendation"""
    try:
        # Parse chain IDs
        chain_id_list = None
        if chain_ids:
            chain_id_list = [int(chain_id) for chain_id in chain_ids.split(',')]
        
        # Get allocation
        allocation = await defi_service.get_portfolio_allocation(
            wallet_address=wallet_address,
            total_value_usd=Decimal(total_value_usd),
            risk_profile=risk_profile,
            chain_ids=chain_id_list
        )
        
        # Convert to response model
        return PortfolioAllocationResponse(
            wallet_address=allocation.wallet_address,
            total_value_usd=allocation.total_value_usd,
            arbitrage_allocation_usd=allocation.arbitrage_allocation_usd,
            yield_allocation_usd=allocation.yield_allocation_usd,
            reserve_allocation_usd=allocation.reserve_allocation_usd,
            arbitrage_opportunities=[ArbitrageOpportunity(**opp) for opp in allocation.arbitrage_opportunities],
            yield_opportunities=[YieldOpportunity(**opp) for opp in allocation.yield_opportunities],
            risk_score=allocation.risk_score,
            expected_monthly_return=allocation.expected_monthly_return,
            timestamp=allocation.timestamp
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio/metrics", response_model=PerformanceMetricsResponse)
async def get_performance_metrics(
    wallet_address: str = Query(..., description="Wallet address"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)")
):
    """Get performance metrics for a wallet"""
    try:
        # Parse dates
        start_date_obj = None
        if start_date:
            start_date_obj = datetime.fromisoformat(start_date)
        
        end_date_obj = None
        if end_date:
            end_date_obj = datetime.fromisoformat(end_date)
        
        # Get metrics
        metrics = await defi_service.get_performance_metrics(
            wallet_address=wallet_address,
            start_date=start_date_obj,
            end_date=end_date_obj
        )
        
        # Convert to response model
        return PerformanceMetricsResponse(
            wallet_address=metrics.wallet_address,
            start_date=metrics.start_date,
            end_date=metrics.end_date,
            starting_balance_usd=metrics.starting_balance_usd,
            ending_balance_usd=metrics.ending_balance_usd,
            total_profit_usd=metrics.total_profit_usd,
            total_profit_percentage=metrics.total_profit_percentage,
            annualized_return=metrics.annualized_return,
            arbitrage_profit_usd=metrics.arbitrage_profit_usd,
            yield_farming_profit_usd=metrics.yield_farming_profit_usd,
            successful_arbitrages=metrics.successful_arbitrages,
            failed_arbitrages=metrics.failed_arbitrages,
            active_yield_positions=metrics.active_yield_positions,
            closed_yield_positions=metrics.closed_yield_positions,
            best_performing_strategy=metrics.best_performing_strategy,
            worst_performing_strategy=metrics.worst_performing_strategy,
            risk_adjusted_return=metrics.risk_adjusted_return
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/arbitrage/execute")
async def execute_arbitrage(
    opportunity_id: str = Query(..., description="Opportunity ID"),
    wallet_address: str = Query(..., description="Wallet address"),
    gas_price_gwei: Optional[str] = Query(None, description="Gas price in Gwei"),
    background_tasks: BackgroundTasks = None
):
    """Execute an arbitrage opportunity"""
    try:
        # Parse gas price
        gas_price = None
        if gas_price_gwei:
            gas_price = Decimal(gas_price_gwei)
        
        # Execute arbitrage
        if background_tasks:
            # Execute in background
            background_tasks.add_task(
                defi_service.execute_arbitrage_opportunity,
                opportunity_id=opportunity_id,
                wallet_address=wallet_address,
                gas_price_gwei=gas_price
            )
            return {"status": "queued", "opportunity_id": opportunity_id}
        else:
            # Execute synchronously
            success = await defi_service.execute_arbitrage_opportunity(
                opportunity_id=opportunity_id,
                wallet_address=wallet_address,
                gas_price_gwei=gas_price
            )
            
            if success:
                return {"status": "success", "opportunity_id": opportunity_id}
            else:
                raise HTTPException(status_code=500, detail="Failed to execute arbitrage opportunity")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/yield/create", response_model=str)
async def create_yield_position(request: CreateYieldPositionRequest):
    """Create a yield farming position"""
    try:
        # Convert deposit amounts
        deposit_amounts = {}
        for deposit in request.deposit_amounts:
            deposit_amounts[deposit.token_address] = deposit.amount
        
        # Create position
        position_id = await defi_service.create_yield_position(
            opportunity_id=request.opportunity_id,
            wallet_address=request.wallet_address,
            deposit_amounts=deposit_amounts,
            gas_price_gwei=request.gas_price_gwei
        )
        
        if position_id:
            return position_id
        else:
            raise HTTPException(status_code=500, detail="Failed to create yield position")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/yield/position/{position_id}", response_model=YieldPositionResponse)
async def get_yield_position(position_id: str = Path(..., description="Position ID")):
    """Get a yield position by ID"""
    try:
        # Get position
        position = await yield_farming_service.get_position(position_id)
        
        if not position:
            raise HTTPException(status_code=404, detail=f"Position {position_id} not found")
        
        # Convert to response model
        return YieldPositionResponse(
            id=position.id,
            opportunity_id=position.opportunity_id,
            wallet_address=position.wallet_address,
            deposit_amounts=position.deposit_amounts,
            deposit_usd_value=position.deposit_usd_value,
            entry_timestamp=position.entry_timestamp,
            last_harvest_timestamp=position.last_harvest_timestamp,
            harvested_rewards=position.harvested_rewards,
            current_value_usd=position.current_value_usd,
            profit_usd=position.profit_usd,
            apy_realized=position.apy_realized,
            status=position.status
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/yield/positions", response_model=List[YieldPositionResponse])
async def get_yield_positions(
    wallet_address: str = Query(..., description="Wallet address"),
    chain_id: Optional[int] = Query(None, description="Chain ID"),
    status: Optional[str] = Query(None, description="Position status")
):
    """Get all yield positions for a wallet"""
    try:
        # Get positions
        positions = await yield_farming_service.get_positions_by_wallet(
            wallet_address=wallet_address,
            chain_id=chain_id,
            status=status
        )
        
        # Convert to response model
        return [
            YieldPositionResponse(
                id=position.id,
                opportunity_id=position.opportunity_id,
                wallet_address=position.wallet_address,
                deposit_amounts=position.deposit_amounts,
                deposit_usd_value=position.deposit_usd_value,
                entry_timestamp=position.entry_timestamp,
                last_harvest_timestamp=position.last_harvest_timestamp,
                harvested_rewards=position.harvested_rewards,
                current_value_usd=position.current_value_usd,
                profit_usd=position.profit_usd,
                apy_realized=position.apy_realized,
                status=position.status
            )
            for position in positions
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/yield/harvest/{position_id}")
async def harvest_yield_position(
    position_id: str = Path(..., description="Position ID"),
    auto_compound: bool = Query(True, description="Whether to auto-compound rewards"),
    gas_price_gwei: Optional[str] = Query(None, description="Gas price in Gwei"),
    background_tasks: BackgroundTasks = None
):
    """Harvest rewards for a yield position"""
    try:
        # Parse gas price
        gas_price = None
        if gas_price_gwei:
            gas_price = Decimal(gas_price_gwei)
        
        # Harvest rewards
        if background_tasks:
            # Harvest in background
            background_tasks.add_task(
                defi_service.harvest_yield_position,
                position_id=position_id,
                auto_compound=auto_compound,
                gas_price_gwei=gas_price
            )
            return {"status": "queued", "position_id": position_id}
        else:
            # Harvest synchronously
            success = await defi_service.harvest_yield_position(
                position_id=position_id,
                auto_compound=auto_compound,
                gas_price_gwei=gas_price
            )
            
            if success:
                return {"status": "success", "position_id": position_id}
            else:
                raise HTTPException(status_code=500, detail="Failed to harvest yield position")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/yield/exit/{position_id}")
async def exit_yield_position(
    position_id: str = Path(..., description="Position ID"),
    gas_price_gwei: Optional[str] = Query(None, description="Gas price in Gwei"),
    background_tasks: BackgroundTasks = None
):
    """Exit a yield farming position"""
    try:
        # Parse gas price
        gas_price = None
        if gas_price_gwei:
            gas_price = Decimal(gas_price_gwei)
        
        # Exit position
        if background_tasks:
            # Exit in background
            background_tasks.add_task(
                defi_service.exit_yield_position,
                position_id=position_id,
                gas_price_gwei=gas_price
            )
            return {"status": "queued", "position_id": position_id}
        else:
            # Exit synchronously
            success = await defi_service.exit_yield_position(
                position_id=position_id,
                gas_price_gwei=gas_price
            )
            
            if success:
                return {"status": "success", "position_id": position_id}
            else:
                raise HTTPException(status_code=500, detail="Failed to exit yield position")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/portfolio/optimize", response_model=OptimizePortfolioResponse)
async def optimize_portfolio(request: OptimizePortfolioRequest):
    """Optimize portfolio allocation"""
    try:
        # Parse chain IDs
        chain_id_list = request.chain_ids
        
        # Optimize portfolio
        result = await defi_service.optimize_portfolio(
            wallet_address=request.wallet_address,
            total_value_usd=request.total_value_usd,
            risk_profile=request.risk_profile,
            chain_ids=chain_id_list,
            rebalance=request.rebalance
        )
        
        # Convert to response model
        return OptimizePortfolioResponse(
            wallet_address=result["wallet_address"],
            risk_profile=result["risk_profile"],
            total_value_usd=Decimal(result["total_value_usd"]),
            target_yield_allocation=Decimal(result["target_yield_allocation"]),
            current_yield_allocation=Decimal(result["current_yield_allocation"]),
            rebalance_needed=result["rebalance_needed"],
            rebalance_performed=result["rebalance_performed"],
            expected_monthly_return=Decimal(result["expected_monthly_return"]),
            arbitrage_opportunities=result["arbitrage_opportunities"],
            yield_opportunities=result["yield_opportunities"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def start():
    """Start the API server"""
    uvicorn.run("src.api.defi_api:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start() 