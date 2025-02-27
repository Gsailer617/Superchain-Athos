"""
DeFi CLI

This module provides a command-line interface for DeFi services:
- Arbitrage opportunity discovery and execution
- Yield farming position management
- Portfolio optimization
- Performance tracking
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union

import click
import tabulate
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.services.defi_service import defi_service
from src.services.arbitrage_service import arbitrage_service
from src.services.yield_farming_service import yield_farming_service
from src.core.dependency_injector import container

# Initialize rich console
console = Console()

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def decimal_to_str(value):
    """Convert Decimal to string for JSON serialization"""
    if isinstance(value, Decimal):
        return str(value)
    elif isinstance(value, dict):
        return {k: decimal_to_str(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [decimal_to_str(v) for v in value]
    return value

def format_usd(value):
    """Format a value as USD"""
    if isinstance(value, str):
        value = Decimal(value)
    return f"${value:.2f}"

def format_percentage(value):
    """Format a value as a percentage"""
    if isinstance(value, str):
        value = Decimal(value)
    return f"{value:.2f}%"

def parse_chain_ids(chain_ids_str):
    """Parse comma-separated chain IDs"""
    if not chain_ids_str:
        return None
    return [int(chain_id) for chain_id in chain_ids_str.split(',')]

def display_arbitrage_opportunities(opportunities, show_json=False):
    """Display arbitrage opportunities"""
    if show_json:
        console.print(json.dumps(decimal_to_str(opportunities), indent=2))
        return

    if not opportunities:
        console.print("[yellow]No arbitrage opportunities found.[/yellow]")
        return

    table = Table(title="Arbitrage Opportunities")
    table.add_column("ID", style="dim")
    table.add_column("Chain", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Expected Profit", style="magenta")
    table.add_column("Risk Level", style="yellow")

    for opp in opportunities:
        table.add_row(
            opp["id"],
            str(opp["chain_id"]),
            opp["type"],
            format_usd(opp["expected_profit_usd"]),
            opp["risk_level"]
        )

    console.print(table)

def display_yield_opportunities(opportunities, show_json=False):
    """Display yield farming opportunities"""
    if show_json:
        console.print(json.dumps(decimal_to_str(opportunities), indent=2))
        return

    if not opportunities:
        console.print("[yellow]No yield farming opportunities found.[/yellow]")
        return

    table = Table(title="Yield Farming Opportunities")
    table.add_column("ID", style="dim")
    table.add_column("Chain", style="cyan")
    table.add_column("Protocol", style="green")
    table.add_column("Pool", style="blue")
    table.add_column("APY", style="magenta")
    table.add_column("TVL", style="yellow")
    table.add_column("Risk Level", style="red")
    table.add_column("Tokens", style="dim")

    for opp in opportunities:
        table.add_row(
            opp["id"],
            str(opp["chain_id"]),
            opp["protocol"],
            opp["pool_name"],
            format_percentage(opp["apy"]),
            format_usd(opp["tvl_usd"]),
            opp["risk_level"],
            ", ".join(opp["tokens"])
        )

    console.print(table)

def display_portfolio_allocation(allocation, show_json=False):
    """Display portfolio allocation"""
    if show_json:
        console.print(json.dumps(decimal_to_str(allocation), indent=2))
        return

    console.print(f"[bold]Portfolio Allocation for {allocation.wallet_address}[/bold]")
    console.print(f"Total Value: {format_usd(allocation.total_value_usd)}")
    console.print(f"Risk Score: {allocation.risk_score}/100")
    console.print(f"Expected Monthly Return: {format_percentage(allocation.expected_monthly_return)}")
    console.print(f"Timestamp: {allocation.timestamp.isoformat()}")
    
    table = Table(title="Allocation Breakdown")
    table.add_column("Category", style="cyan")
    table.add_column("Amount", style="green")
    table.add_column("Percentage", style="yellow")
    
    total = Decimal(allocation.total_value_usd)
    
    table.add_row(
        "Arbitrage",
        format_usd(allocation.arbitrage_allocation_usd),
        format_percentage(Decimal(allocation.arbitrage_allocation_usd) / total * 100)
    )
    table.add_row(
        "Yield Farming",
        format_usd(allocation.yield_allocation_usd),
        format_percentage(Decimal(allocation.yield_allocation_usd) / total * 100)
    )
    table.add_row(
        "Reserve",
        format_usd(allocation.reserve_allocation_usd),
        format_percentage(Decimal(allocation.reserve_allocation_usd) / total * 100)
    )
    
    console.print(table)
    
    console.print("[bold]Recommended Arbitrage Opportunities:[/bold]")
    display_arbitrage_opportunities(allocation.arbitrage_opportunities, show_json=False)
    
    console.print("[bold]Recommended Yield Opportunities:[/bold]")
    display_yield_opportunities(allocation.yield_opportunities, show_json=False)

def display_performance_metrics(metrics, show_json=False):
    """Display performance metrics"""
    if show_json:
        console.print(json.dumps(decimal_to_str(metrics), indent=2))
        return

    console.print(f"[bold]Performance Metrics for {metrics.wallet_address}[/bold]")
    console.print(f"Period: {metrics.start_date.isoformat()} to {metrics.end_date.isoformat()}")
    
    table = Table(title="Performance Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Starting Balance", format_usd(metrics.starting_balance_usd))
    table.add_row("Ending Balance", format_usd(metrics.ending_balance_usd))
    table.add_row("Total Profit", format_usd(metrics.total_profit_usd))
    table.add_row("Total Profit %", format_percentage(metrics.total_profit_percentage))
    table.add_row("Annualized Return", format_percentage(metrics.annualized_return))
    table.add_row("Risk-Adjusted Return", format_percentage(metrics.risk_adjusted_return))
    
    console.print(table)
    
    table = Table(title="Profit Breakdown")
    table.add_column("Category", style="cyan")
    table.add_column("Profit", style="green")
    table.add_column("Percentage", style="yellow")
    
    total_profit = Decimal(metrics.total_profit_usd)
    
    if total_profit > 0:
        table.add_row(
            "Arbitrage",
            format_usd(metrics.arbitrage_profit_usd),
            format_percentage(Decimal(metrics.arbitrage_profit_usd) / total_profit * 100)
        )
        table.add_row(
            "Yield Farming",
            format_usd(metrics.yield_farming_profit_usd),
            format_percentage(Decimal(metrics.yield_farming_profit_usd) / total_profit * 100)
        )
    else:
        table.add_row("Arbitrage", format_usd(metrics.arbitrage_profit_usd), "N/A")
        table.add_row("Yield Farming", format_usd(metrics.yield_farming_profit_usd), "N/A")
    
    console.print(table)
    
    table = Table(title="Activity Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="green")
    
    table.add_row("Successful Arbitrages", str(metrics.successful_arbitrages))
    table.add_row("Failed Arbitrages", str(metrics.failed_arbitrages))
    table.add_row("Active Yield Positions", str(metrics.active_yield_positions))
    table.add_row("Closed Yield Positions", str(metrics.closed_yield_positions))
    
    console.print(table)
    
    console.print(f"Best Performing Strategy: [green]{metrics.best_performing_strategy}[/green]")
    console.print(f"Worst Performing Strategy: [red]{metrics.worst_performing_strategy}[/red]")

def display_yield_position(position, show_json=False):
    """Display yield position"""
    if show_json:
        console.print(json.dumps(decimal_to_str(position), indent=2))
        return

    console.print(f"[bold]Yield Position {position.id}[/bold]")
    console.print(f"Wallet: {position.wallet_address}")
    console.print(f"Opportunity ID: {position.opportunity_id}")
    console.print(f"Status: {position.status}")
    
    table = Table(title="Position Details")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Deposit Value", format_usd(position.deposit_usd_value))
    table.add_row("Current Value", format_usd(position.current_value_usd))
    table.add_row("Profit", format_usd(position.profit_usd))
    table.add_row("APY Realized", format_percentage(position.apy_realized))
    table.add_row("Entry Date", position.entry_timestamp.isoformat())
    
    if position.last_harvest_timestamp:
        table.add_row("Last Harvest", position.last_harvest_timestamp.isoformat())
    else:
        table.add_row("Last Harvest", "Never")
    
    console.print(table)
    
    table = Table(title="Deposit Amounts")
    table.add_column("Token", style="cyan")
    table.add_column("Amount", style="green")
    
    for token, amount in position.deposit_amounts.items():
        table.add_row(token, str(amount))
    
    console.print(table)
    
    if position.harvested_rewards:
        table = Table(title="Harvested Rewards")
        table.add_column("Token", style="cyan")
        table.add_column("Amount", style="green")
        
        for token, amount in position.harvested_rewards.items():
            table.add_row(token, str(amount))
        
        console.print(table)

def display_optimize_result(result, show_json=False):
    """Display portfolio optimization result"""
    if show_json:
        console.print(json.dumps(decimal_to_str(result), indent=2))
        return

    console.print(f"[bold]Portfolio Optimization for {result['wallet_address']}[/bold]")
    console.print(f"Risk Profile: {result['risk_profile']}")
    console.print(f"Total Value: {format_usd(result['total_value_usd'])}")
    console.print(f"Expected Monthly Return: {format_percentage(result['expected_monthly_return'])}")
    
    table = Table(title="Allocation Details")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Target Yield Allocation", format_usd(result['target_yield_allocation']))
    table.add_row("Current Yield Allocation", format_usd(result['current_yield_allocation']))
    table.add_row("Rebalance Needed", "Yes" if result['rebalance_needed'] else "No")
    table.add_row("Rebalance Performed", "Yes" if result['rebalance_performed'] else "No")
    table.add_row("Arbitrage Opportunities", str(result['arbitrage_opportunities']))
    table.add_row("Yield Opportunities", str(result['yield_opportunities']))
    
    console.print(table)

# -----------------------------------------------------------------------------
# CLI Commands
# -----------------------------------------------------------------------------

@click.group()
def cli():
    """DeFi CLI - Command-line interface for DeFi operations"""
    pass

# -----------------------------------------------------------------------------
# Opportunities Commands
# -----------------------------------------------------------------------------

@cli.group()
def opportunities():
    """Commands for managing DeFi opportunities"""
    pass

@opportunities.command("list")
@click.option("--wallet", required=True, help="Wallet address")
@click.option("--chains", default="1,56,137", help="Comma-separated list of chain IDs")
@click.option("--risk", default="moderate", help="Risk profile (conservative, moderate, aggressive)")
@click.option("--max", default=10, help="Maximum number of opportunities to return")
@click.option("--arbitrage/--no-arbitrage", default=True, help="Include arbitrage opportunities")
@click.option("--yield/--no-yield", default=True, help="Include yield farming opportunities")
@click.option("--json", is_flag=True, help="Output as JSON")
async def list_opportunities(wallet, chains, risk, max, arbitrage, yield_, json):
    """List DeFi opportunities for a wallet"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Fetching opportunities...", total=None)
        
        try:
            # Parse chain IDs
            chain_id_list = parse_chain_ids(chains)
            
            # Get opportunities
            opportunities = await defi_service.get_defi_opportunities(
                wallet_address=wallet,
                chain_ids=chain_id_list,
                risk_profile=risk,
                max_opportunities=max,
                include_arbitrage=arbitrage,
                include_yield=yield_
            )
            
            # Display opportunities
            console.print(f"[bold]DeFi Opportunities for {wallet}[/bold]")
            console.print(f"Risk Profile: {opportunities['risk_profile']}")
            console.print(f"Max Risk Level: {opportunities['max_risk_level']}")
            
            if arbitrage:
                console.print("\n[bold]Arbitrage Opportunities:[/bold]")
                display_arbitrage_opportunities(opportunities["arbitrage"], show_json=json)
            
            if yield_:
                console.print("\n[bold]Yield Farming Opportunities:[/bold]")
                display_yield_opportunities(opportunities["yield_farming"], show_json=json)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

# -----------------------------------------------------------------------------
# Portfolio Commands
# -----------------------------------------------------------------------------

@cli.group()
def portfolio():
    """Commands for managing DeFi portfolio"""
    pass

@portfolio.command("allocation")
@click.option("--wallet", required=True, help="Wallet address")
@click.option("--value", required=True, help="Total portfolio value in USD")
@click.option("--risk", default="moderate", help="Risk profile (conservative, moderate, aggressive)")
@click.option("--chains", default=None, help="Comma-separated list of chain IDs")
@click.option("--json", is_flag=True, help="Output as JSON")
async def get_allocation(wallet, value, risk, chains, json):
    """Get portfolio allocation recommendation"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Calculating allocation...", total=None)
        
        try:
            # Parse chain IDs
            chain_id_list = parse_chain_ids(chains)
            
            # Get allocation
            allocation = await defi_service.get_portfolio_allocation(
                wallet_address=wallet,
                total_value_usd=Decimal(value),
                risk_profile=risk,
                chain_ids=chain_id_list
            )
            
            # Display allocation
            display_portfolio_allocation(allocation, show_json=json)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

@portfolio.command("metrics")
@click.option("--wallet", required=True, help="Wallet address")
@click.option("--start-date", default=None, help="Start date (ISO format)")
@click.option("--end-date", default=None, help="End date (ISO format)")
@click.option("--json", is_flag=True, help="Output as JSON")
async def get_metrics(wallet, start_date, end_date, json):
    """Get performance metrics for a wallet"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Calculating metrics...", total=None)
        
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
                wallet_address=wallet,
                start_date=start_date_obj,
                end_date=end_date_obj
            )
            
            # Display metrics
            display_performance_metrics(metrics, show_json=json)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

@portfolio.command("optimize")
@click.option("--wallet", required=True, help="Wallet address")
@click.option("--value", required=True, help="Total portfolio value in USD")
@click.option("--risk", default="moderate", help="Risk profile (conservative, moderate, aggressive)")
@click.option("--chains", default=None, help="Comma-separated list of chain IDs")
@click.option("--rebalance/--no-rebalance", default=False, help="Whether to rebalance the portfolio")
@click.option("--json", is_flag=True, help="Output as JSON")
async def optimize_portfolio(wallet, value, risk, chains, rebalance, json):
    """Optimize portfolio allocation"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Optimizing portfolio...", total=None)
        
        try:
            # Parse chain IDs
            chain_id_list = parse_chain_ids(chains)
            
            # Optimize portfolio
            result = await defi_service.optimize_portfolio(
                wallet_address=wallet,
                total_value_usd=Decimal(value),
                risk_profile=risk,
                chain_ids=chain_id_list,
                rebalance=rebalance
            )
            
            # Display result
            display_optimize_result(result, show_json=json)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

# -----------------------------------------------------------------------------
# Arbitrage Commands
# -----------------------------------------------------------------------------

@cli.group()
def arbitrage():
    """Commands for managing arbitrage opportunities"""
    pass

@arbitrage.command("execute")
@click.option("--id", required=True, help="Opportunity ID")
@click.option("--wallet", required=True, help="Wallet address")
@click.option("--gas", default=None, help="Gas price in Gwei")
async def execute_arbitrage(id, wallet, gas):
    """Execute an arbitrage opportunity"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Executing arbitrage...", total=None)
        
        try:
            # Parse gas price
            gas_price = None
            if gas:
                gas_price = Decimal(gas)
            
            # Execute arbitrage
            success = await defi_service.execute_arbitrage_opportunity(
                opportunity_id=id,
                wallet_address=wallet,
                gas_price_gwei=gas_price
            )
            
            if success:
                console.print(f"[bold green]Successfully executed arbitrage opportunity {id}[/bold green]")
            else:
                console.print(f"[bold red]Failed to execute arbitrage opportunity {id}[/bold red]")
                sys.exit(1)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

# -----------------------------------------------------------------------------
# Yield Farming Commands
# -----------------------------------------------------------------------------

@cli.group()
def yield_():
    """Commands for managing yield farming positions"""
    pass

@yield_.command("create")
@click.option("--id", required=True, help="Opportunity ID")
@click.option("--wallet", required=True, help="Wallet address")
@click.option("--deposit", required=True, multiple=True, help="Deposit amount in format 'token_address:amount'")
@click.option("--gas", default=None, help="Gas price in Gwei")
async def create_position(id, wallet, deposit, gas):
    """Create a yield farming position"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Creating yield position...", total=None)
        
        try:
            # Parse deposit amounts
            deposit_amounts = {}
            for dep in deposit:
                token_address, amount = dep.split(":")
                deposit_amounts[token_address] = Decimal(amount)
            
            # Parse gas price
            gas_price = None
            if gas:
                gas_price = Decimal(gas)
            
            # Create position
            position_id = await defi_service.create_yield_position(
                opportunity_id=id,
                wallet_address=wallet,
                deposit_amounts=deposit_amounts,
                gas_price_gwei=gas_price
            )
            
            if position_id:
                console.print(f"[bold green]Successfully created yield position: {position_id}[/bold green]")
            else:
                console.print(f"[bold red]Failed to create yield position[/bold red]")
                sys.exit(1)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

@yield_.command("get")
@click.option("--id", required=True, help="Position ID")
@click.option("--json", is_flag=True, help="Output as JSON")
async def get_position(id, json):
    """Get a yield position by ID"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Fetching yield position...", total=None)
        
        try:
            # Get position
            position = await yield_farming_service.get_position(id)
            
            if not position:
                console.print(f"[bold red]Position {id} not found[/bold red]")
                sys.exit(1)
            
            # Display position
            display_yield_position(position, show_json=json)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

@yield_.command("list")
@click.option("--wallet", required=True, help="Wallet address")
@click.option("--chain", default=None, type=int, help="Chain ID")
@click.option("--status", default=None, help="Position status")
@click.option("--json", is_flag=True, help="Output as JSON")
async def list_positions(wallet, chain, status, json):
    """List all yield positions for a wallet"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Fetching yield positions...", total=None)
        
        try:
            # Get positions
            positions = await yield_farming_service.get_positions_by_wallet(
                wallet_address=wallet,
                chain_id=chain,
                status=status
            )
            
            if not positions:
                console.print(f"[yellow]No yield positions found for wallet {wallet}[/yellow]")
                return
            
            if json:
                console.print(json.dumps(decimal_to_str(positions), indent=2))
                return
            
            # Display positions
            console.print(f"[bold]Yield Positions for {wallet}[/bold]")
            
            table = Table(title=f"Yield Positions ({len(positions)} positions)")
            table.add_column("ID", style="dim")
            table.add_column("Opportunity", style="cyan")
            table.add_column("Deposit Value", style="green")
            table.add_column("Current Value", style="blue")
            table.add_column("Profit", style="magenta")
            table.add_column("APY", style="yellow")
            table.add_column("Status", style="red")
            
            for position in positions:
                profit_style = "green" if Decimal(position.profit_usd) >= 0 else "red"
                
                table.add_row(
                    position.id,
                    position.opportunity_id,
                    format_usd(position.deposit_usd_value),
                    format_usd(position.current_value_usd),
                    f"[{profit_style}]{format_usd(position.profit_usd)}[/{profit_style}]",
                    format_percentage(position.apy_realized),
                    position.status
                )
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

@yield_.command("harvest")
@click.option("--id", required=True, help="Position ID")
@click.option("--compound/--no-compound", default=True, help="Whether to auto-compound rewards")
@click.option("--gas", default=None, help="Gas price in Gwei")
async def harvest_position(id, compound, gas):
    """Harvest rewards for a yield position"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Harvesting rewards...", total=None)
        
        try:
            # Parse gas price
            gas_price = None
            if gas:
                gas_price = Decimal(gas)
            
            # Harvest rewards
            success = await defi_service.harvest_yield_position(
                position_id=id,
                auto_compound=compound,
                gas_price_gwei=gas_price
            )
            
            if success:
                console.print(f"[bold green]Successfully harvested rewards for position {id}[/bold green]")
            else:
                console.print(f"[bold red]Failed to harvest rewards for position {id}[/bold red]")
                sys.exit(1)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

@yield_.command("exit")
@click.option("--id", required=True, help="Position ID")
@click.option("--gas", default=None, help="Gas price in Gwei")
async def exit_position(id, gas):
    """Exit a yield farming position"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Exiting position...", total=None)
        
        try:
            # Parse gas price
            gas_price = None
            if gas:
                gas_price = Decimal(gas)
            
            # Exit position
            success = await defi_service.exit_yield_position(
                position_id=id,
                gas_price_gwei=gas_price
            )
            
            if success:
                console.print(f"[bold green]Successfully exited position {id}[/bold green]")
            else:
                console.print(f"[bold red]Failed to exit position {id}[/bold red]")
                sys.exit(1)
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    """Run the CLI"""
    # Create event loop
    loop = asyncio.get_event_loop()
    
    # Run the CLI
    try:
        loop.run_until_complete(cli())
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)
    finally:
        loop.close()

if __name__ == "__main__":
    main() 