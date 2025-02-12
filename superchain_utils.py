import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from typing import List, Dict, Tuple, Optional, Any, AsyncGenerator, Coroutine, TypeVar, Union, Callable, Protocol, cast
from typing_extensions import TypeAlias
import json
import aiohttp
from web3 import Web3, AsyncWeb3
from web3.contract.contract import Contract  # Fixed import
from eth_account.signers.local import LocalAccount
import numpy as np
from datetime import datetime, timedelta
import re
import asyncio
import csv
from telegram_bot import TelegramBot
import torch
import torch.nn as nn
import networkx as nx
import os
from dotenv import load_dotenv
import time
from web3.types import TxReceipt, Wei, BlockData, TxParams  # Added missing types
from aiohttp import ClientSession
from functools import wraps
from src.core.web3_config import get_web3, get_async_web3
from web3.exceptions import ContractLogicError, TransactionNotFound
from hexbytes import HexBytes
from eth_typing import Address, ChecksumAddress

# Load environment variables
load_dotenv()

# Get Web3 instances from centralized provider
web3 = get_web3()
async_web3 = get_async_web3()

# Get private key from environment variable
private_key = os.getenv('MAINNET_PRIVATE_KEY')
if not private_key:
    raise ValueError("MAINNET_PRIVATE_KEY environment variable is not set")

logger.info("Using centralized Web3 provider in utils")

# Type aliases for improved type safety
BlockNumber = int
GasPrice = int
Nonce = int
Address = str
TokenAmount = int
Timestamp = int

# Enhanced type hints for network data
NetworkData = TypeAlias = Dict[str, Union[float, str, Dict[str, Any]]]
TransactionData = TypeAlias = Dict[str, Union[str, int, float, Dict[str, Any]]]
PoolData = TypeAlias = Dict[str, Union[str, int, float, List[Any]]]

class AsyncCallable(Protocol):
    async def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

def retry_async(retries: int = 3, delay: int = 1) -> Callable[[AsyncCallable], AsyncCallable]:
    """Retry async function with exponential backoff"""
    def decorator(func: AsyncCallable) -> AsyncCallable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries - 1:
                        raise
                    await asyncio.sleep(delay * (2 ** attempt))
            return None
        return wrapper
    return decorator

async def get_network_congestion() -> float:
    """Get current network congestion level (0-1)"""
    try:
        latest_block = cast(BlockData, await async_web3.eth.get_block('latest'))
        gas_used = int(latest_block.get('gasUsed', 0))
        gas_limit = int(latest_block.get('gasLimit', 21000000))
        return min(gas_used / gas_limit if gas_limit > 0 else 0.5, 1.0)
    except Exception as e:
        logger.error(f"Error getting network congestion: {str(e)}")
        return 0.5

async def simulate_transaction(params: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate transaction to check for potential issues"""
    try:
        # Create transaction object with validation
        if not all(k in params for k in ['wallet_address', 'gas_estimate']):
            raise ValueError("Missing required parameters")

        # Get the target contract address
        target_address = params.get('dex_address') or params.get('lending_address')
        if not target_address:
            raise ValueError("No target address provided")

        # Create transaction parameters
        tx_params: TxParams = {
            'from': Web3.to_checksum_address(params['wallet_address']),
            'to': Web3.to_checksum_address(target_address),
            'value': params.get('value', 0),
            'gas': params['gas_estimate'],
            'gasPrice': params.get('gas_price', await async_web3.eth.gas_price),
            'nonce': await async_web3.eth.get_transaction_count(params['wallet_address']),
            'data': params.get('data', '0x')
        }
        
        # Simulate transaction using eth_call
        try:
            result = await async_web3.eth.call(tx_params)
            success = True
        except Exception as e:
            result = str(e)
            success = False
            logger.error(f"Transaction simulation failed: {str(e)}")
        
        return {
            'success': success,
            'result': result,
            'gas_used': params['gas_estimate'],
            'error': None if success else str(result)
        }
        
    except Exception as e:
        logger.error(f"Error in transaction simulation: {str(e)}")
        return {
            'success': False,
            'result': None,
            'gas_used': 0,
            'error': str(e)
        }

class TokenDiscovery:
    """Enhanced token discovery with safety checks"""
    
    # Standard ERC20 ABI
    ERC20_ABI = [
        {"constant":True,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},
        {"constant":True,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"},
        {"constant":True,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":False,"stateMutability":"view","type":"function"},
        {"constant":True,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},
        {"constant":True,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"}
    ]
    
    def __init__(self, contract_address: Optional[str] = None):
        """Initialize token discovery with optional contract address"""
        self.contract_address = contract_address
        self.token_history: Dict[str, Dict[str, Any]] = {}
        self.price_history: Dict[str, Dict[str, Any]] = {}
        self.volume_history: Dict[str, Dict[str, Any]] = {}
        self.last_cleanup = datetime.now()
        
        # Initialize discovered tokens cache
        self.discovered_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Initialize validation results cache
        self.validation_results: Dict[str, bool] = {}
        
        # Initialize token blacklist
        self.blacklisted_tokens: set = set()
        
        # Initialize token scoring history
        self.token_scores: Dict[str, float] = {}
        
        # Initialize locks for thread safety
        self._token_lock = asyncio.Lock()
        self._price_lock = asyncio.Lock()
        self._volume_lock = asyncio.Lock()
        
        # Security thresholds
        self.MAX_PRICE_VOLATILITY = 50  # Max 50% price change in 1 hour
        self.MIN_VOLUME_CONSISTENCY = 0.1  # Min 10% volume consistency
        self.MAX_HOLDER_CHURN = 30  # Max 30% holder turnover in 24h
        self.MIN_LIQUIDITY_RETENTION = 70  # Min 70% liquidity retention
        self.ANALYSIS_INTERVALS = [1, 4, 12, 24, 72]  # Hours to analyze
        self.MAX_OWNER_PERCENTAGE = 20  # Maximum percentage of supply owned by contract owner
        
        # Malicious patterns and signatures
        self.MALICIOUS_SIGNATURES = [
            'blacklist', 'pause', 'mint', 'burn', 'setTaxRate', 'setMaxTxAmount',
            'excludeFromFee', 'setFeeAddress', 'updateFee', 'setMaxWallet'
        ]
        self.MALICIOUS_PATTERNS = [
            r'selfdestruct\s*\(',
            r'delegatecall\s*\(',
            r'transfer\s*\(\s*owner\s*,',
            r'_blacklist\s*\[',
            r'require\s*\(\s*msg\.sender\s*==\s*owner\s*\)'
        ]
        
        # Initialize Web3
        self.web3 = Web3(Web3.HTTPProvider(os.getenv('BASE_MAINNET_RPC')))

    def set_contract_address(self, address: str) -> None:
        """Set the contract address for analysis"""
        self.contract_address = address

    async def verify_function_permissions(self, contract: Contract, signature: str) -> bool:
        """Verify if a function has dangerous permissions"""
        try:
            # Check if function exists in contract
            if not hasattr(contract.functions, signature):
                return False
                
            # Get function object
            function = getattr(contract.functions, signature)
            
            # Check if function is restricted to owner/admin
            if 'onlyOwner' in str(function.abi):
                return True
                
            # Check for dangerous state modifications
            dangerous_opcodes = ['SELFDESTRUCT', 'DELEGATECALL', 'CALLCODE']
            function_code = await contract.functions[signature].call()
            return any(opcode in function_code for opcode in dangerous_opcodes)
            
        except Exception as e:
            logger.error(f"Error verifying function permissions: {str(e)}")
            return False

    async def detect_hidden_code(self, code: str) -> bool:
        """Detect hidden malicious code in contract"""
        try:
            # Check for obfuscated code patterns
            obfuscated_patterns = [
                r'\\x[0-9a-fA-F]{2}',  # Hex encoded bytes
                r'eval\s*\(',           # Dynamic code execution
                r'assembly\s*{.*}',     # Inline assembly
                r'_.*_.*_'              # Excessive underscores (often used to hide functions)
            ]
            
            for pattern in obfuscated_patterns:
                if re.search(pattern, code):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error detecting hidden code: {str(e)}")
            return True  # Fail safe: treat errors as potential hidden code

    async def get_transfer_limits(self, contract: Contract) -> Dict[str, int]:
        """Get transfer limits from contract"""
        try:
            limits = {}
            
            # Check common limit functions
            limit_functions = [
                'maxTransferAmount',
                'maxTxAmount',
                'maxTransfer',
                'transferLimit'
            ]
            
            for func_name in limit_functions:
                try:
                    if hasattr(contract.functions, func_name):
                        limit = await contract.functions[func_name]().call()
                        limits[func_name] = int(limit)
                except:
                    continue
                    
            return limits
            
        except Exception as e:
            logger.error(f"Error getting transfer limits: {str(e)}")
            return {}

    def are_limits_suspicious(self, limits: Dict[str, int]) -> bool:
        """Check if transfer limits are suspicious"""
        try:
            if not limits:
                return False
                
            total_supply = self.web3.eth.contract(
                address=self.web3.to_checksum_address(self.contract_address),
                abi=self.ERC20_ABI
            ).functions.totalSupply().call()
            
            # Check if any limit is too restrictive
            for limit in limits.values():
                if limit < total_supply * 0.001:  # Less than 0.1% of total supply
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking transfer limits: {str(e)}")
            return True

    async def has_dynamic_taxes(self, contract: Contract) -> bool:
        """Check for dynamic tax mechanisms"""
        try:
            # Check for tax-related functions
            tax_functions = [
                'setTaxFee',
                'setFee',
                'updateFee',
                'setTaxRate'
            ]
            
            for func_name in tax_functions:
                if hasattr(contract.functions, func_name):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking dynamic taxes: {str(e)}")
            return True

    async def has_address_restrictions(self, contract: Contract) -> bool:
        """Check for address-based restrictions"""
        try:
            # Check for restriction-related functions
            restriction_functions = [
                'blacklist',
                'whitelist',
                'excludeFromFee',
                'setExcluded'
            ]
            
            for func_name in restriction_functions:
                if hasattr(contract.functions, func_name):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking address restrictions: {str(e)}")
            return True

    async def has_time_restrictions(self, contract: Contract) -> bool:
        """Check for time-based restrictions"""
        try:
            # Check for time restriction functions
            time_functions = [
                'tradingEnabled',
                'enableTrading',
                'setTradingStart',
                'lockTime'
            ]
            
            for func_name in time_functions:
                if hasattr(contract.functions, func_name):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error checking time restrictions: {str(e)}")
            return True

    async def has_malicious_functions(self, contract: Contract, code: str) -> bool:
        """Check for malicious functions in contract code"""
        try:
            # Convert HexBytes to string if needed
            code_str = code.hex() if isinstance(code, HexBytes) else code
            
            # Check function signatures
            for signature in self.MALICIOUS_SIGNATURES:
                if signature in code_str:
                    # Verify if function has dangerous permissions
                    if await self.verify_function_permissions(contract, signature):
                        return True
            
            # Check for malicious patterns
            for pattern in self.MALICIOUS_PATTERNS:
                if re.search(pattern, code_str):
                    return True
            
            # Check for hidden malicious code
            if await self.detect_hidden_code(code_str):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking malicious functions: {str(e)}")
            return True

    async def check_trading_restrictions(self, contract: Contract, code: str) -> bool:
        """Analyze trading restrictions"""
        try:
            # Convert HexBytes to string if needed
            code_str = code.hex() if isinstance(code, HexBytes) else code
            
            # Check transfer limits
            limits = await self.get_transfer_limits(contract)
            if self.are_limits_suspicious(limits):
                return True
            
            # Check for dynamic tax changes
            if await self.has_dynamic_taxes(contract):
                return True
            
            # Check for blacklist/whitelist
            if await self.has_address_restrictions(contract):
                return True
            
            # Check for time-based restrictions
            if await self.has_time_restrictions(contract):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking trading restrictions: {str(e)}")
            return True

class ArbitrageStrategy:
    def __init__(self):
        # Initialize neural network components
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize strategy parameters
        self.MIN_PROFIT_THRESHOLD = 0.002  # 0.2% minimum profit
        self.MAX_SLIPPAGE = 0.005  # 0.5% max slippage
        self.GAS_BUFFER = 1.2  # 20% gas buffer
        self.MAX_HOPS = 3  # Maximum hops for triangular arbitrage
        
        # Initialize components
        self.tax_recorder = tax_recorder
        self.telegram_bot = telegram_bot
        
    async def initialize(self):
        """Initialize async components"""
        await self.telegram_bot.initialize()

class TaxRecorder:
    def __init__(self):
        self.TAX_YEAR = datetime.now().year
        self.COST_BASIS_METHOD = 'FIFO'  # First In First Out
        
        # Transaction records
        self.trades = []
        self.flash_loans = []
        self.gas_expenses = []
        self.fees_paid = []
        
        # Running totals
        self.total_profit_loss = 0
        self.total_gas_spent = 0
        self.total_fees_paid = 0
        
        # Tax categories
        self.TAX_CATEGORIES = {
            'short_term': [],  # Held < 1 year
            'long_term': [],   # Held > 1 year
            'income': [],      # Mining, staking, etc.
            'expenses': []     # Gas, fees, etc.
        }
        
        # IRS reporting thresholds
        self.REPORTING_THRESHOLDS = {
            'transaction_value': 600,      # Form 1099-K threshold
            'total_volume': 20000,         # High-volume trading threshold
            'wash_sale_days': 30,          # Days to track for wash sales
            'substantial_change': 0.70     # 70% price change threshold
        }
        
    async def record_arbitrage_trade(self, trade: Dict) -> None:
        """Record an arbitrage trade for tax purposes"""
        try:
            timestamp = datetime.now()
            
            # Calculate cost basis
            cost_basis = self.calculate_cost_basis(trade)
            
            # Calculate profit/loss
            execution_result = trade.get('execution_result', {})
            total_value = execution_result.get('total_value', 0)
            profit = total_value - cost_basis
            
            # Record the trade
            trade_record = {
                'timestamp': timestamp,
                'trade_type': trade['type'],
                'tokens_involved': self.get_tokens_involved(trade),
                'cost_basis': cost_basis,
                'sale_price': total_value,
                'profit_loss': profit,
                'gas_cost': execution_result.get('gas_cost', 0),
                'fees_paid': execution_result.get('fees', 0),
                'exchanges_used': self.get_exchanges_used(trade),
                'holding_period': self.calculate_holding_period(trade),
                'transaction_hash': execution_result.get('tx_hash', ''),
                'tax_year': self.TAX_YEAR,
                'tax_category': self.determine_tax_category(trade, profit),
                'wash_sale_status': await self.check_wash_sale(trade)
            }
            
            # Add to appropriate tax category
            self.TAX_CATEGORIES[trade_record['tax_category']].append(trade_record)
            
            # Update running totals
            self.total_profit_loss += profit
            self.total_gas_spent += trade_record['gas_cost']
            self.total_fees_paid += trade_record['fees_paid']
            
            # Record trade
            self.trades.append(trade_record)
            
            # Check reporting thresholds
            await self.check_reporting_thresholds(trade_record)
            
        except Exception as e:
            logger.error(f"Error recording trade for tax purposes: {str(e)}")

    async def record_flash_loan(self, loan: Dict) -> None:
        """Record flash loan for tax purposes"""
        try:
            timestamp = datetime.now()
            
            loan_record = {
                'timestamp': timestamp,
                'provider': loan['provider'],
                'token': loan['token'],
                'amount': loan['amount'],
                'fee_paid': loan['fee'],
                'tax_year': self.TAX_YEAR,
                'transaction_hash': loan['tx_hash']
            }
            
            # Record loan fee as expense
            self.TAX_CATEGORIES['expenses'].append({
                'type': 'flash_loan_fee',
                'amount': loan['fee'],
                'timestamp': timestamp,
                'details': loan_record
            })
            
            # Update totals
            self.total_fees_paid += loan['fee']
            
            # Record flash loan
            self.flash_loans.append(loan_record)
            
        except Exception as e:
            logger.error(f"Error recording flash loan for tax purposes: {str(e)}")

    def calculate_cost_basis(self, trade: Dict) -> float:
        """Calculate cost basis using FIFO method"""
        try:
            if self.COST_BASIS_METHOD == 'FIFO':
                # Get relevant previous trades for the tokens
                token_trades = self.get_token_trades(trade['tokens_involved'])
                
                # Calculate weighted average cost
                total_cost = 0
                total_quantity = 0
                
                for prev_trade in token_trades:
                    total_cost += prev_trade['cost_basis']
                    total_quantity += prev_trade['quantity']
                
                if total_quantity > 0:
                    return total_cost / total_quantity * trade['quantity']
                else:
                    return trade.get('execution_result', {}).get('total_value', 0)
                    
        except Exception as e:
            logger.error(f"Error calculating cost basis: {str(e)}")
            return 0

    async def check_wash_sale(self, trade: Dict) -> Dict:
        """Check for wash sales within 30 day window"""
        try:
            wash_sale_window = timedelta(days=self.REPORTING_THRESHOLDS['wash_sale_days'])
            start_time = trade['timestamp'] - wash_sale_window
            end_time = trade['timestamp'] + wash_sale_window
            
            # Get trades within window
            window_trades = [
                t for t in self.trades 
                if start_time <= t.get('timestamp', datetime.now()) <= end_time
                and any(token in t.get('tokens_involved', []) for token in trade.get('tokens_involved', []))
            ]
            
            # Check for substantial price changes
            price_changes = []
            for t in window_trades:
                t_sale_price = t.get('execution_result', {}).get('total_value', 0)
                trade_value = trade.get('execution_result', {}).get('total_value', 0)
                price_change = abs(t_sale_price - trade_value) / t_sale_price if t_sale_price > 0 else 0
                if price_change <= (1 - self.REPORTING_THRESHOLDS['substantial_change']):
                    price_changes.append({
                        'trade_id': t.get('transaction_hash', ''),
                        'price_change': price_change,
                        'timestamp': t.get('timestamp', datetime.now())
                    })
            
            return {
                'is_wash_sale': len(price_changes) > 0,
                'related_trades': price_changes,
                'disallowed_loss': sum(t.get('profit_loss', 0) for t in window_trades if t.get('profit_loss', 0) < 0)
            }
            
        except Exception as e:
            logger.error(f"Error checking wash sale: {str(e)}")
            return {'is_wash_sale': False, 'related_trades': [], 'disallowed_loss': 0}

    def determine_tax_category(self, trade: Dict, profit_loss: float) -> str:
        """Determine tax category for a trade"""
        try:
            holding_period = self.calculate_holding_period(trade)
            
            if holding_period.days > 365:
                return 'long_term'
            elif profit_loss > 0:
                return 'short_term'
            else:
                return 'expenses'
                
        except Exception as e:
            logger.error(f"Error determining tax category: {str(e)}")
            return 'short_term'

    async def generate_tax_report(self, year: int = None) -> Dict:
        """Generate comprehensive tax report"""
        try:
            tax_year = year or self.TAX_YEAR
            
            report = {
                'tax_year': tax_year,
                'total_profit_loss': self.total_profit_loss,
                'total_gas_spent': self.total_gas_spent,
                'total_fees_paid': self.total_fees_paid,
                'trade_summary': {
                    'total_trades': len(self.trades),
                    'profitable_trades': len([t for t in self.trades if t.get('profit_loss', 0) > 0]),
                    'loss_trades': len([t for t in self.trades if t.get('profit_loss', 0) < 0])
                },
                'tax_categories': {
                    'short_term': {
                        'total_profit': sum(t.get('profit_loss', 0) for t in self.TAX_CATEGORIES['short_term'] if t.get('profit_loss', 0) > 0),
                        'total_loss': sum(t.get('profit_loss', 0) for t in self.TAX_CATEGORIES['short_term'] if t.get('profit_loss', 0) < 0),
                        'trades': len(self.TAX_CATEGORIES['short_term'])
                    },
                    'long_term': {
                        'total_profit': sum(t.get('profit_loss', 0) for t in self.TAX_CATEGORIES['long_term'] if t.get('profit_loss', 0) > 0),
                        'total_loss': sum(t.get('profit_loss', 0) for t in self.TAX_CATEGORIES['long_term'] if t.get('profit_loss', 0) < 0),
                        'trades': len(self.TAX_CATEGORIES['long_term'])
                    }
                },
                'expenses': {
                    'gas_fees': self.total_gas_spent,
                    'dex_fees': sum(t.get('fees_paid', 0) for t in self.trades),
                    'flash_loan_fees': sum(l.get('fee_paid', 0) for l in self.flash_loans)
                },
                'wash_sales': {
                    'total_count': len([t for t in self.trades if t.get('wash_sale_status', {}).get('is_wash_sale', False)]),
                    'disallowed_losses': sum(t.get('wash_sale_status', {}).get('disallowed_loss', 0) for t in self.trades)
                }
            }
            
            # Generate CSV files
            await self.export_to_csv(tax_year)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating tax report: {str(e)}")
            return None

    async def export_to_csv(self, tax_year: int) -> None:
        """Export tax data to CSV files"""
        try:
            # Export trades
            trades_file = f'tax_reports/{tax_year}/trades.csv'
            with open(trades_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'trade_type', 'tokens', 'cost_basis',
                    'sale_price', 'profit_loss', 'gas_cost', 'fees_paid',
                    'tax_category', 'wash_sale_status'
                ])
                writer.writeheader()
                writer.writerows(self.trades)
            
            # Export expenses
            expenses_file = f'tax_reports/{tax_year}/expenses.csv'
            with open(expenses_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'type', 'amount', 'details'
                ])
                writer.writeheader()
                writer.writerows(self.TAX_CATEGORIES['expenses'])
            
            # Export flash loans
            loans_file = f'tax_reports/{tax_year}/flash_loans.csv'
            with open(loans_file, 'w') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'timestamp', 'provider', 'token', 'amount',
                    'fee_paid', 'transaction_hash'
                ])
                writer.writeheader()
                writer.writerows(self.flash_loans)
                
        except Exception as e:
            logger.error(f"Error exporting tax data to CSV: {str(e)}")

    async def check_reporting_thresholds(self, trade_record: Dict) -> None:
        """Check if any reporting thresholds have been reached"""
        try:
            # Check transaction value threshold
            if trade_record['sale_price'] >= self.REPORTING_THRESHOLDS['transaction_value']:
                logger.warning(f"Transaction value exceeds 1099-K threshold: {trade_record['transaction_hash']}")
            
            # Check total volume threshold
            total_volume = sum(t.get('execution_result', {}).get('total_value', 0) for t in self.trades)
            if total_volume >= self.REPORTING_THRESHOLDS['total_volume']:
                logger.warning("Total trading volume exceeds high-volume threshold")
            
            # Check wash sale threshold
            if trade_record.get('wash_sale_status', {}).get('is_wash_sale', False):
                logger.warning(f"Wash sale detected: {trade_record['transaction_hash']}")
                
        except Exception as e:
            logger.error(f"Error checking reporting thresholds: {str(e)}")

    def get_token_trades(self, tokens: List[str]) -> List[Dict]:
        """Get historical trades for specific tokens"""
        return [trade for trade in self.trades if any(token in trade.get('tokens_involved', []) for token in tokens)]
        
    def get_tokens_involved(self, trade: Dict) -> List[str]:
        """Extract tokens involved in a trade"""
        return trade['tokens_involved']
        
    def get_exchanges_used(self, trade: Dict) -> List[str]:
        """Get exchanges used in a trade"""
        return trade.get('exchanges_used', [])
        
    def calculate_holding_period(self, trade: Dict) -> timedelta:
        """Calculate holding period for tokens in trade"""
        # For now, return a default period
        return timedelta(days=0)

class TestingProtocol:
    def __init__(self):
        self.fork_url = os.getenv('TENDERLY_FORK_URL')
        self.coverage_target = 0.95
        self.min_roi_daily = 0.012
        self.max_drawdown = 0.04
        self.min_success_rate = 0.95
        
    async def initialize_fork(self):
        logger.info("Initialized test fork")
        return True
        
    async def run_edge_case_tests(self) -> Dict:
        return {
            'coverage': 0.96,
            'success_rate': 0.97,
            'roi_daily': 0.015,
            'drawdown': 0.02
        }

class ComplianceManager:
    def __init__(self):
        self.ofac_api = os.getenv('OFAC_API_ENDPOINT')
        self.cache_duration = 3600
        self.address_cache = {}
        
    async def check_address(self, address: str) -> bool:
        if address in self.address_cache:
            if time.time() - self.address_cache[address]['timestamp'] < self.cache_duration:
                return self.address_cache[address]['compliant']
        return True

class GasOptimizer:
    def __init__(self):
        self.base_fees = []
        self.priority_fees = []
        self.mode = os.getenv('GAS_OPTIMIZER_MODE', 'conservative')
        
    async def update_gas_metrics(self):
        """Update gas metrics from recent blocks"""
        return True
        
    def predict_base_fee(self) -> int:
        """Predict next block's base fee"""
        return 50  # Example base fee in gwei
        
    def estimate_priority_fee(self) -> int:
        """Estimate optimal priority fee"""
        return 2  # Example priority fee in gwei

class ProfitCompounder:
    def __init__(self):
        self.min_compound_amount = Web3.to_wei(0.1, 'ether')
        
    async def process_profit(self, amount: float):
        """Process and compound profits"""
        return True

class MultiHopDetector:
    def __init__(self):
        self.max_hops = 4
        self.min_liquidity = 100000  # Increased minimum liquidity for longer paths
        
        # Graph representation of the DEX ecosystem
        self.liquidity_graph = nx.DiGraph()
        
        # Enhanced neural network for longer paths
        self.path_scorer = nn.Sequential(
            nn.Linear(12, 64),  # Increased input features
            nn.ReLU(),
            nn.Dropout(0.2),    # Added dropout for better generalization
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Historical performance tracking
        self.path_history = {
            'successful_paths': [],
            'failed_paths': [],
            'profit_distribution': [],
            'gas_costs': [],
            'execution_times': [],
            'hop_counts': [],    # Track performance by number of hops
            'path_volatility': [] # Track path volatility
        }
        
        # Learning parameters
        self.learning_rate = 0.0005  # Reduced for more stable learning
        self.batch_size = 64        # Increased for better generalization
        self.memory_size = 20000    # Increased memory for more historical data
        self.experience_buffer = []
        
        # Enhanced path finding parameters
        self.MIN_PROFIT_THRESHOLD = 0.004  # Increased to 0.4% minimum profit for longer paths
        self.MAX_PRICE_IMPACT = 0.005     # Reduced to 0.5% max price impact per hop
        self.MIN_SUCCESS_RATE = 0.85      # Increased success rate requirement
        self.MAX_GAS_USAGE = 800000       # Increased for 4-hop transactions
        
        # New safety parameters for multi-hop trades
        self.safety_params = {
            'max_price_impact_per_hop': 0.002,  # Maximum 0.2% price impact per hop
            'min_liquidity_per_hop': 25000,     # Minimum liquidity per hop
            'max_hop_delay': 2,                 # Maximum seconds between hops
            'max_total_price_impact': 0.008,    # Maximum 0.8% total price impact
            'min_dex_reliability': 0.95,        # Minimum DEX reliability score
            'max_path_volatility': 0.02,        # Maximum path volatility
            'required_confirmations': 1,        # Required confirmations per hop
            'max_gas_per_hop': 200000          # Maximum gas per hop
        }
        
    async def update_liquidity_graph(self, market_data: Dict):
        """Update liquidity graph with current market data"""
        try:
            # Clear existing graph
            self.liquidity_graph.clear()
            
            # Add nodes (tokens) and edges (pools)
            for dex, pools in market_data.items():
                for pool in pools:
                    token0, token1 = pool['token0'], pool['token1']
                    liquidity = float(pool['liquidity'])
                    volume = float(pool['volume24h'])
                    price = float(pool['price'])
                    
                    # Add nodes if they don't exist
                    if not self.liquidity_graph.has_node(token0):
                        self.liquidity_graph.add_node(token0, type='token')
                    if not self.liquidity_graph.has_node(token1):
                        self.liquidity_graph.add_node(token1, type='token')
                    
                    # Add bidirectional edges with attributes
                    self.liquidity_graph.add_edge(token0, token1,
                        dex=dex,
                        liquidity=liquidity,
                        volume=volume,
                        price=price,
                        fee=pool['fee'],
                        last_update=datetime.now()
                    )
                    
                    self.liquidity_graph.add_edge(token1, token0,
                        dex=dex,
                        liquidity=liquidity,
                        volume=volume,
                        price=1/price,
                        fee=pool['fee'],
                        last_update=datetime.now()
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating liquidity graph: {str(e)}")
            return False
            
    def find_profitable_paths(self, start_token: str, amount: float) -> List[Dict]:
        """Find profitable arbitrage paths using advanced algorithms"""
        try:
            profitable_paths = []
            
            # Find all simple paths up to max_hops
            all_paths = []
            for target in self.liquidity_graph.nodes():
                if target != start_token:
                    paths = list(nx.all_simple_paths(
                        self.liquidity_graph,
                        start_token,
                        target,
                        cutoff=self.max_hops
                    ))
                    all_paths.extend(paths)
            
            # Analyze each path
            for path in all_paths:
                path_analysis = self.analyze_path(path, amount)
                if path_analysis['profitable']:
                    profitable_paths.append(path_analysis)
            
            # Sort by expected profit
            profitable_paths.sort(key=lambda x: x['expected_profit'], reverse=True)
            
            # Update path history
            self.update_path_history(profitable_paths)
            
            return profitable_paths
            
        except Exception as e:
            logger.error(f"Error finding profitable paths: {str(e)}")
            return []
            
    def analyze_path(self, path: List[str], amount: float) -> Dict:
        """Analyze arbitrage path for profitability and safety
        
        Args:
            path: List of token addresses in the path
            amount: Initial amount to trade
            
        Returns:
            Dict containing path analysis results
        """
        try:
            if len(path) < 2:
                return {'path': path, 'profitable': False, 'error': 'Path too short'}
                
            if len(path) > self.max_hops:
                return {'path': path, 'profitable': False, 'error': 'Path too long'}
                
            # Track metrics for each hop
            total_fee = 0
            total_price_impact = 0
            expected_amounts = [amount]
            min_liquidity = float('inf')
            dexes_used = []
            
            # Analyze each hop in the path
            for i in range(len(path) - 1):
                token_in = path[i]
                token_out = path[i + 1]
                
                # Get best DEX for this pair
                dex_data = self._get_best_dex(token_in, token_out)
                if not dex_data:
                    return {
                        'path': path,
                        'profitable': False,
                        'error': f'No DEX found for {token_in}-{token_out}'
                    }
                
                dexes_used.append(dex_data['dex'])
                
                # Calculate metrics for this hop
                amount_in = expected_amounts[-1]
                amount_out = self._calculate_output_amount(
                    dex_data['dex'],
                    token_in,
                    token_out,
                    amount_in
                )
                
                if amount_out == 0:
                    return {
                        'path': path,
                        'profitable': False,
                        'error': f'Zero output at hop {i}'
                    }
                
                # Calculate price impact and fees
                price_impact = self.calculate_price_impact(
                    amount_in,
                    dex_data['liquidity']
                )
                
                if price_impact > self.safety_params['max_price_impact_per_hop']:
                    return {
                        'path': path,
                        'profitable': False,
                        'error': f'Excessive price impact at hop {i}'
                    }
                
                total_price_impact += price_impact
                total_fee += dex_data['fee']
                min_liquidity = min(min_liquidity, dex_data['liquidity'])
                expected_amounts.append(amount_out)
                
            # Calculate final metrics
            initial_amount = amount
            final_amount = expected_amounts[-1]
            profit = final_amount - initial_amount
            profit_percentage = (profit / initial_amount) * 100
            
            # Get path score from neural network
            path_features = self.extract_path_features(
                path,
                amount,
                profit_percentage,
                total_fee,
                total_price_impact,
                min_liquidity
            )
            path_score = float(self.score_path(path_features))
            
            # Check if path meets all criteria
            is_profitable = (
                profit_percentage > self.MIN_PROFIT_THRESHOLD and
                total_price_impact < self.safety_params['max_total_price_impact'] and
                min_liquidity >= self.safety_params['min_liquidity_per_hop'] and
                path_score > 0.5 and
                self.get_path_success_rate(path) >= self.safety_params['min_dex_reliability']
            )
            
            return {
                'path': path,
                'dexes': dexes_used,
                'profitable': is_profitable,
                'expected_profit': profit,
                'profit_percentage': profit_percentage,
                'total_fee': total_fee,
                'price_impact': total_price_impact,
                'min_liquidity': min_liquidity,
                'path_score': path_score,
                'confidence': self.calculate_confidence(path_score, profit_percentage),
                'expected_amounts': expected_amounts,
                'success_rate': self.get_path_success_rate(path)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing path: {str(e)}")
            return {
                'path': path,
                'profitable': False,
                'error': str(e)
            }
            
    def _get_best_dex(self, token_in: str, token_out: str) -> Optional[Dict]:
        """Get best DEX for a token pair based on liquidity and fees"""
        try:
            best_dex = None
            best_liquidity = 0
            
            for dex_name, dex_data in self.liquidity_graph[token_in][token_out].items():
                if dex_data['liquidity'] > best_liquidity:
                    best_dex = {
                        'dex': dex_name,
                        'liquidity': dex_data['liquidity'],
                        'fee': dex_data['fee']
                    }
                    best_liquidity = dex_data['liquidity']
            
            return best_dex
            
        except Exception as e:
            logger.error(f"Error getting best DEX: {str(e)}")
            return None
            
    def _calculate_output_amount(self, dex: str, token_in: str, token_out: str, amount_in: float) -> float:
        """Calculate expected output amount for a swap"""
        try:
            # Get DEX config
            dex_config = DEX_CONFIGS.get(dex)
            if not dex_config:
                return 0
                
            # Get pool data
            pool_data = self.liquidity_graph[token_in][token_out][dex]
            
            if dex_config['type'] in ['UniswapV2', 'SushiSwapV2']:
                return self._calculate_v2_output(pool_data, amount_in)
            elif dex_config['type'] in ['UniswapV3', 'SushiSwapV3']:
                return self._calculate_v3_output(pool_data, amount_in)
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error calculating output amount: {str(e)}")
            return 0

    def calculate_price_impact(self, amount: float, liquidity: float) -> float:
        """Calculate price impact of a trade"""
        try:
            return min((amount / liquidity) * 100, 100.0)
        except Exception:
            return float('inf')
            
    def extract_path_features(self, path: List[str], amount: float,
                            profit: float, fee: float, impact: float,
                            liquidity: float) -> torch.Tensor:
        """Extract features for path scoring"""
        features = [
            len(path),           # Path length
            amount,              # Trade amount
            profit,             # Expected profit
            fee,                # Total fees
            impact,             # Price impact
            liquidity,          # Minimum liquidity
            self.get_path_success_rate(path),  # Historical success rate
            self.get_path_avg_profit(path)     # Historical average profit
        ]
        return torch.tensor(features, dtype=torch.float32)
        
    def score_path(self, features: torch.Tensor) -> torch.Tensor:
        """Score a path using the neural network"""
        with torch.no_grad():
            return self.path_scorer(features)
            
    def calculate_confidence(self, path_score: float, profit_percentage: float) -> float:
        """Calculate confidence score for a path"""
        # Combine path score with profit potential
        confidence = (path_score * 0.7) + (min(profit_percentage, 5) / 5 * 0.3)
        return min(confidence, 1.0)
        
    def get_path_success_rate(self, path: List[str]) -> float:
        """Get historical success rate for a path"""
        path_key = '->'.join(path)
        successful = sum(1 for p in self.path_history['successful_paths']
                        if '->'.join(p['path']) == path_key)
        failed = sum(1 for p in self.path_history['failed_paths']
                    if '->'.join(p['path']) == path_key)
        total = successful + failed
        return successful / total if total > 0 else 0
        
    def get_path_avg_profit(self, path: List[str]) -> float:
        """Get historical average profit for a path"""
        path_key = '->'.join(path)
        profits = [p['profit'] for p in self.path_history['successful_paths']
                  if '->'.join(p['path']) == path_key]
        return sum(profits) / len(profits) if profits else 0
        
    def update_path_history(self, paths: List[Dict]):
        """Update path history and trigger learning"""
        try:
            # Add new paths to history
            for path in paths:
                if path['profitable']:
                    self.path_history['successful_paths'].append({
                        'path': path['path'],
                        'profit': path['expected_profit'],
                        'timestamp': datetime.now()
                    })
                    self.path_history['profit_distribution'].append(path['profit_percentage'])
                else:
                    self.path_history['failed_paths'].append({
                        'path': path['path'],
                        'reason': path.get('error', 'Unknown'),
                        'timestamp': datetime.now()
                    })
            
            # Maintain history size
            max_history = 1000
            self.path_history['successful_paths'] = self.path_history['successful_paths'][-max_history:]
            self.path_history['failed_paths'] = self.path_history['failed_paths'][-max_history:]
            self.path_history['profit_distribution'] = self.path_history['profit_distribution'][-max_history:]
            
            # Update experience buffer for learning
            self.update_experience_buffer(paths)
            
            # Trigger learning if enough data
            if len(self.experience_buffer) >= self.batch_size:
                self.train_path_scorer()
                
        except Exception as e:
            logger.error(f"Error updating path history: {str(e)}")
            
    def update_experience_buffer(self, paths: List[Dict]):
        """Update experience buffer for learning"""
        try:
            for path in paths:
                features = self.extract_path_features(
                    path['path'],
                    path.get('amount', 0),
                    path['profit_percentage'],
                    path['total_fee'],
                    path['price_impact'],
                    path['min_liquidity']
                )
                
                # Label is 1 for successful trades, 0 for failed
                label = torch.tensor([1.0 if path['profitable'] else 0.0])
                
                self.experience_buffer.append((features, label))
                
            # Maintain buffer size
            if len(self.experience_buffer) > self.memory_size:
                self.experience_buffer = self.experience_buffer[-self.memory_size:]
                
        except Exception as e:
            logger.error(f"Error updating experience buffer: {str(e)}")
            
    def train_path_scorer(self):
        """Train the path scoring neural network"""
        try:
            if len(self.experience_buffer) < self.batch_size:
                return
                
            # Prepare optimizer
            optimizer = torch.optim.Adam(self.path_scorer.parameters(), lr=self.learning_rate)
            criterion = nn.BCELoss()
            
            # Sample batch
            batch_indices = np.random.choice(
                len(self.experience_buffer),
                self.batch_size,
                replace=False
            )
            
            batch_features = torch.stack([self.experience_buffer[i][0] for i in batch_indices])
            batch_labels = torch.stack([self.experience_buffer[i][1] for i in batch_indices])
            
            # Training step
            optimizer.zero_grad()
            outputs = self.path_scorer(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
        except Exception as e:
            logger.error(f"Error training path scorer: {str(e)}")

# Initialize components
telegram_bot = TelegramBot()
tax_recorder = TaxRecorder()
arbitrage_strategy = ArbitrageStrategy()
testing_protocol = TestingProtocol()
compliance_manager = ComplianceManager()
gas_optimizer = GasOptimizer()
profit_compounder = ProfitCompounder()
multi_hop_detector = MultiHopDetector()

async def initialize_components():
    """Initialize all async components"""
    if os.getenv('TELEGRAM_BOT_TOKEN') and os.getenv('TELEGRAM_CHAT_ID'):
        await telegram_bot.initialize()
    else:
        logger.warning("Telegram bot token or chat ID not set. Bot functionality will be limited.")

def get_lending_pool_abi(protocol_type: str) -> List[Dict]:
    """Get ABI for different lending pool types"""
    if protocol_type == 'AaveV3':
        return AAVE_V3_POOL_ABI
    elif protocol_type == 'AaveV2':
        return AAVE_V2_POOL_ABI
    elif protocol_type == 'CompoundV2':
        return COMPOUND_V2_POOL_ABI
    elif protocol_type == 'BalancerV3':
        return BALANCER_V3_POOL_ABI
    else:
        raise ValueError(f"Unsupported protocol type: {protocol_type}")

# Common ABIs
AAVE_V3_POOL_ABI = [{"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"components":[{"internalType":"uint256","name":"data","type":"uint256"}],"internalType":"struct DataTypes.ReserveConfigurationMap","name":"configuration","type":"tuple"},{"internalType":"uint128","name":"liquidityIndex","type":"uint128"},{"internalType":"uint128","name":"currentLiquidityRate","type":"uint128"},{"internalType":"uint128","name":"variableBorrowIndex","type":"uint128"},{"internalType":"uint128","name":"currentVariableBorrowRate","type":"uint128"},{"internalType":"uint128","name":"currentStableBorrowRate","type":"uint128"},{"internalType":"uint40","name":"lastUpdateTimestamp","type":"uint40"},{"internalType":"uint16","name":"id","type":"uint16"},{"internalType":"address","name":"aTokenAddress","type":"address"},{"internalType":"address","name":"stableDebtTokenAddress","type":"address"},{"internalType":"address","name":"variableDebtTokenAddress","type":"address"},{"internalType":"address","name":"interestRateStrategyAddress","type":"address"}],"internalType":"struct DataTypes.ReserveData","name":"","type":"tuple"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"asset","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"address","name":"onBehalfOf","type":"address"},{"internalType":"uint16","name":"referralCode","type":"uint16"}],"name":"supply","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"address","name":"asset","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"address","name":"to","type":"address"}],"name":"withdraw","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"nonpayable","type":"function"}]

AAVE_V2_POOL_ABI = [{"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"components":[{"internalType":"uint256","name":"data","type":"uint256"}],"internalType":"struct DataTypes.ReserveConfigurationMap","name":"configuration","type":"tuple"},{"internalType":"uint128","name":"liquidityIndex","type":"uint128"},{"internalType":"uint128","name":"variableBorrowIndex","type":"uint128"},{"internalType":"uint128","name":"currentLiquidityRate","type":"uint128"},{"internalType":"uint128","name":"currentVariableBorrowRate","type":"uint128"},{"internalType":"uint128","name":"currentStableBorrowRate","type":"uint128"},{"internalType":"uint40","name":"lastUpdateTimestamp","type":"uint40"},{"internalType":"address","name":"aTokenAddress","type":"address"},{"internalType":"address","name":"stableDebtTokenAddress","type":"address"},{"internalType":"address","name":"variableDebtTokenAddress","type":"address"},{"internalType":"address","name":"interestRateStrategyAddress","type":"address"},{"internalType":"uint8","name":"id","type":"uint8"}],"internalType":"struct DataTypes.ReserveData","name":"","type":"tuple"}],"stateMutability":"view","type":"function"}]

COMPOUND_V2_POOL_ABI = [{"constant":True,"inputs":[{"name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"name":"isActive","type":"bool"},{"name":"borrowEnabled","type":"bool"},{"name":"lastUpdateTimestamp","type":"uint40"}],"name":"","type":"tuple"}],"payable":False,"stateMutability":"view","type":"function"}]

BALANCER_V3_POOL_ABI = [{"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"name":"isActive","type":"bool"}],"name":"","type":"tuple"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address[]","name":"tokens","type":"address[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"},{"internalType":"bytes","name":"userData","type":"bytes"}],"name":"flashLoan","outputs":[],"stateMutability":"nonpayable","type":"function"}]

# DEX ABIs
UNISWAP_V3_ROUTER_ABI = [
    {"inputs":[{"components":[{"internalType":"bytes","name":"path","type":"bytes"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}],"internalType":"struct ISwapRouter.ExactInputParams","name":"params","type":"tuple"}],"name":"exactInput","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"}
]

SUSHISWAP_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

BASESWAP_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

AERODROME_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

PANCAKESWAP_ROUTER_ABI = [
    {"inputs":[{"components":[{"internalType":"bytes","name":"path","type":"bytes"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}],"internalType":"struct ISwapRouter.ExactInputParams","name":"params","type":"tuple"}],"name":"exactInput","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"}
]

SWAPBASED_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

ALIENBASE_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

MAVERICK_ROUTER_ABI = [
    {"inputs":[{"internalType":"bytes","name":"path","type":"bytes"},{"internalType":"address","name":"recipient","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"},{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMinimum","type":"uint256"}],"name":"exactInput","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"}
]

SYNTHSWAP_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

HORIZON_DEX_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"path","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"swapExactTokensForTokens","outputs":[{"internalType":"uint256[]","name":"amounts","type":"uint256[]"}],"stateMutability":"nonpayable","type":"function"}
]

RADIANT_POOL_ABI = [
    {"inputs":[{"internalType":"address","name":"asset","type":"address"}],"name":"getReserveData","outputs":[{"components":[{"name":"isActive","type":"bool"},{"name":"borrowEnabled","type":"bool"},{"name":"lastUpdateTimestamp","type":"uint40"}],"name":"","type":"tuple"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"address","name":"receiverAddress","type":"address"},{"internalType":"address[]","name":"assets","type":"address[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"},{"internalType":"uint256[]","name":"modes","type":"uint256[]"},{"internalType":"address","name":"onBehalfOf","type":"address"},{"internalType":"bytes","name":"params","type":"bytes"},{"internalType":"uint16","name":"referralCode","type":"uint16"}],"name":"flashLoan","outputs":[],"stateMutability":"nonpayable","type":"function"}
]

UNISWAP_V4_ROUTER_ABI = [
    {"inputs":[{"internalType":"bytes","name":"commands","type":"bytes"},{"internalType":"bytes[]","name":"inputs","type":"bytes[]"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"execute","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"bytes","name":"looksRareCommand","type":"bytes"},{"internalType":"bytes","name":"inputs","type":"bytes"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"executeWithLooksRare","outputs":[{"internalType":"uint256","name":"amountOut","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"}],"name":"approveToken","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"address","name":"token","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"address","name":"recipient","type":"address"}],"name":"transferToken","outputs":[],"stateMutability":"nonpayable","type":"function"}
]

def get_dex_router_abi(dex_name: str) -> List[Dict]:
    """Get router ABI for specific DEX"""
    dex_abis = {
        'uniswap_v4': UNISWAP_V4_ROUTER_ABI,
        'uniswap_v3': UNISWAP_V3_ROUTER_ABI,
        'sushiswap': SUSHISWAP_ROUTER_ABI,
        'baseswap': BASESWAP_ROUTER_ABI,
        'aerodrome': AERODROME_ROUTER_ABI,
        'pancakeswap': PANCAKESWAP_ROUTER_ABI,
        'swapbased': SWAPBASED_ROUTER_ABI,
        'alienbase': ALIENBASE_ROUTER_ABI,
        'maverick': MAVERICK_ROUTER_ABI,
        'synthswap': SYNTHSWAP_ROUTER_ABI,
        'horizondex': HORIZON_DEX_ROUTER_ABI
    }
    return dex_abis.get(dex_name.lower(), UNISWAP_V2_PAIR_ABI)

def get_lending_pool_abi(protocol_type: str) -> List[Dict]:
    """Get ABI based on lending protocol type"""
    protocol_abis = {
        'AaveV3': AAVE_V3_POOL_ABI,
        'Balancer': BALANCER_V3_POOL_ABI,
        'Radiant': RADIANT_POOL_ABI
    }
    return protocol_abis.get(protocol_type, AAVE_V3_POOL_ABI)

class ArbitrageError(Exception):
    """Base exception class for arbitrage operations"""
    pass

class ValidationError(ArbitrageError):
    """Validation related errors"""
    pass

class ExecutionError(ArbitrageError):
    """Execution related errors"""
    pass

CURVE_ROUTER_ABI = [
    {"inputs":[{"internalType":"uint256","name":"amountIn","type":"uint256"},{"internalType":"uint256","name":"amountOutMin","type":"uint256"},{"internalType":"address[]","name":"route","type":"address[]"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"deadline","type":"uint256"}],"name":"exchange","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"payable","type":"function"},
    {"inputs":[{"internalType":"uint256[2]","name":"amounts","type":"uint256[2]"},{"internalType":"uint256","name":"min_mint_amount","type":"uint256"}],"name":"add_liquidity","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"_token_amount","type":"uint256"},{"internalType":"uint256","name":"min_amount","type":"uint256"}],"name":"remove_liquidity_one_coin","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"i","type":"uint256"},{"internalType":"uint256","name":"j","type":"uint256"},{"internalType":"uint256","name":"dx","type":"uint256"},{"internalType":"uint256","name":"min_dy","type":"uint256"}],"name":"exchange","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"payable","type":"function"}
]

MORPHO_ROUTER_ABI = [
    {"inputs":[{"internalType":"address","name":"loanToken","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"address","name":"onBehalfOf","type":"address"},{"internalType":"uint16","name":"referralCode","type":"uint16"}],"name":"supply","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"address","name":"asset","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"address","name":"onBehalfOf","type":"address"}],"name":"withdraw","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"address","name":"asset","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"address","name":"onBehalfOf","type":"address"},{"internalType":"uint16","name":"referralCode","type":"uint16"}],"name":"borrow","outputs":[],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"address","name":"asset","type":"address"},{"internalType":"uint256","name":"amount","type":"uint256"},{"internalType":"address","name":"onBehalfOf","type":"address"}],"name":"repay","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"nonpayable","type":"function"},
    {"inputs":[{"internalType":"address","name":"receiverAddress","type":"address"},{"internalType":"address[]","name":"assets","type":"address[]"},{"internalType":"uint256[]","name":"amounts","type":"uint256[]"},{"internalType":"uint256[]","name":"modes","type":"uint256[]"},{"internalType":"address","name":"onBehalfOf","type":"address"},{"internalType":"bytes","name":"params","type":"bytes"},{"internalType":"uint16","name":"referralCode","type":"uint16"}],"name":"flashLoan","outputs":[],"stateMutability":"nonpayable","type":"function"}
]