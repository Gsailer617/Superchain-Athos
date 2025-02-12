import os
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, cast, TypeVar, Callable, Awaitable, TypedDict, Sequence
from datetime import datetime, timedelta
import asyncio
import matplotlib.pyplot as plt
import io
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot, Message, CallbackQuery
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.error import TelegramError
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
import time
import traceback
from functools import lru_cache
import structlog
import re

from utils.performance_tracking import PerformanceTracker
from utils.visualization_utils import VisualizationUtils
from config.environment import env_config
from utils.resource_manager import resource_manager
from utils.llm_interface import LLMInterface

# Configure logging
logger = structlog.get_logger(__name__)

# Type definitions
class TradeData(TypedDict):
    profit: float
    gas_used: float
    tokens: List[str]
    tokens_involved: List[str]
    timestamp: datetime
    tx_hash: str
    execution_time: float

class PerformanceMetrics(TypedDict):
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_profit: float
    total_gas_spent: float
    best_trade: Optional[TradeData]
    worst_trade: Optional[TradeData]

class TokenAnalytics(TypedDict):
    trades: int
    profit: float
    volume: float

class DexAnalytics(TypedDict):
    volume: float
    trades: int
    success_rate: float

class Analytics(TypedDict):
    hourly_profits: List[float]
    token_performance: Dict[str, TokenAnalytics]
    dex_performance: Dict[str, DexAnalytics]
    gas_history: List[float]
    trade_sizes: List[float]
    execution_times: List[float]
    slippage_history: List[float]
    opportunity_history: List[Dict[str, Any]]
    error_history: List[Dict[str, Any]]

class NotificationData(TypedDict):
    message: str
    chat_ids: Sequence[int]

class ChartConfig:
    """Configuration for chart appearance and behavior"""
    def __init__(self, height: int = 400):
        self.height = height

class RateLimiter:
    def __init__(self, max_calls_per_minute: int):
        self.max_calls_per_minute = max_calls_per_minute
        self.calls = []
        
    async def wait(self):
        now = time.time()
        self.calls = [call for call in self.calls if call > now - 60]
        
        if len(self.calls) >= self.max_calls_per_minute:
            sleep_time = self.calls[0] - (now - 60)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
        self.calls.append(now)

class TelegramBot:
    def __init__(self, is_test: bool = False):
        """Initialize Telegram bot with configuration"""
        # Get configuration from environment
        self.bot_token = env_config.TELEGRAM_BOT_TOKEN
        self.admin_ids_str = env_config.ADMIN_IDS
        self.chat_id = env_config.CHAT_ID
        
        self.initialized = False
        self.bot_active = False
        self.application: Optional[Application] = None
        self.bot: Optional[Bot] = None
        self.is_test = is_test
        self.system_paused = False
        self.auto_trade = True
        self.agent = None
        
        # Initialize command_handlers list
        self.command_handlers = []
        
        # Initialize performance tracker
        self.tracker = PerformanceTracker()
        
        # Initialize visualization utils
        VisualizationUtils.setup_style()
        
        # Convert admin_ids to list of integers
        try:
            self.admin_ids = [int(id.strip()) for id in self.admin_ids_str.split(',') if id.strip()]
        except (ValueError, AttributeError):
            self.admin_ids = []
            if not self.is_test:
                logger.error("Failed to parse TELEGRAM_ADMIN_IDS - must be comma-separated integers")
        
        # Convert chat_id to integer
        try:
            self.chat_id = int(self.chat_id) if self.chat_id else None
        except ValueError:
            self.chat_id = None
            if not self.is_test:
                logger.error("Invalid TELEGRAM_CHAT_ID - must be an integer")
        
        # Verify credentials are set
        if not self.is_test:
            if not self.bot_token:
                logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
            if not self.admin_ids:
                logger.error("TELEGRAM_ADMIN_IDS environment variable not set or invalid")
            if not self.chat_id:
                logger.error("TELEGRAM_CHAT_ID environment variable not set or invalid")
        
        # Initialize performance metrics
        self.performance: PerformanceMetrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'total_gas_spent': 0.0,
            'best_trade': None,
            'worst_trade': None
        }
        
        # Initialize analytics
        self.analytics: Analytics = {
            'hourly_profits': [],
            'token_performance': {},
            'dex_performance': {},
            'gas_history': [],
            'trade_sizes': [],
            'execution_times': [],
            'slippage_history': [],
            'opportunity_history': [],
            'error_history': []
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'min_profit': 0.01,  # Minimum profit in ETH
            'max_gas': 0.05,     # Maximum gas cost in ETH
            'max_slippage': 0.02, # Maximum slippage percentage
            'min_net_profit_usd': 50,  # Minimum net profit in USD
            'profit_margin_threshold': 1.5  # Minimum profit/gas ratio
        }
        
        # Chart settings
        self.CHART_SETTINGS = {
            'default_timeframe': '24h',
            'available_timeframes': ['1h', '24h', '7d', '30d'],
            'max_data_points': 1000,
            'chart_style': 'dark_background',
            'color_palette': 'husl'
        }
        
        # Alert thresholds for notifications
        self.alert_thresholds = {
            'min_profit': 0.002,  # 0.2% minimum profit for alerts
            'max_gas': 500,       # Maximum gas price in GWEI
            'min_liquidity': 50000,  # Minimum liquidity in USD
            'price_change': 0.05,    # 5% price change alert
            'volume_spike': 2.0,     # 2x volume increase alert
            'max_slippage': 0.01,    # 1% maximum slippage
            'min_net_profit_usd': 1.0,  # Minimum net profit in USD after gas
            'profit_margin_threshold': 0.05,  # 5% minimum profit margin over gas costs
            'max_opportunity_age': 5,  # Seconds
            'max_position_size': 0.2,  # % of capital
            'daily_loss_limit': -5.0  # %
        }

        # Add synchronization primitives
        self._update_lock = asyncio.Lock()
        self._notification_queue = asyncio.Queue()
        self._last_notification_time = {}
        self.rate_limiter = RateLimiter(max_calls_per_minute=30)

        # Add command handlers for training suggestions
        self.command_handlers.extend([
            CommandHandler('suggest', self.handle_suggestion),
            CommandHandler('parameters', self.show_current_parameters)
        ])

        # Add command handlers for token management
        self.command_handlers.extend([
            CommandHandler('addtoken', self.handle_add_token),
            CommandHandler('removetoken', self.handle_remove_token),
            CommandHandler('listtokens', self.handle_list_tokens),
            CommandHandler('tokeninfo', self.handle_token_info)
        ])

        # Initialize LLM interface
        self.llm = LLMInterface()

    async def initialize(self):
        """Initialize the Telegram bot with improved error handling"""
        try:
            if self.is_test:
                self.initialized = True
                self.bot_active = True
                logger.info("Telegram bot initialized in test mode")
                return True
                
            if not self.bot_token or not self.admin_ids:
                logger.warning("Telegram bot token or admin IDs not set")
                return False
                
            # Test connection with retry
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    self.bot = Bot(token=self.bot_token)
                    await self.bot.get_me()
                    break
                except TelegramError as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to initialize Telegram bot after {max_retries} attempts: {e}")
                        return False
                    await asyncio.sleep(retry_delay * (2 ** attempt))
            
            # Initialize application with error handling
            try:
                self.application = Application.builder().token(self.bot_token).build()
                
                # Set up handlers only after application is initialized
                await self._setup_handlers()
                
                await self.application.initialize()
                await self.application.start()
                
                # Start notification processor
                resource_manager.create_task(
                    self._process_notification_queue(),
                    name='notification_processor'
                )
                
                # Start polling with automatic reconnection
                resource_manager.create_task(
                    self._maintain_polling(),
                    name='polling_maintainer'
                )
                
                self.initialized = True
                self.bot_active = True
                
                await self._send_startup_notification()
                return True
                
            except Exception as e:
                logger.error(f"Error initializing application: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            return False
            
    async def _maintain_polling(self):
        """Maintain polling connection with automatic reconnection"""
        while True:
            try:
                if self.application and self.application.updater and not self.application.updater.running:
                    await self.application.updater.start_polling()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Polling error: {e}")
                await asyncio.sleep(5)  # Back off on error
                
    async def _process_notification_queue(self):
        """Process notifications with rate limiting and error handling"""
        while True:
            try:
                notification: NotificationData = await self._notification_queue.get()
                await self.rate_limiter.wait()
                
                message = notification['message']
                chat_ids = notification['chat_ids']
                
                if not self.bot:
                    continue
                    
                # Cast bot to Bot type to satisfy type checker
                bot = cast(Bot, self.bot)
                
                for chat_id in chat_ids:
                    try:
                        # Avoid duplicate notifications
                        last_time = self._last_notification_time.get(chat_id, 0)
                        if time.time() - last_time < 1:  # 1 second minimum interval
                            continue
                            
                        await bot.send_message(
                            chat_id=chat_id,
                            text=message,
                            parse_mode='HTML'
                        )
                        self._last_notification_time[chat_id] = time.time()
                        
                    except TelegramError as e:
                        logger.error("Error sending message", 
                            error=str(e),
                            chat_id=chat_id
                        )
                        
            except Exception as e:
                logger.error("Error processing notification", error=str(e))
                await asyncio.sleep(1)
                
    async def notify_opportunity(self, opportunity: Dict) -> bool:
        """Enhanced opportunity notification with LLM analysis and rate limiting"""
        if not self.initialized:
            logger.warning("Telegram bot not initialized")
            return False
            
        if self.is_test:
            logger.info(f"Test mode: Would notify opportunity: {opportunity}")
            return True
            
        try:
            # Get LLM analysis of the opportunity
            analysis = await self.llm.analyze_opportunity(opportunity)
            
            formatted_message = (
                "🔍 *New Arbitrage Opportunity*\n\n"
                f"Type: {opportunity.get('type', 'Unknown')}\n"
                f"Expected Profit: ${opportunity.get('expected_profit', 0):.2f}\n"
                f"Confidence: {opportunity.get('confidence', 0)*100:.1f}%\n"
                f"Risk Score: {opportunity.get('risk_score', 0)*100:.1f}%\n"
                f"Gas Cost: ${opportunity.get('gas_cost', 0):.2f}\n"
                f"Net Profit: ${opportunity.get('net_profit', 0):.2f}\n"
                f"Tokens: {' → '.join(opportunity.get('tokens_involved', []))}\n\n"
                f"*Analysis:*\n{analysis}"
            )
            
            # Queue notification for all recipients
            chat_ids = set(id for id in [self.chat_id] + self.admin_ids if id is not None)
            await self._notification_queue.put({
                'message': formatted_message,
                'chat_ids': list(chat_ids)
            })
            
            return True
            
        except Exception as e:
            logger.error("Failed to queue notification", error=str(e))
            return False
            
    async def notify_error(self, error: Union[str, Exception]) -> bool:
        """Enhanced error notification with LLM analysis and proper error handling"""
        if not self.initialized or not self.bot:
            logger.warning("Telegram bot not initialized")
            return False
            
        if self.is_test:
            logger.info(f"Test mode: Would notify error: {error}")
            return True
            
        try:
            # Get error details
            error_data = {
                'error_type': type(error).__name__ if isinstance(error, Exception) else 'Unknown',
                'message': str(error),
                'context': 'Runtime Error',
                'stack_trace': traceback.format_exc() if isinstance(error, Exception) else '',
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to error history
            self.analytics['error_history'].append(error_data)
            
            # Get LLM analysis if available
            analysis = ""
            if hasattr(self, 'llm') and self.llm:
                try:
                    analysis = await self.llm.explain_error(error_data)
                except Exception as llm_error:
                    logger.error("Failed to get LLM analysis", error=str(llm_error))
                    analysis = "Error analysis unavailable"
            
            message = (
                "⚠️ *Error Alert*\n\n"
                f"Error Type: {error_data['error_type']}\n"
                f"Message: {error_data['message']}\n\n"
                f"*Analysis:*\n{analysis}\n\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            # Queue notification for all recipients
            chat_ids = set(id for id in [self.chat_id] + self.admin_ids if id is not None)
            await self._notification_queue.put({
                'message': message,
                'chat_ids': list(chat_ids)
            })
            
            return True
            
        except Exception as notify_error:
            logger.error("Critical error in notify_error", error=str(notify_error))
            return False

    async def notify_admins(self, message: str) -> None:
        """Send message to admin users with rate limiting"""
        if not self.initialized or not self.bot:
            return
            
        try:
            # Queue notification for admins
            chat_ids = [admin_id for admin_id in self.admin_ids if admin_id is not None]
            await self._notification_queue.put({
                'message': message,
                'chat_ids': chat_ids
            })
            
        except Exception as e:
            logger.error("Error notifying admins", error=str(e))

    async def update_performance_metrics(self, trade: TradeData) -> None:
        """Thread-safe performance update with improved error handling"""
        async with self._update_lock:
            try:
                # Update basic metrics
                self.performance['total_trades'] += 1
                profit = trade['profit']
                gas_used = trade['gas_used']
                net_profit = profit - gas_used
                
                if net_profit > 0:
                    self.performance['successful_trades'] += 1
                else:
                    self.performance['failed_trades'] += 1
                    
                self.performance['total_profit'] += profit
                self.performance['total_gas_spent'] += gas_used
                
                # Update trade records
                self._update_trade_records(trade, net_profit)
                
                # Update analytics
                self._update_analytics(trade)
                
                # Log the update
                self._log_performance_update(trade)
                
                # Notify if significant
                if self._is_significant_update(trade):
                    await self._notify_performance_update(trade)
                    
            except Exception as e:
                logger.error("Error updating performance metrics", 
                    error=str(e),
                    trade_id=trade.get('tx_hash', 'unknown'),
                    stack_trace=traceback.format_exc()
                )
                await self.notify_error(e)

    def _update_trade_records(self, trade: TradeData, net_profit: float) -> None:
        """Update best/worst trade records with validation"""
        try:
            if not self._is_valid_trade(trade):
                logger.warning("Invalid trade data received", trade_id=trade.get('tx_hash', 'unknown'))
                return
                
            # Update best trade
            if not self.performance['best_trade'] or net_profit > (self.performance['best_trade']['profit'] if self.performance['best_trade'] else float('-inf')):
                self.performance['best_trade'] = trade
                
            # Update worst trade
            if not self.performance['worst_trade'] or net_profit < (self.performance['worst_trade']['profit'] if self.performance['worst_trade'] else float('inf')):
                self.performance['worst_trade'] = trade
                
        except Exception as e:
            logger.error("Error updating trade records", error=str(e))

    def _update_analytics(self, trade: TradeData) -> None:
        """Update analytics with improved error handling"""
        try:
            # Update execution times
            if 'execution_time' in trade:
                self.analytics['execution_times'].append(trade['execution_time'])
                
            # Update trade sizes
            net_profit = trade['profit'] - trade['gas_used']
            self.analytics['trade_sizes'].append(net_profit)
            
            # Update token performance
            token_pair = '/'.join(trade['tokens_involved'])
            if token_pair not in self.analytics['token_performance']:
                self.analytics['token_performance'][token_pair] = {
                    'trades': 0,
                    'profit': 0.0,
                    'volume': 0.0
                }
                
            token_perf = self.analytics['token_performance'][token_pair]
            token_perf['trades'] += 1
            token_perf['profit'] += net_profit
            token_perf['volume'] += abs(net_profit)
            
            # Trim old data periodically
            self._trim_old_data()
            
        except Exception as e:
            logger.error("Error updating analytics", 
                error=str(e),
                trade_id=trade.get('tx_hash', 'unknown')
            )

    def _trim_old_data(self, max_age: timedelta = timedelta(hours=24)) -> None:
        """Remove old data to prevent memory bloat"""
        try:
            cutoff = datetime.now() - max_age
            
            for key in ['profits', 'gas_prices', 'opportunities', 'network_status']:
                if key in self.data:
                    self.data[key] = [
                        item for item in self.data[key]
                        if item['timestamp'] > cutoff
                    ]
                    
        except Exception as e:
            logger.error("Error trimming old data", error=str(e))

    def _is_valid_trade(self, trade: TradeData) -> bool:
        """Validate trade data structure"""
        required_fields = {'profit', 'gas_used', 'tokens_involved', 'timestamp', 'tx_hash'}
        return (
            isinstance(trade, dict) and
            all(field in trade for field in required_fields) and
            isinstance(trade['tokens_involved'], list) and
            len(trade['tokens_involved']) >= 2 and
            isinstance(trade['profit'], (int, float)) and
            isinstance(trade['gas_used'], (int, float))

    def _is_significant_update(self, trade: TradeData) -> bool:
        """Check if update is significant enough to notify"""
        try:
            profit = trade['profit']
            gas_used = trade['gas_used']
            net_profit = profit - gas_used
            
            return (
                net_profit > self.alert_thresholds['min_profit'] or
                gas_used > self.alert_thresholds['max_gas'] or
                self.performance['total_trades'] % 10 == 0  # Every 10 trades
            )
        except Exception as e:
            logger.error("Error checking update significance", error=str(e))
            return False

    async def _notify_performance_update(self, trade: TradeData) -> None:
        """Notify significant performance updates with rate limiting"""
        try:
            message = self._format_performance_update(trade)
            
            # Queue notification for all recipients with rate limiting
            chat_ids = set(id for id in [self.chat_id] + self.admin_ids if id is not None)
            await self._notification_queue.put({
                'message': message,
                'chat_ids': list(chat_ids)
            })
            
        except Exception as e:
            logger.error("Error notifying performance update", error=str(e))

    async def stop(self):
        """Stop the Telegram bot"""
        try:
            if self.application:
                # First stop the updater if it's running
                if self.application.updater and self.application.updater.running:
                    await self.application.updater.stop()
                
                # Then stop and shutdown the application
                await self.application.stop()
                await self.application.shutdown()
                
            if self.bot:
                # Close the bot's session
                if hasattr(self.bot, 'close'):
                    await self.bot.close()
                    
            self.initialized = False
            self.bot_active = False
            logger.info("Telegram bot stopped successfully")
                
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {str(e)}")

    async def _setup_handlers(self):
        """Set up command handlers for the bot with improved error handling"""
        try:
            if not self.application:
                logger.error("Cannot set up handlers: application not initialized")
                return
                
            handlers = [
            # Basic commands
                CommandHandler('startbot', self.startbot_command),
                CommandHandler('help', self.help_command),
                CommandHandler('status', self.status_command),
            
            # Performance commands
                CommandHandler('performance', self.performance_command),
                CommandHandler('summary', self.summary_command),
                CommandHandler('stats', self.performance_command),  # Alias for performance
                CommandHandler('profit', self.performance_command),  # Alias for performance
                
                # Analysis commands
                CommandHandler('gas', self.performance_command),  # Show gas section
                CommandHandler('tokens', self.performance_command),  # Show token section
                CommandHandler('dexes', self.performance_command),  # Show DEX section
                CommandHandler('charts', self.cmd_charts),
            
            # Control commands
                CommandHandler('pause', self.pause_command),
                CommandHandler('resume', self.resume_command),
                CommandHandler('auto', self.auto_command),
                CommandHandler('manual', self.manual_command),
                CommandHandler('settings', self.cmd_settings),
                
                # LLM-enhanced commands
                CommandHandler('explain', self.explain_command),
                CommandHandler('analyze', self.analyze_command),
                CommandHandler('errors', self.errors_command),
                
                # Callback handler
                CallbackQueryHandler(self.handle_callback),
                
                # Natural language handler
                MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
            ]
            
            # Add all handlers
            for handler in handlers:
                self.application.add_handler(handler)
                
            # Add error handler
            self.application.add_error_handler(self._handle_callback_error)
            
            logger.info("Command handlers set up successfully")
            
        except Exception as e:
            logger.error(f"Error setting up command handlers: {e}")
            raise
            
    async def startbot_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /startbot command"""
        if not update.effective_message:
            return
        message = update.effective_message
        await message.reply_text(
            "🤖 Welcome to the Superchain Arbitrage Bot!\n\n"
            "I'll help you monitor and manage arbitrage opportunities.\n"
            "Use /help to see available commands."
        )
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /help command"""
        if not update.effective_message:
            return
        message = update.effective_message
        help_text = (
            "Available commands:\n\n"
            "🚀 Start:\n"
            "/startbot - Initialize the bot\n\n"
            "📊 Monitoring:\n"
            "/status - Check current bot status\n"
            "/performance - View performance metrics\n"
            "/summary - Get latest performance summary\n\n"
            "🎮 Control:\n"
            "/pause - Pause operations\n"
            "/resume - Resume operations\n"
            "/auto - Enable auto-trading\n"
            "/manual - Disable auto-trading"
        )
        await message.reply_text(help_text)
        
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /status command"""
        if not update.effective_message:
            return
        message = update.effective_message
        status = "🟢 Active" if not self.system_paused else "🔴 Paused"
        mode = "🤖 Auto" if self.auto_trade else "👨‍💻 Manual"
        await message.reply_text(f"Status: {status}\nMode: {mode}")
        
    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /performance command"""
        if not update.effective_message:
            return
        message = update.effective_message
        await message.reply_text("Performance metrics coming soon!")
        
    async def summary_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /summary command"""
        if not update.effective_message:
            return
        message = update.effective_message
        await message.reply_text("Performance summary coming soon!")
        
    async def pause_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /pause command"""
        if not update.effective_message:
            return
        message = update.effective_message
        self.system_paused = True
        await message.reply_text("🔴 System paused")
        
    async def resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /resume command"""
        if not update.effective_message:
            return
        message = update.effective_message
        self.system_paused = False
        await message.reply_text("🟢 System resumed")
        
    async def auto_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /auto command"""
        if not update.effective_message:
            return
        message = update.effective_message
        self.auto_trade = True
        await message.reply_text("🤖 Auto-trading enabled")
        
    async def manual_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /manual command"""
        if not update.effective_message:
            return
        message = update.effective_message
        self.auto_trade = False
        await message.reply_text("👨‍💻 Manual mode enabled")

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command"""
        if not self.bot_active or not update.effective_message:
            return
            
        best_trade = self.performance['best_trade']
        worst_trade = self.performance['worst_trade']
        
        performance = f"""
📈 *Performance Summary*

*Trade Statistics*
Total Trades: {self.performance['total_trades']}
Successful: {self.performance['successful_trades']}
Failed: {self.performance['failed_trades']}
Success Rate: {self.calculate_success_rate():.1f}%

*Profit/Loss*
Total Profit: ${self.performance['total_profit']:.2f}
Total Gas Spent: ${self.performance['total_gas_spent']:.2f}
Net Profit: ${(self.performance['total_profit'] - self.performance['total_gas_spent']):.2f}

*Best Trade*
Tokens: {(' → '.join(best_trade['tokens']) if best_trade is not None else 'No trades yet')}
Profit: ${(best_trade['profit'] if best_trade is not None else 0):.2f}

*Worst Trade*
Tokens: {(' → '.join(worst_trade['tokens']) if worst_trade is not None else 'No trades yet')}
Profit: ${(worst_trade['profit'] if worst_trade is not None else 0):.2f}
"""
        await update.effective_message.reply_text(performance, parse_mode='Markdown')

    async def cmd_charts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /charts command"""
        if not self.bot_active or not update.effective_message:
            return
            
        keyboard = [
            [
                InlineKeyboardButton("Profit Chart 📈", callback_data='chart_profit'),
                InlineKeyboardButton("Volume Chart 📊", callback_data='chart_volume')
            ],
            [
                InlineKeyboardButton("Gas Analysis ⛽", callback_data='chart_gas'),
                InlineKeyboardButton("Performance 📊", callback_data='chart_performance')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.effective_message.reply_text(
            "📊 *Performance Charts*\nSelect a chart to view:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        if not self.bot_active or not update.effective_message:
            return
            
        keyboard = [
            [
                InlineKeyboardButton("Auto Mode 🤖", callback_data='auto'),
                InlineKeyboardButton("Manual Mode 👨", callback_data='manual')
            ],
            [
                InlineKeyboardButton("Edit Thresholds 🎯", callback_data='edit_thresholds'),
                InlineKeyboardButton("Reset Stats 🔄", callback_data='reset_stats')
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.effective_message.reply_text(
            "⚙️ *Settings*\n"
            f"Current Mode: {'🤖 Auto' if self.auto_trade else '👨 Manual'}\n"
            "Select an option:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle callback queries with improved error handling"""
        if not update.callback_query:
            return
            
        query = update.callback_query
        await query.answer()
        
        try:
            if not query.data:
                return
                
            # Execute trade callbacks
            if query.data.startswith('execute_'):
                opp_id = query.data.split('_')[1]
                await self._handle_trade_execution(query, opp_id)
                
            # Skip trade callbacks
            elif query.data.startswith('skip_'):
                opp_id = query.data.split('_')[1]
                await self._handle_trade_skip(query, opp_id)
                
            # Mode switching callbacks
            elif query.data in ['auto', 'manual']:
                await self._handle_mode_switch(query)
                
            # Settings callbacks
            elif query.data == 'reset_stats':
                await self._handle_stats_reset(query)
                
            # Chart callbacks
            elif query.data.startswith('chart_'):
                await self._handle_chart_request(update, query)
                
        except Exception as e:
            await self._handle_callback_error(update, e)

    async def _handle_trade_execution(self, query: CallbackQuery, opp_id: str) -> None:
        """Handle trade execution callback"""
        try:
            await query.edit_message_text(
                f"✅ Executing trade {opp_id}...",
                parse_mode='Markdown'
            )
            # Additional trade execution logic here
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            await query.edit_message_text(
                f"❌ Failed to execute trade: {str(e)}",
                parse_mode='Markdown'
            )

    async def _handle_trade_skip(self, query: CallbackQuery, opp_id: str) -> None:
        """Handle trade skip callback"""
        try:
            await query.edit_message_text(
                f"❌ Skipped trade {opp_id}",
                parse_mode='Markdown'
            )
            # Additional skip logic here
                
        except Exception as skip_error:
            logger.error(f"Error skipping trade: {skip_error}")
            await query.edit_message_text(
                "❌ Failed to skip trade",
                parse_mode='Markdown'
            )
                
    async def _handle_mode_switch(self, query: CallbackQuery) -> None:
        """Handle mode switching callback"""
        try:
            self.auto_trade = query.data == 'auto'
            mode_text = "🤖 Auto" if self.auto_trade else "👨‍💻 Manual"
            
            await query.edit_message_text(
                f"Switched to {mode_text} Mode\n"
                f"{'Bot will automatically execute profitable trades.' if self.auto_trade else 'Bot will ask for confirmation before executing trades.'}",
                parse_mode='Markdown'
            )
                
        except Exception as mode_error:
            logger.error(f"Error switching mode: {mode_error}")
            await query.edit_message_text(
                "❌ Failed to switch mode",
                parse_mode='Markdown'
            )

    async def _handle_stats_reset(self, query: CallbackQuery) -> None:
        """Handle statistics reset callback"""
        try:
            self.reset_performance()
            await query.edit_message_text(
                "🔄 Performance statistics have been reset.",
                parse_mode='Markdown'
            )
                
        except Exception as e:
            logger.error(f"Error resetting stats: {e}")
            await query.edit_message_text(
                "❌ Failed to reset statistics",
                parse_mode='Markdown'
            )

    async def _handle_chart_request(self, update: Update, query: CallbackQuery) -> None:
        """Handle chart request callback"""
        try:
            chart_type = query.data.split('_')[1]
            await self.send_chart(update, chart_type)
                
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            if query.message:
                await query.edit_message_text(
                    "❌ Failed to generate chart",
                    parse_mode='Markdown'
                )

    def reset_performance(self):
        """Reset performance statistics"""
        self.performance = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'total_gas_spent': 0.0,
            'best_trade': None,
            'worst_trade': None
        }
        self.analytics = {
            'hourly_profits': [],
            'token_performance': {},
            'dex_performance': {},
            'gas_history': [],
            'trade_sizes': [],
            'execution_times': [],
            'slippage_history': [],
            'opportunity_history': [],
            'error_history': []
        }

    async def send_chart(self, update: Update, chart_type: str):
        """Generate and send performance charts"""
        if not update.effective_message:
            return
            
        try:
            plt.style.use(self.CHART_SETTINGS['chart_style'])
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == 'profit':
                self.plot_profit_chart(ax)
            elif chart_type == 'volume':
                self.plot_volume_chart(ax)
            elif chart_type == 'gas':
                self.plot_gas_chart(ax)
            elif chart_type == 'performance':
                self.plot_performance_chart(ax)
                
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            await update.effective_message.reply_photo(
                photo=buf,
                caption=f"📊 {chart_type.title()} Chart"
            )
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            if update.effective_message:
                await update.effective_message.reply_text(
                    f"❌ Error generating chart: {str(e)}"
                )

    def plot_profit_chart(self, ax):
        """Plot profit over time chart"""
        if not self.analytics['hourly_profits']:
            ax.text(0.5, 0.5, 'No profit data available', ha='center', va='center')
            return
            
        df = pd.DataFrame(self.analytics['hourly_profits'])
        df.plot(ax=ax, kind='line', marker='o')
        ax.set_title('Hourly Profit')
        ax.set_xlabel('Time')
        ax.set_ylabel('Profit (USD)')
        ax.grid(True)

    def plot_volume_chart(self, ax):
        """Plot trading volume chart"""
        if not self.analytics['trade_sizes']:
            ax.text(0.5, 0.5, 'No volume data available', ha='center', va='center')
            return
            
        df = pd.DataFrame(self.analytics['trade_sizes'])
        df.plot(ax=ax, kind='bar')
        ax.set_title('Trading Volume')
        ax.set_xlabel('Trade')
        ax.set_ylabel('Volume (USD)')
        ax.grid(True)

    def plot_gas_chart(self, ax):
        """Plot gas price analysis"""
        if not self.analytics['gas_history']:
            ax.text(0.5, 0.5, 'No gas data available', ha='center', va='center')
            return
            
        df = pd.DataFrame(self.analytics['gas_history'])
        df.plot(ax=ax, kind='line', marker='o')
        ax.set_title('Gas Price History')
        ax.set_xlabel('Time')
        ax.set_ylabel('Gas Price (GWEI)')
        ax.grid(True)

    def plot_performance_chart(self, ax):
        """Plot overall performance metrics"""
        if self.performance['total_trades'] == 0:
            ax.text(0.5, 0.5, 'No performance data available', ha='center', va='center')
            return
            
        metrics = [
            ('Success Rate', self.calculate_success_rate()),
            ('Total Profit', self.performance['total_profit']),
            ('Gas Spent', self.performance['total_gas_spent']),
            ('Net Profit', self.performance['total_profit'] - self.performance['total_gas_spent'])
        ]
        
        x = range(len(metrics))
        heights = [m[1] for m in metrics]
        labels = [m[0] for m in metrics]
        
        ax.bar(x, heights)
        ax.set_title('Overall Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(True)

    def calculate_success_rate(self) -> float:
        """Calculate success rate of trades"""
        if self.performance['total_trades'] == 0:
            return 0.0
        return (self.performance['successful_trades'] / self.performance['total_trades']) * 100

    async def _handle_callback_error(self, update: Update, error: Exception) -> None:
        """Handle errors in callback queries"""
        try:
            if not update.callback_query:
                return
                
            query = update.callback_query
            error_message = (
                "❌ Error processing request\n"
                f"Type: {type(error).__name__}\n"
                "Please try again or contact support."
            )
            
            await query.answer(
                text="An error occurred",
                show_alert=True
            )
            
            if query.message:
                await query.edit_message_text(
                    text=error_message,
                    parse_mode='Markdown'
                )
            
            # Log the error
            logger.error(
                "Callback error",
                error_type=type(error).__name__,
                error_message=str(error),
                update_id=update.update_id,
                user_id=update.effective_user.id if update.effective_user else None
            )
            
            # Notify admins
            await self.notify_error(error)
            
        except Exception as e:
            logger.error(f"Error in error handler: {e}")

    def _log_performance_update(self, trade: TradeData) -> None:
        """Log performance update with structured logging"""
        try:
            logger.info(
                "Performance update",
                total_trades=self.performance['total_trades'],
                successful_trades=self.performance['successful_trades'],
                failed_trades=self.performance['failed_trades'],
                total_profit=self.performance['total_profit'],
                total_gas_spent=self.performance['total_gas_spent'],
                latest_trade_profit=trade['profit'],
                latest_trade_gas=trade['gas_used']
            )
        except Exception as e:
            logger.error(f"Error logging performance update: {e}")

    async def explain_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /explain command - Get LLM explanation of latest opportunity"""
        if not self.bot_active or not update.effective_message:
            return
            
        try:
            # Get latest opportunity data from analytics
            latest_opp = self.analytics.get('opportunity_history', [])[-1] if self.analytics.get('opportunity_history') else None
            if not latest_opp:
                await update.effective_message.reply_text("No recent opportunities to explain.")
                return
                
            # Get LLM analysis
            analysis = await self.llm.analyze_opportunity(latest_opp)
            
            message = (
                "🔍 *Opportunity Analysis*\n\n"
                f"{analysis}\n\n"
                f"_Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
            )
            
            await update.effective_message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in explain command: {e}")
            if update.effective_message:
                await update.effective_message.reply_text("Error generating explanation.")

    async def analyze_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /analyze command - Get LLM analysis of performance metrics"""
        if not self.bot_active or not update.effective_message:
            return
            
        try:
            # Get performance data
            performance_data = self.tracker.get_performance_summary()
            
            # Get LLM analysis
            analysis = await self.llm.analyze_performance_metrics(performance_data)
            
            message = (
                "📊 *Performance Analysis*\n\n"
                f"{analysis}\n\n"
                f"_Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
            )
            
            await update.effective_message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in analyze command: {e}")
            if update.effective_message:
                await update.effective_message.reply_text("Error generating analysis.")

    async def errors_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /errors command - Get LLM analysis of recent errors"""
        if not self.bot_active or not update.effective_message:
            return
            
        try:
            # Get recent errors from analytics
            recent_errors = self.analytics.get('error_history', [])[-5:]  # Last 5 errors
            if not recent_errors:
                await update.effective_message.reply_text("No recent errors to analyze.")
                return
                
            # Get LLM analysis for each error
            analyses = []
            for error in recent_errors:
                analysis = await self.llm.explain_error(error)
                analyses.append(f"Error: {error['message']}\nAnalysis: {analysis}\n")
                
            message = (
                "⚠️ *Recent Error Analysis*\n\n" +
                "\n".join(analyses) + "\n" +
                f"_Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
            )
            
            await update.effective_message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in errors command: {e}")
            if update.effective_message:
                await update.effective_message.reply_text("Error analyzing errors.")

    async def notify_trade_execution(self, trade: Dict) -> bool:
        """Send trade execution notification"""
        if not self.initialized or not self.bot:
            logger.warning("Telegram bot not initialized")
            return False
            
        if self.is_test:
            # In test mode, just log the trade execution
            logger.info(f"Test mode: Would notify trade execution: {trade}")
            return True
            
        try:
            profit = trade.get('profit', 0)
            tx_hash = trade.get('tx_hash', 'Unknown')
            
            message = (
                f"🔄 Trade Completed\n"
                f"Profit: {profit:.6f} ETH\n"
                f"Transaction: {tx_hash}\n"
            )
            
            bot = self.bot  # Local variable to avoid multiple attribute access
            success = True
            
            # Send to main chat
            if self.chat_id is not None:
                try:
                    await bot.send_message(
                        chat_id=self.chat_id,
                        text=message,
                        parse_mode='HTML'
                    )
                except Exception as e:
                    logger.error(f"Failed to send to main chat: {e}")
                    success = False
            
            # Also notify admins if they're different from chat_id
            for admin_id in self.admin_ids:
                if admin_id != self.chat_id:
                    try:
                        await bot.send_message(
                            chat_id=admin_id,
                            text=message,
                            parse_mode='HTML'
                        )
                    except Exception as e:
                        logger.error(f"Failed to send to admin {admin_id}: {e}")
                        success = False
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send trade result notification: {str(e)}")
            return False

    async def close(self):
        """Close the Telegram bot"""
        try:
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
            self.initialized = False
            self.bot_active = False
            logger.info("Telegram bot closed")
        except Exception as e:
            logger.error(f"Error closing Telegram bot: {str(e)}")
            
    def is_trade_profitable(self, opportunity: Dict) -> bool:
        """Check if a trade opportunity is profitable"""
        if not opportunity:
            return False
            
        # Get profit metrics
        expected_profit = opportunity.get('expected_profit', 0)
        gas_cost = opportunity.get('gas_cost', 0)
        net_profit = expected_profit - gas_cost
        
        # Check against thresholds
        if net_profit < self.alert_thresholds['min_profit']:
            return False
            
        if gas_cost > self.alert_thresholds['max_gas']:
            return False
            
        if opportunity.get('slippage', 0) > self.alert_thresholds['max_slippage']:
            return False
            
        if net_profit < self.alert_thresholds['min_net_profit_usd']:
            return False
            
        # Calculate profit margin over gas costs
        if gas_cost > 0:
            profit_margin = net_profit / gas_cost
            if profit_margin < self.alert_thresholds['profit_margin_threshold']:
                return False
                
        return True
        
    def update_performance(self, trade: TradeData) -> None:
        """Update performance metrics with trade results"""
        if not trade:
            return
            
        self.performance['total_trades'] += 1
        
        profit = trade['profit']
        gas_used = trade['gas_used']
        net_profit = profit - gas_used
        
        # Update successful/failed trades based on profit
        if net_profit > 0:
            self.performance['successful_trades'] += 1
        else:
            self.performance['failed_trades'] += 1
            
        self.performance['total_profit'] += profit
        self.performance['total_gas_spent'] += gas_used
        
        # Update best trade
        if not self.performance['best_trade'] or net_profit > self.performance['best_trade']['profit']:
            self.performance['best_trade'] = {
                'profit': net_profit,
                'tokens_involved': trade['tokens_involved'],
                'timestamp': trade['timestamp'],
                'tx_hash': trade['tx_hash'],
                'gas_used': trade['gas_used'],
                'execution_time': trade['execution_time']
            }
            
        # Update worst trade
        if not self.performance['worst_trade'] or net_profit < self.performance['worst_trade']['profit']:
            self.performance['worst_trade'] = {
                'profit': net_profit,
                'tokens_involved': trade['tokens_involved'],
                'timestamp': trade['timestamp'],
                'tx_hash': trade['tx_hash'],
                'gas_used': trade['gas_used'],
                'execution_time': trade['execution_time']
            }
            
        # Update analytics
        self.analytics['execution_times'].append(trade['execution_time'])
        self.analytics['trade_sizes'].append(net_profit)
        
        # Update token performance
        token_pair = '/'.join(trade['tokens_involved'])
        if token_pair not in self.analytics['token_performance']:
            self.analytics['token_performance'][token_pair] = {
                'trades': 0,
                'profit': 0.0,
                'volume': 0.0
            }
        self.analytics['token_performance'][token_pair]['trades'] += 1
        self.analytics['token_performance'][token_pair]['profit'] += net_profit
        self.analytics['token_performance'][token_pair]['volume'] += abs(net_profit)

    @lru_cache(maxsize=1000)
    def _get_token_performance(self, token_pair: str) -> TokenAnalytics:
        """Get cached token performance metrics"""
        if token_pair not in self.analytics['token_performance']:
            return {'trades': 0, 'profit': 0.0, 'volume': 0.0}
        return self.analytics['token_performance'][token_pair]

    @lru_cache(maxsize=1000)
    def _get_dex_performance(self, dex: str) -> DexAnalytics:
        """Get cached DEX performance metrics"""
        if dex not in self.analytics['dex_performance']:
            return {'volume': 0.0, 'trades': 0, 'success_rate': 0.0}
        return self.analytics['dex_performance'][dex]

    def _format_performance_update(self, trade: TradeData) -> str:
        """Format performance update message with proper typing"""
        return (
            "📊 Performance Update\n\n"
            f"Total Trades: {self.performance['total_trades']}\n"
            f"Success Rate: {self.calculate_success_rate():.1f}%\n"
            f"Total Profit: ${self.performance['total_profit']:.2f}\n"
            f"Gas Spent: ${self.performance['total_gas_spent']:.2f}\n"
            f"Latest Trade Profit: ${trade['profit']:.2f}"
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle non-command messages with natural language processing"""
        if not update.effective_message or not self.bot_active:
            return
            
        try:
            # Get user message and ID
            message = update.effective_message
            user_id = message.from_user.id if message.from_user else None
            text = message.text.lower()
            
            if not user_id:
                return

            # Check if user is admin
            if user_id not in self.admin_ids:
                await message.reply_text("Sorry, only admins can suggest tokens.")
                return

            # Check for token suggestions in natural language
            token_patterns = [
                r'add token (?:address )?(?:is )?([0x][a-fA-F0-9]{40})',
                r'track (?:token )?(?:address )?([0x][a-fA-F0-9]{40})',
                r'monitor (?:token )?(?:address )?([0x][a-fA-F0-9]{40})',
                r'suggest (?:token )?(?:address )?([0x][a-fA-F0-9]{40})',
                r'check (?:token )?(?:address )?([0x][a-fA-F0-9]{40})'
            ]

            for pattern in token_patterns:
                match = re.search(pattern, text)
                if match:
                    token_address = match.group(1)
                    # Process token suggestion
                    await message.reply_text(f"I'll check token {token_address} for you...")
                    
                    if self.agent and self.agent.token_discovery:
                        # Validate token first
                        is_valid = await self.agent.token_discovery.validate_token(token_address)
                        if not is_valid:
                            await message.reply_text(
                                f"❌ Token {token_address} failed validation checks.\n"
                                "The token must meet minimum requirements for liquidity, "
                                "security, and holder distribution."
                            )
                            return

                        # Add to discovered tokens
                        self.agent.token_discovery.discovered_tokens.add(token_address)
                        
                        # Store in Redis for persistence
                        await self.agent.token_discovery.redis.sadd("discovered_tokens", token_address)
                        
                        # Get token info for detailed response
                        validation_result = await self.agent.token_discovery.get_cached_validation(token_address)
                        
                        # Format detailed response
                        response = (
                            f"✅ Token {token_address} added successfully!\n\n"
                            "Token Analysis:\n"
                        )
                        
                        if validation_result:
                            response += (
                                f"• Security Score: {validation_result.get('security_score', 0):.2f}\n"
                                f"• Social Sentiment: {validation_result.get('social_sentiment', {}).get('score', 0):.2f}\n"
                            )
                            
                            # Add metadata if available
                            metadata = validation_result.get('metadata', {})
                            if metadata:
                                response += "\nAdditional Information:\n"
                                if 'liquidity' in metadata:
                                    response += f"• Liquidity: ${float(metadata['liquidity']):,.2f}\n"
                                if 'holder_data' in metadata:
                                    holder_data = metadata['holder_data']
                                    response += f"• Total Holders: {holder_data.get('total_holders', 0):,}\n"
                        
                        await message.reply_text(response)
                        return
                    else:
                        await message.reply_text("❌ Token discovery system not initialized.")
                        return

            # If no token suggestion found, process as regular message
            response = await self.llm.process_query(text, user_id)
            await message.reply_text(response, parse_mode='Markdown')
            
        except Exception as msg_error:
            logger.error("Error handling message", error=str(msg_error))
            if update.effective_message:
                await update.effective_message.reply_text(
                    "I apologize, but I encountered an error processing your message. Please try again."
                )

    async def _send_startup_notification(self):
        """Send a notification when the bot starts up"""
        try:
            startup_message = (
                "🚀 Superchain Arbitrage Bot Started\n\n"
                "System is now monitoring for opportunities.\n"
                "Use /help to see available commands."
            )
            
            if self.chat_id and self.bot:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=startup_message
                )
                logger.info("Startup notification sent successfully")
            else:
                logger.warning("No chat_id set for startup notification")
                
        except Exception as e:
            logger.error(f"Error sending startup notification: {e}")
            # Don't raise the exception - startup notification is not critical

    async def send_performance_update(self):
        """Send performance update with charts"""
        try:
            if not self.application or not self.chat_id or not self.bot:
                logger.warning("Cannot send performance update: required components not initialized")
                return

            # Get performance summary
            summary = self.tracker.get_performance_summary()
            
            # Create performance chart
            chart_buf = VisualizationUtils.create_performance_summary(
                self.tracker.performance
            )
            
            # Format message
            message = (
                "📊 Performance Update\n\n"
                f"Total Trades: {summary['total_trades']}\n"
                f"Success Rate: {summary['success_rate']:.2f}%\n"
                f"Total Profit: {summary['total_profit']:.4f} ETH\n"
                f"Gas Spent: {summary['total_gas_spent']:.4f} ETH\n"
                f"Net Profit: {summary['net_profit']:.4f} ETH"
            )
            
            # Send message and chart using concrete Bot instance
            bot = self.bot
            if isinstance(bot, Bot):
                await bot.send_message(
                    chat_id=self.chat_id,
                    text=message
                )
                await bot.send_photo(
                    chat_id=self.chat_id,
                    photo=chart_buf
                )
            else:
                logger.error("Bot instance is not properly initialized")
            
        except Exception as e:
            logger.error(f"Error sending performance update: {str(e)}")
            
    async def send_token_analytics(self):
        """Send token analytics chart"""
        try:
            chart_buf = VisualizationUtils.create_token_analytics_static_chart(
                self.tracker.token_analytics
            )
            
            await self.application.bot.send_photo(
                chat_id=self.chat_id,
                photo=chart_buf,
                caption="📈 Token Performance Analytics"
            )
            
        except Exception as e:
            logger.error(f"Error sending token analytics: {str(e)}")
            
    async def send_dex_analytics(self):
        """Send DEX analytics chart"""
        try:
            chart_buf = VisualizationUtils.create_dex_analytics_chart(
                self.tracker.dex_analytics
            )
            
            await self.application.bot.send_photo(
                chat_id=self.chat_id,
                photo=chart_buf,
                caption="🔄 DEX Performance Analytics"
            )
            
        except Exception as e:
            logger.error(f"Error sending DEX analytics: {str(e)}")
            
    def update_metrics(self, metrics: Dict):
        """Update performance metrics"""
        self.tracker.update_metrics(metrics)
        
    def record_trade(self, trade_data: Dict):
        """Record trade performance"""
        self.tracker.record_trade(trade_data)
        
    def update_token_analytics(self, token: str, data: Dict):
        """Update token analytics"""
        self.tracker.update_token_analytics(token, data)
        
    def update_dex_analytics(self, dex: str, data: Dict):
        """Update DEX analytics"""
        self.tracker.update_dex_analytics(dex, data)

    async def handle_suggestion(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /suggest command - Get LLM suggestions for optimization"""
        if not update.effective_message:
            return
            
        try:
            suggestions = await self.llm.get_optimization_suggestions(self.performance)
            await update.effective_message.reply_text(suggestions, parse_mode='Markdown')
        except Exception as suggest_error:
            logger.error("Error handling suggestion", error=str(suggest_error))
            await update.effective_message.reply_text("Error generating suggestions")

    async def show_current_parameters(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /parameters command - Show current system parameters"""
        if not update.effective_message:
            return
            
        try:
            params = (
                "🔧 *Current Parameters*\n\n"
                f"Min Profit: {self.alert_thresholds['min_profit']} ETH\n"
                f"Max Gas: {self.alert_thresholds['max_gas']} GWEI\n"
                f"Max Slippage: {self.alert_thresholds['max_slippage']*100}%\n"
                f"Min Net Profit: ${self.alert_thresholds['min_net_profit_usd']}\n"
                f"Profit Margin Threshold: {self.alert_thresholds['profit_margin_threshold']}"
            )
            await update.effective_message.reply_text(params, parse_mode='Markdown')
        except Exception as e:
            logger.error("Error showing parameters", error=str(e))
            await update.effective_message.reply_text("Error retrieving parameters")

    async def handle_add_token(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /addtoken command to add a new token for discovery"""
        if not await self._verify_admin(update):
            return

        try:
            # Get token address from command
            if not context.args or len(context.args) < 1:
                await update.message.reply_text(
                    "Please provide a token address.\n"
                    "Usage: /addtoken <token_address>"
                )
                return

            token_address = context.args[0]
            
            # Validate and add token
            if self.agent and self.agent.token_discovery:
                # Validate token first
                is_valid = await self.agent.token_discovery.validate_token(token_address)
                if not is_valid:
                    await update.message.reply_text(f"❌ Token {token_address} failed validation checks.")
                    return

                # Add to discovered tokens
                self.agent.token_discovery.discovered_tokens.add(token_address)
                
                # Store in Redis for persistence
                await self.agent.token_discovery.redis.sadd("discovered_tokens", token_address)
                
                await update.message.reply_text(f"✅ Token {token_address} added successfully!")
            else:
                await update.message.reply_text("❌ Token discovery system not initialized.")

        except Exception as e:
            logger.error(f"Error adding token: {str(e)}")
            await update.message.reply_text("❌ Error adding token. Please check the address and try again.")

    async def handle_remove_token(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /removetoken command to remove a token from discovery"""
        if not await self._verify_admin(update):
            return

        try:
            if not context.args or len(context.args) < 1:
                await update.message.reply_text(
                    "Please provide a token address.\n"
                    "Usage: /removetoken <token_address>"
                )
                return

            token_address = context.args[0]
            
            if self.agent and self.agent.token_discovery:
                # Remove from both memory and Redis
                self.agent.token_discovery.discovered_tokens.discard(token_address)
                await self.agent.token_discovery.redis.srem("discovered_tokens", token_address)
                
                await update.message.reply_text(f"✅ Token {token_address} removed successfully!")
            else:
                await update.message.reply_text("❌ Token discovery system not initialized.")

        except Exception as e:
            logger.error(f"Error removing token: {str(e)}")
            await update.message.reply_text("❌ Error removing token.")

    async def handle_list_tokens(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /listtokens command to list all discovered tokens"""
        if not await self._verify_admin(update):
            return

        try:
            if self.agent and self.agent.token_discovery:
                # Get tokens from both memory and Redis
                memory_tokens = self.agent.token_discovery.discovered_tokens
                redis_tokens = await self.agent.token_discovery.redis.smembers("discovered_tokens")
                
                # Combine and deduplicate
                all_tokens = set(memory_tokens) | set(redis_tokens)
                
                if not all_tokens:
                    await update.message.reply_text("No tokens currently being tracked.")
                    return
                
                # Format the response
                response = "📝 Discovered Tokens:\n\n"
                for token in sorted(all_tokens):
                    response += f"• {token}\n"
                
                await update.message.reply_text(response)
            else:
                await update.message.reply_text("❌ Token discovery system not initialized.")

        except Exception as e:
            logger.error(f"Error listing tokens: {str(e)}")
            await update.message.reply_text("❌ Error retrieving token list.")

    async def handle_token_info(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the /tokeninfo command to get detailed information about a token"""
        if not await self._verify_admin(update):
            return

        try:
            if not context.args or len(context.args) < 1:
                await update.message.reply_text(
                    "Please provide a token address.\n"
                    "Usage: /tokeninfo <token_address>"
                )
                return

            token_address = context.args[0]
            
            if self.agent and self.agent.token_discovery:
                # Get token validation and analytics
                validation_result = await self.agent.token_discovery.get_cached_validation(token_address)
                if not validation_result:
                    await update.message.reply_text(f"❌ No information available for token {token_address}")
                    return
                
                # Format the response
                response = f"📊 Token Information for {token_address}:\n\n"
                response += f"Security Score: {validation_result.get('security_score', 0):.2f}\n"
                response += f"Social Sentiment: {validation_result.get('social_sentiment', {}).get('score', 0):.2f}\n"
                
                # Add metadata if available
                metadata = validation_result.get('metadata', {})
                if metadata:
                    response += "\nMetadata:\n"
                    for key, value in metadata.items():
                        response += f"• {key}: {value}\n"
                
                await update.message.reply_text(response)
            else:
                await update.message.reply_text("❌ Token discovery system not initialized.")

        except Exception as e:
            logger.error(f"Error getting token info: {str(e)}")
            await update.message.reply_text("❌ Error retrieving token information.")

# Initialize Telegram bot
telegram_bot = TelegramBot() 