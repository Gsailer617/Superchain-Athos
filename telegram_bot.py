import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import matplotlib.pyplot as plt
import io
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import pandas as pd
import seaborn as sns
from matplotlib.dates import DateFormatter
import torch
import torch.nn as nn
from telegram.error import TelegramError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TelegramBot:
    def __init__(self):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.admin_id = os.getenv('TELEGRAM_ADMIN_ID')
        self.bot = None
        self.initialized = False
        self.application = None
        self.bot_active = False
        self.auto_trade = False
        self.system_paused = False
        
        # Verify credentials are set
        if not self.token:
            logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
        if not self.chat_id:
            logger.error("TELEGRAM_CHAT_ID environment variable not set")
        if not self.admin_id:
            logger.error("TELEGRAM_ADMIN_ID environment variable not set")
            
        # Convert chat_id to integer if present
        try:
            if self.chat_id:
                self.chat_id = int(self.chat_id)
        except ValueError:
            logger.error("TELEGRAM_CHAT_ID must be a valid integer")
            
        # Convert admin_id to integer if present
        try:
            if self.admin_id:
                self.admin_id = int(self.admin_id)
        except ValueError:
            logger.error("TELEGRAM_ADMIN_ID must be a valid integer")
        
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
        
        # Performance tracking
        self.performance = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0,
            'total_gas_spent': 0,
            'best_trade': None,
            'worst_trade': None
        }
        
        # Chart settings
        self.CHART_SETTINGS = {
            'default_timeframe': '24h',
            'available_timeframes': ['1h', '24h', '7d', '30d'],
            'max_data_points': 1000,
            'chart_style': 'dark_background',
            'color_palette': 'husl'
        }
        
        # Analytics tracking
        self.analytics = {
            'hourly_profits': [],
            'token_performance': {},
            'dex_performance': {},
            'gas_history': [],
            'trade_sizes': [],
            'execution_times': [],
            'slippage_history': [],
            'opportunity_history': []
        }

    async def initialize(self):
        """Initialize the Telegram bot"""
        try:
            if not self.token or not self.chat_id:
                logger.warning("Telegram bot token or chat ID not set")
                return False
                
            self.bot = Bot(token=self.token)
            # Test the connection
            await self.bot.get_me()
            self.initialized = True
            logger.info("Telegram bot initialized successfully")
            
            # Initialize bot application
            self.application = Application.builder().token(self.token).build()
            
            # Add handlers
            self.application.add_handler(CommandHandler("start", self.start_command))
            self.application.add_handler(CommandHandler("help", self.help_command))
            self.application.add_handler(CommandHandler("status", self.status_command))
            self.application.add_handler(CommandHandler("performance", self.cmd_performance))
            self.application.add_handler(CommandHandler("charts", self.cmd_charts))
            self.application.add_handler(CommandHandler("pause", self.cmd_pause))
            self.application.add_handler(CommandHandler("resume", self.cmd_resume))
            self.application.add_handler(CommandHandler("settings", self.cmd_settings))
            self.application.add_handler(CallbackQueryHandler(self.handle_callback))
            
            # Start the bot
            await self.application.initialize()
            await self.application.start()
            
            # Start polling in the background
            asyncio.create_task(self.application.updater.start_polling())
            
            self.bot_active = True
            
            # Send initialization message
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text="🟢 Arbitrage Bot initialized and ready for trading!"
                )
            except Exception as e:
                logger.warning(f"Could not send initialization message: {str(e)}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {str(e)}")
            return False

    async def stop(self):
        """Stop the Telegram bot"""
        try:
            if self.bot:
                # Close the bot's session
                if hasattr(self.bot, 'close'):
                    await self.bot.close()
                self.initialized = False
                logger.info("Telegram bot stopped")
                
        except Exception as e:
            logger.error(f"Error stopping Telegram bot: {str(e)}")

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command"""
        await update.message.reply_text(
            "Welcome to the Base Chain Arbitrage Bot!\n"
            "Use /help to see available commands."
        )
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /help command"""
        help_text = (
            "Available commands:\n"
            "/start - Start the bot\n"
            "/help - Show this help message\n"
            "/status - Show current bot status"
        )
        await update.message.reply_text(help_text)
        
    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /status command"""
        status_text = (
            f"Bot Status: {'Active' if self.bot_active else 'Inactive'}\n"
            f"Chat ID: {self.chat_id}\n"
            "System Status: Running"
        )
        await update.message.reply_text(status_text)

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command"""
        if not self.bot_active:
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
Tokens: {' → '.join(best_trade['tokens']) if best_trade else 'No trades yet'}
Profit: ${best_trade['profit']:.2f if best_trade else 0}

*Worst Trade*
Tokens: {' → '.join(worst_trade['tokens']) if worst_trade else 'No trades yet'}
Profit: ${worst_trade['profit']:.2f if worst_trade else 0}
"""
        await update.message.reply_text(performance, parse_mode='Markdown')

    async def cmd_charts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /charts command"""
        if not self.bot_active:
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
        
        await update.message.reply_text(
            "📊 *Performance Charts*\nSelect a chart to view:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        if not self.bot_active:
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
        
        await update.message.reply_text(
            "⚙️ *Settings*\n"
            f"Current Mode: {'🤖 Auto' if self.auto_trade else '👨 Manual'}\n"
            "Select an option:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command"""
        if not self.bot_active:
            return
            
        if self.system_paused:
            await update.message.reply_text("⚠️ System is already paused!")
            return
            
        self.system_paused = True
        await update.message.reply_text(
            "🔴 Arbitrage system paused\n"
            "All monitoring and trading operations have been suspended.\n"
            "Use /resume to restart operations."
        )

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command"""
        if not self.bot_active:
            return
            
        if not self.system_paused:
            await update.message.reply_text("⚠️ System is already running!")
            return
            
        self.system_paused = False
        await update.message.reply_text(
            "🟢 Arbitrage system resumed\n"
            "Now monitoring for opportunities."
        )

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline keyboards"""
        if not self.bot_active:
            return
            
        query = update.callback_query
        await query.answer()
        
        try:
            if query.data.startswith('execute_'):
                opp_id = query.data.split('_')[1]
                await query.edit_message_text(
                    f"✅ Executing trade {opp_id}...",
                    parse_mode='Markdown'
                )
                
            elif query.data.startswith('skip_'):
                opp_id = query.data.split('_')[1]
                await query.edit_message_text(
                    f"❌ Skipped trade {opp_id}",
                    parse_mode='Markdown'
                )
                
            elif query.data == 'auto':
                self.auto_trade = True
                await query.edit_message_text(
                    "🤖 Switched to Auto Mode\n"
                    "Bot will automatically execute profitable trades.",
                    parse_mode='Markdown'
                )
                
            elif query.data == 'manual':
                self.auto_trade = False
                await query.edit_message_text(
                    "👨 Switched to Manual Mode\n"
                    "Bot will ask for confirmation before executing trades.",
                    parse_mode='Markdown'
                )
                
            elif query.data == 'reset_stats':
                self.reset_performance()
                await query.edit_message_text(
                    "🔄 Performance statistics have been reset.",
                    parse_mode='Markdown'
                )
                
            elif query.data.startswith('chart_'):
                chart_type = query.data.split('_')[1]
                await self.send_chart(update, chart_type)
                
        except Exception as e:
            logger.error(f"Error handling callback: {str(e)}")
            await query.edit_message_text(
                f"❌ Error: {str(e)}",
                parse_mode='Markdown'
            )

    def reset_performance(self):
        """Reset performance statistics"""
        self.performance = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0,
            'total_gas_spent': 0,
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
            'opportunity_history': []
        }

    async def send_chart(self, update: Update, chart_type: str):
        """Generate and send performance charts"""
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

    async def notify_opportunity(self, message: str):
        """Send notification about arbitrage opportunity"""
        try:
            if not self.initialized or not self.bot:
                logger.warning("Telegram bot not initialized")
                return False
                
            # Format message for better readability
            if isinstance(message, dict):
                formatted_message = (
                    "🔍 New Arbitrage Opportunity\n\n"
                    f"Type: {message.get('type', 'Unknown')}\n"
                    f"Expected Profit: ${message.get('expected_profit', 0):.2f}\n"
                    f"Confidence: {message.get('confidence', 0)*100:.1f}%\n"
                    f"Risk Score: {message.get('risk_score', 0)*100:.1f}%\n"
                    f"Gas Cost: ${message.get('gas_cost', 0):.2f}\n"
                    f"Net Profit: ${message.get('net_profit', 0):.2f}\n"
                    f"Tokens: {' → '.join(message.get('tokens_involved', []))}"
                )
            else:
                formatted_message = str(message)
            
            # Verify chat ID is valid
            try:
                chat_id = int(self.chat_id)
            except (ValueError, TypeError):
                logger.error(f"Invalid chat ID: {self.chat_id}")
                return False
            
            # Send message with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=formatted_message,
                        parse_mode='HTML'
                    )
                    return True
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Retry {attempt + 1}/{max_retries} failed: {str(e)}")
                    await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {str(e)}")
            return False

    async def notify_error(self, error: str):
        """Send error notification"""
        try:
            if not self.initialized or not self.bot:
                logger.warning("Telegram bot not initialized")
                return False
                
            message = f"⚠️ Error: {error}"
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to send error notification: {str(e)}")
            return False
            
    async def notify_trade_result(self, trade_result: dict):
        """Send notification about trade result"""
        try:
            if not self.initialized or not self.bot:
                logger.warning("Telegram bot not initialized")
                return False
                
            profit = trade_result.get('profit', 0)
            tx_hash = trade_result.get('tx_hash', 'Unknown')
            
            message = (
                f"🔄 Trade Completed\n"
                f"Profit: {profit:.6f} ETH\n"
                f"Transaction: {tx_hash}\n"
            )
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to send trade result notification: {str(e)}")
            return False

# Initialize Telegram bot
telegram_bot = TelegramBot() 