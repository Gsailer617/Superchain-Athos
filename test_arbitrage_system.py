import unittest
import asyncio
import os
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import torch
import pandas as pd
from web3 import Web3
from SuperchainArbitrageAgent import SuperchainArbitrageAgent
from telegram_bot import TelegramBot
import matplotlib.pyplot as plt
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestArbitrageSystem(unittest.IsolatedAsyncioTestCase):
    """Comprehensive test suite for the arbitrage system"""
    
    async def asyncSetUp(self):
        """Set up test environment"""
        # Set up test environment variables
        os.environ['TELEGRAM_BOT_TOKEN'] = 'test_token'
        os.environ['ADMIN_IDS'] = '123456,789012'
        
        # Initialize components
        self.web3_provider = 'http://localhost:8545'  # Test provider
        self.telegram_bot = TelegramBot()
        self.agent = SuperchainArbitrageAgent(self.web3_provider)
        
        # Test market data
        self.test_market_data = {
            'price_impact': 0.001,
            'liquidity': 1000000,
            'volatility': 0.02,
            'gas_price': 30,  # GWEI
            'block_time': 12,
            'price_difference': 0.005,
            'volume_24h': 500000,
            'tvl': 2000000,
            'token_score': 0.85,
            'market_sentiment': 0.7,
            'network_congestion': 0.4,
            'slippage': 0.002
        }
        
        # Test token pairs for Base Sepolia
        self.test_pairs = [
            ('0x4200000000000000000000000000000000000006',  # WETH
             '0xd9aAEc86B65D86f6A7B5B1b0c42FFA531710b6CA'), # USDbC
            ('0x2Ae3F1Ec7F1F5012CFEab0185bfc7aa3cf0DEc22',  # cbETH
             '0x4200000000000000000000000000000000000006')  # WETH
        ]

    async def asyncTearDown(self):
        """Clean up after tests"""
        plt.close('all')
        if hasattr(self, 'agent'):
            await self.agent.close()
        if hasattr(self, 'telegram_bot'):
            await self.telegram_bot.close()

    async def test_1_system_initialization(self):
        """Test system initialization and configuration"""
        logger.info("Testing system initialization...")
        
        # Check agent initialization
        self.assertIsNotNone(self.agent.web3)
        self.assertIsNotNone(self.agent.feature_extractor)
        self.assertIsNotNone(self.agent.prediction_head)
        self.assertIsNotNone(self.agent.dex_attention)
        self.assertIsNotNone(self.agent.defillama_client)
        
        # Check Telegram bot initialization
        self.assertIsNotNone(self.telegram_bot.bot_token)
        self.assertTrue(len(self.telegram_bot.admin_ids) > 0)
        
        # Verify DEX configurations
        self.assertTrue(len(self.agent.supported_dexes) > 0)
        for dex_name, dex_config in self.agent.supported_dexes.items():
            self.assertIn('address', dex_config)
            self.assertIn('type', dex_config)
            
        logger.info("✅ System initialization tests passed")

    async def test_2_neural_network(self):
        """Test the neural network's prediction capabilities"""
        logger.info("Testing neural network predictions...")
        
        # Test data
        test_features = torch.tensor([
            0.5,  # price_difference
            1000.0,  # volume
            0.8,  # liquidity
            50.0,  # gas_price
            0.01,  # slippage
            0.95,  # confidence
            0.1,  # volatility
            0.7,  # market_depth
            0.6,  # correlation
            0.9,  # momentum
            0.4,  # resistance
            0.3   # support
        ], dtype=torch.float32)
        
        # Get predictions
        predictions, dex_weights = self.agent.model(test_features)
        
        # Basic sanity checks
        self.assertEqual(len(predictions), 3)  # profit, confidence, risk
        self.assertEqual(len(dex_weights), 10)  # 10 supported DEXes
        
        # Value range checks
        self.assertTrue(torch.all(dex_weights >= 0) and torch.all(dex_weights <= 1))
        self.assertAlmostEqual(float(torch.sum(dex_weights)), 1.0, places=5)
        
        logger.info("✅ Neural network tests passed")

    async def test_3_opportunity_analysis(self):
        """Test opportunity analysis and validation"""
        logger.info("Testing opportunity analysis...")
        
        # Test flash loan opportunity analysis
        flash_opportunity = await self.agent.analyze_flash_loan_opportunity(
            token_pair=self.test_pairs[0],
            amount=1000.0,
            market_data=self.test_market_data
        )
        
        # Validate flash loan opportunity structure
        self.assertIsNotNone(flash_opportunity)
        self.assertIn('flash_loan_provider', flash_opportunity)
        self.assertIn('flash_loan_fee', flash_opportunity)
        self.assertIn('gross_profit', flash_opportunity)
        self.assertIn('net_profit', flash_opportunity)
        
        # Test flash loan safety validation
        is_safe = self.agent.validate_flash_loan_safety({
            'amount': 1000.0,
            'expected_profit': 10.0,
            'gas_cost': 1.0,
            'net_profit_usd': 100.0
        })
        self.assertTrue(is_safe)
        
        # Test regular opportunity analysis
        regular_opportunity = await self.agent.analyze_opportunity(
            token_pair=self.test_pairs[0],
            amount=1000.0,
            market_data=self.test_market_data
        )
        
        # Validate regular opportunity structure
        self.assertIn('id', regular_opportunity)
        self.assertIn('profit_prediction', regular_opportunity)
        self.assertIn('confidence', regular_opportunity)
        self.assertIn('risk_score', regular_opportunity)
        self.assertIn('metrics', regular_opportunity)
        
        # Validate metrics
        metrics = regular_opportunity['metrics']
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('sortino_ratio', metrics)
        self.assertIn('execution_probability', metrics)
        
        # Test profitability check
        is_profitable = self.telegram_bot.is_trade_profitable(regular_opportunity)
        self.assertIsInstance(is_profitable, bool)
        
        logger.info("✅ Opportunity analysis tests passed")

    async def test_4_telegram_notifications(self):
        """Test Telegram bot notifications and controls"""
        logger.info("Testing Telegram notifications...")
        
        # Test flash loan opportunity notification
        test_flash_opportunity = {
            'id': 'test_flash_1',
            'type': 'Flash Loan Arbitrage',
            'tokens_involved': ['WETH', 'USDbC'],
            'expected_profit': 5.0,
            'flash_loan_fee': 0.5,
            'gas_cost': 0.3,
            'net_profit': 4.2,
            'confidence': 0.95,
            'risk_score': 0.1,
            'price_impact': 0.001,
            'slippage': 0.002,
            'liquidity': 1000000,
            'volume_24h': 500000,
            'volatility': 0.02,
            'flash_loan_provider': 'aave'
        }
        
        await self.telegram_bot.notify_opportunity(test_flash_opportunity)
        
        # Test flash loan execution notification
        test_flash_execution = {
            'id': 'test_flash_1',
            'type': 'Flash Loan Arbitrage',
            'status': 'completed',
            'profit': 4.2,
            'gas_used': 0.3,
            'execution_time': 1.0,
            'flash_loan_provider': 'aave',
            'tokens_involved': ['WETH', 'USDbC']
        }
        
        await self.telegram_bot.notify_trade_execution(test_flash_execution)
        
        # Update performance metrics
        self.telegram_bot.update_performance(test_flash_execution)
        
        # Validate performance tracking
        self.assertEqual(self.telegram_bot.performance['total_trades'], 1)
        self.assertEqual(self.telegram_bot.performance['successful_trades'], 1)
        self.assertAlmostEqual(self.telegram_bot.performance['total_profit'], 4.2)
        
        logger.info("✅ Telegram notification tests passed")

    async def test_5_live_monitoring(self):
        """Test live monitoring functionality"""
        logger.info("Testing live monitoring...")
        
        # Start monitoring for a short duration
        monitoring_task = asyncio.create_task(self._run_monitoring_test())
        
        # Wait for monitoring
        await asyncio.sleep(30)  # Monitor for 30 seconds
        
        # Stop monitoring
        monitoring_task.cancel()
        
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass
        
        logger.info("✅ Live monitoring test completed")

    async def _run_monitoring_test(self):
        """Helper method to run monitoring"""
        try:
            await self.agent.monitor_superchain()
        except asyncio.CancelledError:
            logger.info("Monitoring stopped")
            raise

    async def test_6_performance_analysis(self):
        """Test performance analysis and visualization"""
        logger.info("Testing performance analysis...")
        
        # Add some test trades
        test_trades = [
            {'profit': 1.0, 'gas_used': 0.2, 'tokens_involved': ['WETH', 'USDbC']},
            {'profit': 0.5, 'gas_used': 0.3, 'tokens_involved': ['WETH', 'USDT']},
            {'profit': -0.2, 'gas_used': 0.1, 'tokens_involved': ['WBTC', 'USDbC']}
        ]
        
        for trade in test_trades:
            self.telegram_bot.update_performance(trade)
        
        # Verify performance metrics
        self.assertEqual(self.telegram_bot.performance['total_trades'], 3)
        self.assertEqual(self.telegram_bot.performance['successful_trades'], 2)
        self.assertEqual(self.telegram_bot.performance['failed_trades'], 1)
        self.assertAlmostEqual(self.telegram_bot.performance['total_profit'], 1.3)
        self.assertAlmostEqual(self.telegram_bot.performance['total_gas_spent'], 0.6)
        
        logger.info("✅ Performance analysis tests passed")

if __name__ == '__main__':
    unittest.main() 