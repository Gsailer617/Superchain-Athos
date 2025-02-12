import pytest
import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from src.core.error_recovery import error_recovery
from src.core.circuit_breaker import circuit_breaker_registry
from src.core.lock_manager import distributed_lock_manager
from src.core.config_manager import config_manager
from src.monitoring.performance_monitor import performance_monitor
from src.monitoring.monitor_manager import MonitorManager
from src.market.analyzer import MarketAnalyzer
from src.execution.transaction_builder import TransactionBuilder
from src.validation.transaction_validator import TransactionValidator

class IntegrationTestContext:
    """Context for integration tests"""
    def __init__(self):
        self.monitor_manager = MonitorManager()
        self.market_analyzer = MarketAnalyzer()
        self.transaction_builder = TransactionBuilder()
        self.transaction_validator = TransactionValidator()
        self.test_data: Dict[str, Any] = {}

    async def setup(self) -> None:
        """Setup test context"""
        # Initialize core components
        await error_recovery.initialize()
        await circuit_breaker_registry.initialize()
        await distributed_lock_manager.initialize()
        await config_manager.initialize()
        
        # Initialize monitoring
        await self.monitor_manager.start()
        
        # Initialize market components
        await self.market_analyzer.initialize()
        await self.transaction_builder.initialize()
        await self.transaction_validator.initialize()

    async def teardown(self) -> None:
        """Cleanup test context"""
        await self.monitor_manager.stop()
        await error_recovery.shutdown()
        await circuit_breaker_registry.shutdown()
        await distributed_lock_manager.shutdown()
        await config_manager.shutdown()

@pytest.mark.integration
class TestSystemIntegration:
    """System integration test suite"""
    
    @pytest.fixture
    async def test_context(self):
        """Create test context"""
        context = IntegrationTestContext()
        await context.setup()
        yield context
        await context.teardown()

    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, test_context):
        """Test complete end-to-end system flow"""
        # 1. Market Analysis
        market_data = await test_context.market_analyzer.analyze_market()
        assert market_data is not None
        assert 'opportunities' in market_data
        
        # 2. Opportunity Validation
        for opportunity in market_data['opportunities']:
            validation_result = await test_context.transaction_validator.validate_opportunity(
                opportunity
            )
            assert validation_result.is_valid
            
            # 3. Transaction Building
            tx = await test_context.transaction_builder.build_transaction(opportunity)
            assert tx is not None
            assert 'gas_price' in tx
            
            # 4. Record metrics
            test_context.monitor_manager.record_trade(
                strategy=opportunity['strategy'],
                token_pair=opportunity['token_pair'],
                profit=opportunity['predicted_profit'],
                gas_price=tx['gas_price']
            )

    @pytest.mark.asyncio
    async def test_error_recovery_integration(self, test_context):
        """Test error recovery system integration"""
        # Simulate cascading failures
        async def trigger_failures():
            # 1. Network failure
            with pytest.raises(aiohttp.ClientError):
                async with aiohttp.ClientSession() as session:
                    await session.get('http://invalid-url')
            
            # 2. Redis failure
            await distributed_lock_manager.acquire_lock('test_lock')  # Should fail
            
            # 3. Market data failure
            await test_context.market_analyzer.analyze_market()  # Should recover
        
        await trigger_failures()
        
        # Verify error recovery
        assert error_recovery.recovery_attempts > 0
        assert error_recovery.last_error is not None
        
        # Verify system recovered
        market_data = await test_context.market_analyzer.analyze_market()
        assert market_data is not None

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, test_context):
        """Test monitoring system integration"""
        # Generate test data
        test_trades = [
            {
                'strategy': 'test_strategy',
                'token_pair': 'ETH-USDC',
                'profit': 1.0 * i,
                'gas_price': 50.0,
                'success': True
            }
            for i in range(5)
        ]
        
        # Record trades
        for trade in test_trades:
            test_context.monitor_manager.record_trade(**trade)
        
        # Wait for processing
        await asyncio.sleep(1)
        
        # Verify metrics
        metrics = await test_context.monitor_manager.get_metrics()
        assert metrics['total_trades'] == len(test_trades)
        assert metrics['successful_trades'] == len(test_trades)
        
        # Verify performance monitoring
        performance_data = performance_monitor.get_performance_summary()
        assert performance_data is not None
        assert 'cpu_usage' in performance_data
        assert 'memory_usage' in performance_data

    @pytest.mark.asyncio
    async def test_config_integration(self, test_context):
        """Test configuration management integration"""
        # Update configuration
        new_config = {
            'trading': {
                'max_slippage': 0.01,
                'min_profit': 0.005
            }
        }
        
        await config_manager.update_config(new_config)
        
        # Verify configuration propagation
        assert test_context.transaction_validator.max_slippage == 0.01
        assert test_context.transaction_validator.min_profit == 0.005
        
        # Verify config change notification
        assert config_manager.last_update is not None
        assert config_manager.version > 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, test_context):
        """Test circuit breaker integration"""
        # Configure test breaker
        breaker = await circuit_breaker_registry.register_breaker(
            'test_service',
            failure_threshold=3,
            reset_timeout=5
        )
        
        # Trigger failures
        for _ in range(4):
            try:
                raise Exception("Test failure")
            except Exception as e:
                await breaker.record_failure(e)
        
        # Verify breaker opened
        assert await breaker.is_open()
        
        # Wait for reset
        await asyncio.sleep(6)
        
        # Verify breaker closed
        assert not await breaker.is_open()

    @pytest.mark.asyncio
    async def test_distributed_lock_integration(self, test_context):
        """Test distributed lock integration"""
        lock_name = "test_integration_lock"
        
        # Acquire lock
        acquired = await distributed_lock_manager.acquire_lock(lock_name)
        assert acquired
        
        # Verify lock exists
        assert await distributed_lock_manager.lock_exists(lock_name)
        
        # Try to acquire again (should fail)
        acquired_again = await distributed_lock_manager.acquire_lock(lock_name)
        assert not acquired_again
        
        # Release lock
        await distributed_lock_manager.release_lock(lock_name)
        
        # Verify lock released
        assert not await distributed_lock_manager.lock_exists(lock_name)

    @pytest.mark.asyncio
    async def test_market_analyzer_integration(self, test_context):
        """Test market analyzer integration"""
        # Get market data
        market_data = await test_context.market_analyzer.analyze_market()
        
        # Verify market data structure
        assert 'opportunities' in market_data
        assert 'market_conditions' in market_data
        assert 'timestamp' in market_data
        
        # Analyze specific opportunity
        if market_data['opportunities']:
            opportunity = market_data['opportunities'][0]
            analysis = await test_context.market_analyzer.analyze_opportunity(
                opportunity['token_pair']
            )
            
            # Verify analysis data
            assert 'liquidity' in analysis
            assert 'volume' in analysis
            assert 'price_impact' in analysis
            
            # Record analysis in monitoring
            test_context.monitor_manager.record_market_analysis(analysis) 