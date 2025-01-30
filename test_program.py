import asyncio
import logging
from superchain_utils import (
    telegram_bot, arbitrage_strategy, testing_protocol,
    compliance_manager, gas_optimizer, profit_compounder,
    multi_hop_detector, tax_recorder
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_telegram_bot():
    """Test Telegram bot functionality"""
    try:
        # Initialize bot
        await telegram_bot.initialize()
        
        # Test notification
        test_message = "Test notification message"
        notification_result = await telegram_bot.notify_opportunity(test_message)
        assert notification_result, "Telegram notification failed"
        
        logger.info("Telegram bot tests passed")
        return True
    except Exception as e:
        logger.error(f"Telegram bot test failed: {str(e)}")
        return False

async def test_arbitrage_strategy():
    """Test arbitrage strategy functionality"""
    try:
        # Initialize strategy
        await arbitrage_strategy.initialize()
        
        # Test with sample market data
        sample_market_data = {
            'aerodrome': [{
                'token0': 'ETH',
                'token1': 'USDC',
                'liquidity': 1000000,
                'volume24h': 500000,
                'price': 2000,
                'fee': 0.003
            }]
        }
        
        # Update market data
        await multi_hop_detector.update_liquidity_graph(sample_market_data)
        
        logger.info("Arbitrage strategy tests passed")
        return True
    except Exception as e:
        logger.error(f"Arbitrage strategy test failed: {str(e)}")
        return False

async def test_tax_recorder():
    """Test tax recording functionality"""
    try:
        # Create sample trade data
        sample_trade = {
            'type': 'arbitrage',
            'tokens_involved': ['ETH', 'USDC'],
            'quantity': 1.0,
            'execution_result': {
                'profit': 0.1,
                'total_value': 2000,
                'entry_value': 1990,
                'gas_cost': 0.005,
                'fees': 0.002,
                'tx_hash': '0x123'
            },
            'timestamp': '2024-01-30T00:00:00'
        }
        
        # Record trade
        await tax_recorder.record_arbitrage_trade(sample_trade)
        
        # Test report generation
        report = await tax_recorder.generate_tax_report()
        assert report is not None, "Tax report generation failed"
        
        logger.info("Tax recorder tests passed")
        return True
    except Exception as e:
        logger.error(f"Tax recorder test failed: {str(e)}")
        return False

async def test_multi_hop_detector():
    """Test multi-hop detection functionality"""
    try:
        # Sample market data
        market_data = {
            'aerodrome': [{
                'token0': 'ETH',
                'token1': 'USDC',
                'liquidity': 1000000,
                'volume24h': 500000,
                'price': 2000,
                'fee': 0.003
            }]
        }
        
        # Update liquidity graph
        success = await multi_hop_detector.update_liquidity_graph(market_data)
        assert success, "Failed to update liquidity graph"
        
        # Find paths
        paths = multi_hop_detector.find_profitable_paths('ETH', 1.0)
        logger.info(f"Found {len(paths)} potential arbitrage paths")
        
        logger.info("Multi-hop detector tests passed")
        return True
    except Exception as e:
        logger.error(f"Multi-hop detector test failed: {str(e)}")
        return False

async def cleanup():
    """Cleanup all components"""
    try:
        # Stop Telegram bot
        if telegram_bot.initialized:
            await telegram_bot.stop()
        
        # Close any open connections
        # Add other cleanup as needed
        
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

async def main():
    """Run comprehensive system test"""
    try:
        logger.info("Starting comprehensive system test...")
        
        # Test all components
        components_to_test = [
            ('Telegram Bot', test_telegram_bot()),
            ('Arbitrage Strategy', test_arbitrage_strategy()),
            ('Tax Recorder', test_tax_recorder()),
            ('Multi-Hop Detector', test_multi_hop_detector())
        ]
        
        for component_name, test_coro in components_to_test:
            logger.info(f"Testing {component_name}...")
            success = await test_coro
            assert success, f"{component_name} tests failed"
        
        logger.info("All component tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
    finally:
        logger.info("Cleaning up resources...")
        await cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 