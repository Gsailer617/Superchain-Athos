import os
import sys
import asyncio
import logging
from dotenv import load_dotenv
from web3 import Web3
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SuperchainArbitrageAgent import SuperchainArbitrageAgent
from src.core.web3_config import get_web3, get_provider

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def main():
    monitoring_task = None
    try:
        # Get Web3 from centralized provider
        web3_provider = get_provider()
        w3 = web3_provider.web3
        logger.info("Using centralized Web3 provider in example script")
        
        # Initialize agent with config path
        agent = SuperchainArbitrageAgent(config_path='config.json')
        
        # Print initial configuration
        logging.info("Current configuration:")
        logging.info(f"Network: Base Mainnet")
        
        # Start monitoring in a task
        monitoring_task = asyncio.create_task(agent.monitor_superchain())
        logging.info("Agent started successfully")
        
        # Keep the script running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Stopping agent...")
        if monitoring_task:
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass
        logging.info("Agent stopped")
        
    except Exception as e:
        logging.error(f"Error running agent: {str(e)}")
        if monitoring_task:
            monitoring_task.cancel()
            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    asyncio.run(main()) 