#!/usr/bin/env python3
"""Example script demonstrating the integration of gas optimization with execution components"""

import asyncio
import logging
import os
import json
from web3 import Web3
from dotenv import load_dotenv

from src.gas import GasExecutionIntegrator, EnhancedGasOptimizer
from src.execution import ExecutionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def main():
    """Main function demonstrating gas optimization with execution"""
    try:
        # Load configuration
        config = {
            'gas': {
                'use_eip1559': True,
                'max_priority_fee': 2_000_000_000,  # 2 Gwei
                'max_fee_per_gas': 100_000_000_000,  # 100 Gwei
                'use_external_api': True,
                'etherscan_api_key': os.getenv('ETHERSCAN_API_KEY', '')
            },
            'execution': {
                'simulate_transactions': True,
                'retry_count': 3,
                'confirmation_blocks': 2
            }
        }
        
        # Connect to Ethereum node
        web3 = Web3(Web3.HTTPProvider(os.getenv('ETH_RPC_URL', 'http://localhost:8545')))
        
        if not web3.is_connected():
            logger.error("Failed to connect to Ethereum node")
            return
            
        logger.info(f"Connected to Ethereum node: {web3.eth.chain_id}")
        
        # Create gas execution integrator
        integrator = GasExecutionIntegrator(web3, config)
        
        # Example transaction parameters
        tx_params = {
            'from': os.getenv('SENDER_ADDRESS'),
            'to': os.getenv('RECEIVER_ADDRESS', '0x742d35Cc6634C0532925a3b844Bc454e4438f44e'),
            'value': web3.to_wei(0.001, 'ether'),
            'data': '0x'
        }
        
        # Estimate transaction cost with different optimization modes
        logger.info("Estimating transaction costs with different optimization modes...")
        
        for mode in ['economy', 'normal', 'performance', 'urgent']:
            estimate = await integrator.estimate_transaction_cost(tx_params, mode)
            
            logger.info(f"Mode: {mode}")
            logger.info(f"  Estimated cost: {estimate['estimated_cost_eth']:.6f} ETH")
            logger.info(f"  Estimated wait time: {estimate['estimated_wait_time']} seconds")
            logger.info(f"  Network congestion: {estimate['network_congestion']:.2f}")
            
            # Print gas parameters
            gas_params = estimate['gas_params']
            if 'maxFeePerGas' in gas_params:
                logger.info(f"  Max fee: {web3.from_wei(gas_params['maxFeePerGas'], 'gwei'):.2f} Gwei")
                logger.info(f"  Priority fee: {web3.from_wei(gas_params.get('maxPriorityFeePerGas', 0), 'gwei'):.2f} Gwei")
            elif 'gasPrice' in gas_params:
                logger.info(f"  Gas price: {web3.from_wei(gas_params['gasPrice'], 'gwei'):.2f} Gwei")
            
            logger.info("")
        
        # Ask for confirmation before sending a real transaction
        if os.getenv('EXECUTE_TRANSACTION', 'false').lower() == 'true':
            logger.info("Executing transaction with optimized gas settings...")
            
            # Execute transaction with normal optimization mode
            result = await integrator.execute_with_optimized_gas(
                tx_params,
                optimization_mode='normal',
                wait_for_receipt=True
            )
            
            # Print transaction result
            logger.info(f"Transaction hash: {result.get('tx_hash')}")
            
            if 'receipt' in result:
                receipt = result['receipt']
                logger.info(f"Transaction status: {'Success' if receipt.get('status') == 1 else 'Failed'}")
                logger.info(f"Gas used: {receipt.get('gasUsed')}")
                logger.info(f"Effective gas price: {web3.from_wei(receipt.get('effectiveGasPrice', 0), 'gwei'):.2f} Gwei")
                
            if 'gas_optimization' in result:
                opt = result['gas_optimization']
                logger.info(f"Optimization mode: {opt.get('mode')}")
                logger.info(f"Estimated savings: {opt.get('estimated_savings', 0) * 100:.2f}%")
        else:
            logger.info("Transaction execution skipped. Set EXECUTE_TRANSACTION=true to execute a real transaction.")
        
        # Clean up resources
        await integrator.cleanup()
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        
if __name__ == "__main__":
    asyncio.run(main()) 