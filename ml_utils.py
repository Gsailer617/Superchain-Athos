import torch
import torch.nn as nn
from web3 import Web3
from web3.contract import Contract
from typing import Dict, Callable, List
import json
import logging
from agent.event_monitor import EventMonitor

logger = logging.getLogger(__name__)

def get_dex_router_abi(dex_name: str) -> List:
    """Get ABI for DEX router"""
    try:
        # Load ABI from file based on DEX name
        with open(f'abis/{dex_name.lower()}_router.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading ABI for {dex_name}: {str(e)}")
        return [] 