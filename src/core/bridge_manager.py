from typing import Dict, List, Optional, Any, Tuple, cast
from web3 import Web3, AsyncWeb3
from web3.types import TxParams, Wei, HexStr, TxReceipt, _Hash32
import logging
from dataclasses import dataclass
import json
import aiohttp
import asyncio
from hexbytes import HexBytes

from ..config.chain_specs import ChainSpec
from ..utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

@dataclass
class BridgeInfo:
    """Information about a bridge"""
    name: str
    source_chain: str
    destination_chain: str
    token_address: str
    bridge_address: str
    is_native: bool = False
    min_amount: Wei = Wei(0)
    max_amount: Optional[Wei] = None
    estimated_time: int = 0  # seconds

class BridgeManager:
    """Manager for cross-chain bridge operations"""
    
    def __init__(self):
        self._bridges: Dict[str, Dict[str, List[BridgeInfo]]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._initialize_bridges()
    
    def _initialize_bridges(self) -> None:
        """Initialize supported bridges"""
        # Base <-> Ethereum (Official Bridge)
        self._add_bridge(BridgeInfo(
            name="Base Bridge",
            source_chain="ethereum",
            destination_chain="base",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0x49048044D57e1C92A77f79988d21Fa8fAF74E97e",
            is_native=True,
            estimated_time=1800  # 30 minutes
        ))
        
        # Mode <-> Ethereum (Mode Bridge)
        self._add_bridge(BridgeInfo(
            name="Mode Bridge",
            source_chain="ethereum",
            destination_chain="mode",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0x735aDBo45B7842eA46A2b2B7A81B368F8f9dEF8E",
            is_native=True,
            estimated_time=1800
        ))
        
        # Arbitrum <-> Ethereum (Arbitrum Bridge)
        self._add_bridge(BridgeInfo(
            name="Arbitrum Bridge",
            source_chain="ethereum",
            destination_chain="arbitrum",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0x8315177aB297bA92A06054cE80a67Ed4DBd7ed3a",
            is_native=True,
            estimated_time=900  # 15 minutes
        ))
        
        # Optimism <-> Ethereum (Optimism Bridge)
        self._add_bridge(BridgeInfo(
            name="Optimism Bridge",
            source_chain="ethereum",
            destination_chain="optimism",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0x99C9fc46f92E8a1c0deC1b1747d010903E884bE1",
            is_native=True,
            estimated_time=1800
        ))
        
        # Polygon <-> Ethereum (Polygon PoS Bridge)
        self._add_bridge(BridgeInfo(
            name="Polygon PoS Bridge",
            source_chain="ethereum",
            destination_chain="polygon",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
            is_native=True,
            estimated_time=3600  # 1 hour
        ))
        
        # BNB Chain <-> Ethereum (BSC Bridge)
        self._add_bridge(BridgeInfo(
            name="BSC Bridge",
            source_chain="ethereum",
            destination_chain="bnb",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0x2170Ed0880ac9A755fd29B2688956BD959F933F8",
            is_native=True,
            estimated_time=1800
        ))
        
        # Linea <-> Ethereum (Linea Bridge)
        self._add_bridge(BridgeInfo(
            name="Linea Bridge",
            source_chain="ethereum",
            destination_chain="linea",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0xE87d317eB8dcc9afE24d9f63D6C760e52Bc18A40",
            is_native=True,
            estimated_time=1800
        ))
        
        # Mantle <-> Ethereum (Mantle Bridge)
        self._add_bridge(BridgeInfo(
            name="Mantle Bridge",
            source_chain="ethereum",
            destination_chain="mantle",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0x95fC37A27a2f68e3A647CDc081F0A89bb47c3B79",
            is_native=True,
            estimated_time=1800
        ))
        
        # Sonic <-> Mode (Sonic Bridge)
        self._add_bridge(BridgeInfo(
            name="Sonic Bridge",
            source_chain="mode",
            destination_chain="sonic",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0x9D37A8bB37225d944F4d7f75c6F126dc21531C06",
            is_native=True,
            estimated_time=900
        ))
        
        # Avalanche <-> Ethereum (Avalanche Bridge)
        self._add_bridge(BridgeInfo(
            name="Avalanche Bridge",
            source_chain="ethereum",
            destination_chain="avalanche",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0x8EB8a3b98659Cce290402893d0123abb75E3ab28",
            is_native=True,
            estimated_time=1800
        ))
        
        # Gnosis <-> Ethereum (xDai Bridge)
        self._add_bridge(BridgeInfo(
            name="xDai Bridge",
            source_chain="ethereum",
            destination_chain="gnosis",
            token_address="0x0000000000000000000000000000000000000000",  # ETH
            bridge_address="0x4aa42145Aa6Ebf72e164C9bBC74fbD3788045016",
            is_native=True,
            estimated_time=900
        ))
    
    def _add_bridge(self, bridge: BridgeInfo) -> None:
        """Add bridge to supported bridges"""
        # Initialize source chain if needed
        if bridge.source_chain not in self._bridges:
            self._bridges[bridge.source_chain] = {}
        
        # Initialize destination chain if needed
        if bridge.destination_chain not in self._bridges[bridge.source_chain]:
            self._bridges[bridge.source_chain][bridge.destination_chain] = []
        
        # Add bridge
        self._bridges[bridge.source_chain][bridge.destination_chain].append(bridge)
    
    def get_bridges(
        self,
        source_chain: str,
        destination_chain: str
    ) -> List[BridgeInfo]:
        """Get available bridges between chains"""
        return self._bridges.get(source_chain, {}).get(destination_chain, [])
    
    @retry_with_backoff(max_retries=3)
    async def prepare_bridge_transaction(
        self,
        web3: AsyncWeb3,
        bridge: BridgeInfo,
        amount: Wei,
        from_address: str,
        to_address: str
    ) -> TxParams:
        """Prepare transaction for bridge transfer"""
        # Load bridge ABI based on bridge type
        bridge_abi = self._get_bridge_abi(bridge.name)
        
        # Create contract instance
        contract = web3.eth.contract(
            address=web3.to_checksum_address(bridge.bridge_address),
            abi=bridge_abi
        )
        
        # Prepare transaction based on bridge type
        if bridge.name == "Base Bridge":
            return await self._prepare_base_bridge_tx(
                contract, amount, from_address, to_address
            )
        elif bridge.name == "Mode Bridge":
            return await self._prepare_mode_bridge_tx(
                contract, amount, from_address, to_address
            )
        elif bridge.name == "Arbitrum Bridge":
            return await self._prepare_arbitrum_bridge_tx(
                contract, amount, from_address, to_address
            )
        # Add other bridge-specific transaction preparation
        else:
            raise ValueError(f"Unsupported bridge: {bridge.name}")
    
    def _get_bridge_abi(self, bridge_name: str) -> List[Dict[str, Any]]:
        """Get ABI for specific bridge"""
        # Load bridge ABI from JSON files
        abi_path = f"src/abis/{bridge_name.lower().replace(' ', '_')}.json"
        try:
            with open(abi_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ABI for {bridge_name}: {str(e)}")
            raise
    
    async def _prepare_base_bridge_tx(
        self,
        contract: Any,
        amount: Wei,
        from_address: str,
        to_address: str
    ) -> TxParams:
        """Prepare Base bridge transaction"""
        return {
            'from': from_address,
            'to': contract.address,
            'value': amount,
            'data': contract.encodeABI(
                fn_name='depositETH',
                args=[amount, to_address]
            )
        }
    
    async def _prepare_mode_bridge_tx(
        self,
        contract: Any,
        amount: Wei,
        from_address: str,
        to_address: str
    ) -> TxParams:
        """Prepare Mode bridge transaction"""
        return {
            'from': from_address,
            'to': contract.address,
            'value': amount,
            'data': contract.encodeABI(
                fn_name='bridgeETH',
                args=[to_address, amount]
            )
        }
    
    async def _prepare_arbitrum_bridge_tx(
        self,
        contract: Any,
        amount: Wei,
        from_address: str,
        to_address: str
    ) -> TxParams:
        """Prepare Arbitrum bridge transaction"""
        return {
            'from': from_address,
            'to': contract.address,
            'value': amount,
            'data': contract.encodeABI(
                fn_name='depositETH',
                args=[amount]
            )
        }
    
    async def monitor_bridge_transaction(
        self,
        source_web3: AsyncWeb3,
        destination_web3: AsyncWeb3,
        bridge: BridgeInfo,
        tx_hash: str,
        timeout: int = 3600
    ) -> Dict[str, Any]:
        """Monitor bridge transaction status"""
        start_time = asyncio.get_event_loop().time()
        
        while True:
            # Check if timeout exceeded
            if asyncio.get_event_loop().time() - start_time > timeout:
                return {
                    'status': 'timeout',
                    'message': f'Bridge transaction monitoring timed out after {timeout} seconds'
                }
            
            try:
                # Check source chain confirmation
                source_receipt = await source_web3.eth.get_transaction_receipt(HexBytes(tx_hash))
                if not source_receipt or not source_receipt['status']:
                    return {
                        'status': 'failed',
                        'message': 'Source chain transaction failed'
                    }
                
                # Check destination chain based on bridge type
                if bridge.name == "Base Bridge":
                    status = await self._check_base_bridge_status(
                        destination_web3, cast(Dict[str, Any], source_receipt)
                    )
                elif bridge.name == "Mode Bridge":
                    status = await self._check_mode_bridge_status(
                        destination_web3, cast(Dict[str, Any], source_receipt)
                    )
                # Add other bridge-specific status checks
                else:
                    status = {'status': 'unknown'}
                
                if status['status'] in ['completed', 'failed']:
                    return status
                
            except Exception as e:
                logger.warning(f"Error monitoring bridge transaction: {str(e)}")
            
            # Wait before next check
            await asyncio.sleep(30)
            
            # Return pending status if we haven't returned yet
            return {
                'status': 'pending',
                'message': 'Transaction still processing'
            }
    
    async def _check_base_bridge_status(
        self,
        destination_web3: AsyncWeb3,
        source_receipt: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check Base bridge transaction status"""
        # Implementation specific to Base bridge
        return {
            'status': 'pending',
            'message': 'Base bridge status check not yet implemented'
        }
    
    async def _check_mode_bridge_status(
        self,
        destination_web3: AsyncWeb3,
        source_receipt: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check Mode bridge transaction status"""
        # Implementation specific to Mode bridge
        return {
            'status': 'pending',
            'message': 'Mode bridge status check not yet implemented'
        }
    
    def get_supported_tokens(
        self,
        source_chain: str,
        destination_chain: str
    ) -> List[str]:
        """Get supported tokens for bridge pair"""
        bridges = self.get_bridges(source_chain, destination_chain)
        return list(set(bridge.token_address for bridge in bridges))
    
    def estimate_bridge_time(
        self,
        source_chain: str,
        destination_chain: str,
        bridge_name: Optional[str] = None
    ) -> int:
        """Estimate bridge transfer time in seconds"""
        bridges = self.get_bridges(source_chain, destination_chain)
        if not bridges:
            raise ValueError(f"No bridges found between {source_chain} and {destination_chain}")
        
        if bridge_name:
            bridge = next((b for b in bridges if b.name == bridge_name), None)
            if not bridge:
                raise ValueError(f"Bridge {bridge_name} not found")
            return bridge.estimated_time
        
        # Return fastest bridge time if no specific bridge specified
        return min(bridge.estimated_time for bridge in bridges) 