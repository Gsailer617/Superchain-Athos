from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from web3 import Web3

from .bridge_adapter import BridgeAdapter, BridgeConfig, BridgeState
from .debridge_adapter import DeBridgeAdapter
from .superbridge_adapter import SuperbridgeAdapter
from .across_adapter import AcrossAdapter
from .chain_configurations import CHAIN_CONFIGS, ChainConfig
from .register_adapters import get_registered_adapters

class BridgeIntegration:
    """Integration layer between bridge adapters and chain configurations"""
    
    def __init__(self, web3: Web3):
        self.web3 = web3
        self.chain_configs = CHAIN_CONFIGS
        self.bridge_adapters = get_registered_adapters()
        self._bridge_instances: Dict[str, BridgeAdapter] = {}
        
    def get_supported_bridges(
        self,
        source_chain: str,
        target_chain: str,
        token: str
    ) -> List[Tuple[str, BridgeAdapter]]:
        """Get list of supported bridges for a given chain pair and token
        
        Args:
            source_chain: Source chain name
            target_chain: Target chain name
            token: Token symbol
            
        Returns:
            List of (bridge_name, bridge_adapter) tuples
        """
        supported_bridges = []
        
        # Get chain configs
        source_config = self.chain_configs.get(source_chain)
        target_config = self.chain_configs.get(target_chain)
        
        if not source_config or not target_config:
            return []
            
        # Check each bridge adapter
        for bridge_name, adapter_class in self.bridge_adapters.items():
            try:
                # Get or create bridge instance
                bridge = self._get_bridge_instance(bridge_name, adapter_class)
                
                # Check if bridge supports this transfer
                if bridge.validate_transfer(source_chain, target_chain, token, 0):
                    supported_bridges.append((bridge_name, bridge))
                    
            except Exception as e:
                continue
                
        return supported_bridges
    
    def get_optimal_bridge(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float,
        max_time: Optional[int] = None,
        max_cost: Optional[float] = None
    ) -> Tuple[Optional[str], Optional[BridgeAdapter], Dict[str, Any]]:
        """Get optimal bridge for a transfer based on constraints
        
        Args:
            source_chain: Source chain name
            target_chain: Target chain name
            token: Token symbol
            amount: Amount to transfer
            max_time: Maximum acceptable time in seconds
            max_cost: Maximum acceptable cost in native token
            
        Returns:
            Tuple of (bridge_name, bridge_adapter, bridge_info)
        """
        best_bridge = None
        best_adapter = None
        best_score = float('-inf')
        best_info = {}
        
        supported_bridges = self.get_supported_bridges(source_chain, target_chain, token)
        
        for bridge_name, adapter in supported_bridges:
            try:
                # Get bridge state
                state = adapter.get_bridge_state(source_chain, target_chain)
                if state != BridgeState.ACTIVE:
                    continue
                
                # Get time estimate
                time_estimate = adapter.estimate_time(source_chain, target_chain)
                if max_time and time_estimate > max_time:
                    continue
                
                # Get fee estimate
                fees = adapter.estimate_fees(source_chain, target_chain, token, amount)
                total_cost = fees.get('total', float('inf'))
                if max_cost and total_cost > max_cost:
                    continue
                
                # Get liquidity
                liquidity = adapter.monitor_liquidity(target_chain, token)
                
                # Calculate score (can be customized based on preferences)
                time_score = 1.0 / (1.0 + time_estimate / 3600)  # Normalized by hour
                cost_score = 1.0 / (1.0 + total_cost)
                liquidity_score = min(1.0, liquidity / (amount * 2))  # 2x liquidity ideal
                
                # Weighted score
                score = (
                    0.4 * cost_score +
                    0.3 * time_score +
                    0.3 * liquidity_score
                )
                
                if score > best_score:
                    best_score = score
                    best_bridge = bridge_name
                    best_adapter = adapter
                    best_info = {
                        'estimated_time': time_estimate,
                        'fees': fees,
                        'liquidity': liquidity,
                        'score': score,
                        'score_breakdown': {
                            'cost_score': cost_score,
                            'time_score': time_score,
                            'liquidity_score': liquidity_score
                        }
                    }
                
            except Exception as e:
                continue
        
        return best_bridge, best_adapter, best_info
    
    def prepare_bridge_transfer(
        self,
        source_chain: str,
        target_chain: str,
        token: str,
        amount: float,
        recipient: str,
        bridge_name: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Prepare a bridge transfer transaction
        
        Args:
            source_chain: Source chain name
            target_chain: Target chain name
            token: Token symbol
            amount: Amount to transfer
            recipient: Recipient address
            bridge_name: Optional specific bridge to use
            
        Returns:
            Tuple of (transaction_data, bridge_info)
        """
        try:
            if bridge_name:
                # Use specified bridge
                adapter_class = self.bridge_adapters.get(bridge_name)
                if not adapter_class:
                    raise ValueError(f"Bridge {bridge_name} not found")
                bridge = self._get_bridge_instance(bridge_name, adapter_class)
            else:
                # Find optimal bridge
                bridge_name, bridge, bridge_info = self.get_optimal_bridge(
                    source_chain,
                    target_chain,
                    token,
                    amount
                )
                if not bridge:
                    return None, {'error': 'No suitable bridge found'}
            
            # Prepare transfer transaction
            tx_params = bridge.prepare_transfer(
                source_chain,
                target_chain,
                token,
                amount,
                recipient
            )
            
            # Get bridge info
            bridge_info = {
                'bridge': bridge_name,
                'estimated_time': bridge.estimate_time(source_chain, target_chain),
                'fees': bridge.estimate_fees(source_chain, target_chain, token, amount),
                'liquidity': bridge.monitor_liquidity(target_chain, token),
                'state': bridge.get_bridge_state(source_chain, target_chain)
            }
            
            return tx_params, bridge_info
            
        except Exception as e:
            return None, {'error': str(e)}
    
    def monitor_bridge_transfer(
        self,
        source_chain: str,
        target_chain: str,
        tx_hash: str,
        bridge_name: str
    ) -> Dict[str, Any]:
        """Monitor status of a bridge transfer
        
        Args:
            source_chain: Source chain name
            target_chain: Target chain name
            tx_hash: Transaction hash
            bridge_name: Bridge used for transfer
            
        Returns:
            Transfer status information
        """
        try:
            adapter_class = self.bridge_adapters.get(bridge_name)
            if not adapter_class:
                raise ValueError(f"Bridge {bridge_name} not found")
                
            bridge = self._get_bridge_instance(bridge_name, adapter_class)
            
            # Get source transaction status
            source_status = bridge.get_transaction_status(source_chain, tx_hash)
            
            # Get bridge state
            bridge_state = bridge.get_bridge_state(source_chain, target_chain)
            
            # Get message verification status if applicable
            message_verified = False
            if source_status.get('status') == 'success':
                message_hash = source_status.get('message_hash')
                if message_hash:
                    message_verified = bridge.verify_message(
                        source_chain,
                        target_chain,
                        message_hash,
                        source_status.get('proof', b'')
                    )
            
            return {
                'source_status': source_status,
                'bridge_state': bridge_state,
                'message_verified': message_verified,
                'confirmations': source_status.get('confirmations', 0),
                'estimated_completion': source_status.get('estimated_completion')
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _get_bridge_instance(
        self,
        bridge_name: str,
        adapter_class: type
    ) -> BridgeAdapter:
        """Get or create bridge adapter instance"""
        if bridge_name not in self._bridge_instances:
            self._bridge_instances[bridge_name] = adapter_class(
                self._create_bridge_config(bridge_name),
                self.web3
            )
        return self._bridge_instances[bridge_name]
    
    def _create_bridge_config(self, bridge_name: str) -> BridgeConfig:
        """Create bridge configuration"""
        # This would be expanded based on specific bridge requirements
        return BridgeConfig(
            name=bridge_name,
            supported_chains=list(self.chain_configs.keys()),
            min_amount=0.0,
            max_amount=float('inf'),
            fee_multiplier=1.0,
            gas_limit_multiplier=1.2,
            confirmation_blocks=1
        ) 