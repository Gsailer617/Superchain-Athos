from typing import Dict, Optional, Any, List, Union, cast, TypedDict
import json
import logging
from web3 import Web3
from web3.contract.contract import Contract
from web3.types import TxParams, Wei, HexBytes, HexStr
from eth_typing import Address, HexAddress
from pathlib import Path
from src.core.chain_connector import get_chain_connector
from src.core.chain_config import get_chain_registry

logger = logging.getLogger(__name__)

class ContractDict(TypedDict):
    """Type for contract dictionary"""
    contract: Contract
    address: str
    abi: Dict[str, Any]

class ContractManager:
    """Manages contract interactions across multiple chains"""
    
    def __init__(self):
        """Initialize contract manager"""
        self.chain_connector = get_chain_connector()
        self.chain_registry = get_chain_registry()
        self.contracts: Dict[str, Dict[str, ContractDict]] = {}
        self.abis: Dict[str, Any] = {}
        self._load_abis()
        self._load_contract_addresses()
        
    def _load_contract_addresses(self):
        """Load contract addresses for each chain"""
        self.contract_addresses = {
            'ethereum': {
                'uniswap_v3_factory': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                'uniswap_v3_router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'aave_v3_pool': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2',
                'curve_pool_registry': '0x90E00ACe148ca3b23Ac1bC8C240C2a7Dd9c2d7f5',
                'balancer_vault': '0xBA12222222228d8Ba445958a75a0704d566BF2C8'
            },
            'polygon': {
                'uniswap_v3_factory': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                'uniswap_v3_router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
                'aave_v3_pool': '0x794a61358D6845594F94dc1DB02A252b5b4814aD',
                'curve_pool_registry': '0x47bB542B9dE58b970bA50c9dae444DDB4c16751a',
                'balancer_vault': '0xBA12222222228d8Ba445958a75a0704d566BF2C8'
            },
            'base': {
                'uniswap_v3_factory': '0x33128a8fC17869897dcE68Ed026d694621f6FDfD',
                'uniswap_v3_router': '0x2626664c2603336E57B271c5C0b26F421741e481',
                'aerodrome_factory': '0x420DD381b31aEf6683db6B902084cB0FFEe076115',
                'baseswap_factory': '0xFDa619b6d20975be80A10332cD39b9a4b0FAa8BB',
                'spark_pool': '0xC13e21B648A5Ee794902342038FF3aDAB66BE987'
            }
        }
        
    def _load_abis(self):
        """Load contract ABIs from artifacts"""
        artifacts_dir = Path('artifacts/contracts')
        self.abis = {}
        
        # Load core protocol ABIs
        protocol_abis = {
            'UniswapV3Factory': 'IUniswapV3Factory.sol/IUniswapV3Factory.json',
            'UniswapV3Router': 'ISwapRouter.sol/ISwapRouter.json',
            'AaveV3Pool': 'IPool.sol/IPool.json',
            'CurveRegistry': 'ICurveRegistry.sol/ICurveRegistry.json',
            'BalancerVault': 'IVault.sol/IVault.json'
        }
        
        for name, path in protocol_abis.items():
            try:
                with open(artifacts_dir / path) as f:
                    contract_json = json.load(f)
                    self.abis[name] = contract_json['abi']
            except Exception as e:
                logger.error(f"Error loading ABI for {name}: {str(e)}")
                
    async def get_contract(
        self,
        chain: str,
        protocol: str,
        name: str
    ) -> Optional[Contract]:
        """Get contract instance"""
        try:
            if chain not in self.contracts:
                self.contracts[chain] = {}
            
            if protocol not in self.contracts[chain]:
                self.contracts[chain][protocol] = {}
                
            if name in self.contracts[chain][protocol]:
                return self.contracts[chain][protocol][name]['contract']
                
            # Load contract
            web3 = await self.chain_connector.get_connection(chain)
            if not web3:
                return None
                
            address = self.chain_registry.get_contract_address(chain, protocol, name)
            if not address:
                return None
                
            contract = web3.eth.contract(
                address=Web3.to_checksum_address(address),
                abi=self.abis.get(name, [])
            )
            
            self.contracts[chain][protocol][name] = {
                'contract': contract,
                'address': address,
                'abi': self.abis.get(name, [])
            }
            
            return contract
            
        except Exception as e:
            logger.error(f"Error getting contract {name} on {chain}: {str(e)}")
            return None
    
    async def execute_flash_loan(
        self,
        chain: str,
        protocol: str,
        token_address: str,
        amount: int,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute flash loan on specified chain and protocol
        
        Args:
            chain: Chain name
            protocol: Protocol name (e.g. 'aave', 'balancer')
            token_address: Token to borrow
            amount: Amount to borrow
            params: Additional parameters for flash loan
            
        Returns:
            Transaction result
        """
        try:
            if protocol == 'aave':
                return await self._execute_aave_flash_loan(chain, token_address, amount, params)
            elif protocol == 'balancer':
                return await self._execute_balancer_flash_loan(chain, token_address, amount, params)
            else:
                raise ValueError(f"Unsupported flash loan protocol: {protocol}")
                
        except Exception as e:
            logger.error(f"Error executing flash loan on {protocol} {chain}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def execute_flash_swap(
        self,
        chain: str,
        protocol: str,
        token0: str,
        token1: str,
        amount: int,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute flash swap on specified chain and protocol
        
        Args:
            chain: Chain name
            protocol: Protocol name (e.g. 'uniswap', 'curve')
            token0: First token in pair
            token1: Second token in pair
            amount: Amount to swap
            params: Additional parameters for flash swap
            
        Returns:
            Transaction result
        """
        try:
            if protocol == 'uniswap':
                return await self._execute_uniswap_flash_swap(chain, token0, token1, amount, params)
            elif protocol == 'curve':
                return await self._execute_curve_flash_swap(chain, token0, token1, amount, params)
            else:
                raise ValueError(f"Unsupported flash swap protocol: {protocol}")
                
        except Exception as e:
            logger.error(f"Error executing flash swap on {protocol} {chain}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_aave_flash_loan(
        self,
        chain: str,
        token_address: str,
        amount: int,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Aave flash loan"""
        try:
            pool = await self.get_contract(chain, 'aave', 'pool')
            if not pool:
                raise ValueError("Failed to get Aave pool contract")
            
            # Get flash loan receiver contract
            receiver = await self.get_contract(chain, 'arbitrage', 'flash_receiver')
            if not receiver:
                raise ValueError("Failed to get flash loan receiver contract")
            
            # Prepare flash loan parameters
            assets = [token_address]
            amounts = [amount]
            modes = [0]  # 0 = no debt, 1 = stable, 2 = variable
            
            # Encode params for the callback
            encoded_params = Web3.encode_abi(
                ['string', 'address', 'bytes'],
                [
                    params.get('target_chain', ''),
                    params.get('target_bridge', '0x0000000000000000000000000000000000000000'),
                    params.get('cross_chain_data', b'')
                ]
            )
            
            # Execute flash loan
            tx_params: TxParams = {
                'from': params.get('sender'),
                'gas': Wei(1000000)
            }
            
            tx_hash = await pool.functions.flashLoan(
                receiver.address,
                assets,
                amounts,
                modes,
                params.get('on_behalf_of', receiver.address),
                encoded_params,
                0  # referral code
            ).transact(tx_params)
            
            return {
                'success': True,
                'transaction_hash': HexStr(tx_hash.hex()),
                'chain': chain
            }
            
        except Exception as e:
            logger.error(f"Error executing Aave flash loan on {chain}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_balancer_flash_loan(
        self,
        chain: str,
        token_address: str,
        amount: int,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Balancer flash loan"""
        try:
            vault = await self.get_contract(chain, 'balancer', 'vault')
            if not vault:
                raise ValueError("Failed to get Balancer vault contract")
            
            # Get flash loan receiver contract
            receiver = await self.get_contract(chain, 'arbitrage', 'flash_receiver')
            if not receiver:
                raise ValueError("Failed to get flash loan receiver contract")
            
            # Prepare flash loan parameters
            tokens = [token_address]
            amounts = [amount]
            
            # Encode user data
            user_data = Web3.encode_abi(
                ['string', 'address', 'bytes'],
                [
                    params.get('target_chain', ''),
                    params.get('target_bridge', '0x0000000000000000000000000000000000000000'),
                    params.get('cross_chain_data', b'')
                ]
            )
            
            # Execute flash loan
            tx_params: TxParams = {
                'from': params.get('sender'),
                'gas': Wei(1000000)
            }
            
            tx_hash = await vault.functions.flashLoan(
                receiver.address,
                tokens,
                amounts,
                user_data
            ).transact(tx_params)
            
            return {
                'success': True,
                'transaction_hash': HexStr(tx_hash.hex()),
                'chain': chain
            }
            
        except Exception as e:
            logger.error(f"Error executing Balancer flash loan on {chain}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_uniswap_flash_swap(
        self,
        chain: str,
        token0: str,
        token1: str,
        amount: int,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Uniswap flash swap"""
        try:
            factory = await self.get_contract(chain, 'uniswap', 'factory')
            if not factory:
                raise ValueError("Failed to get Uniswap factory contract")
            
            # Get pool address
            pool_address = await factory.functions.getPool(
                token0,
                token1,
                params.get('fee', 3000)  # Default to 0.3% fee tier
            ).call()
            
            if pool_address == '0x0000000000000000000000000000000000000000':
                raise ValueError(f"No Uniswap pool found for {token0}/{token1}")
            
            # Get pool contract
            web3 = await self.chain_connector.get_connection(chain)
            if not web3:
                raise ValueError(f"Failed to get Web3 connection for {chain}")
                
            pool = web3.eth.contract(
                address=Web3.to_checksum_address(pool_address),
                abi=self.abis['UniswapV3Pool']
            )
            
            # Encode callback data
            callback_data = Web3.encode_abi(
                ['address', 'address', 'bytes'],
                [
                    token0,
                    token1,
                    params.get('callback_data', b'')
                ]
            )
            
            # Calculate sqrt price limit
            sqrt_price_limit_x96 = 0  # 0 for unlimited price impact
            
            # Execute flash swap
            tx_params: TxParams = {
                'from': params.get('sender'),
                'gas': Wei(1000000)
            }
            
            tx_hash = await pool.functions.swap(
                params.get('recipient'),
                params.get('zero_for_one', True),
                amount,
                sqrt_price_limit_x96,
                callback_data
            ).transact(tx_params)
            
            return {
                'success': True,
                'transaction_hash': HexStr(tx_hash.hex()),
                'chain': chain,
                'pool_address': pool_address
            }
            
        except Exception as e:
            logger.error(f"Error executing Uniswap flash swap on {chain}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def _execute_curve_flash_swap(
        self,
        chain: str,
        token0: str,
        token1: str,
        amount: int,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Curve flash swap"""
        try:
            registry = await self.get_contract(chain, 'curve', 'registry')
            if not registry:
                raise ValueError("Failed to get Curve registry contract")
            
            # Find pool for token pair
            pool_address = await registry.functions.find_pool_for_coins(
                token0,
                token1
            ).call()
            
            if pool_address == '0x0000000000000000000000000000000000000000':
                raise ValueError(f"No Curve pool found for {token0}/{token1}")
            
            # Get pool contract
            web3 = await self.chain_connector.get_connection(chain)
            if not web3:
                raise ValueError(f"Failed to get Web3 connection for {chain}")
                
            pool = web3.eth.contract(
                address=Web3.to_checksum_address(pool_address),
                abi=self.abis['CurvePool']
            )
            
            # Get token indices in pool
            token0_index = await pool.functions.coins(0).call()
            token1_index = await pool.functions.coins(1).call()
            
            # Encode callback data
            callback_data = Web3.encode_abi(
                ['address', 'address', 'bytes'],
                [
                    token0,
                    token1,
                    params.get('callback_data', b'')
                ]
            )
            
            # Execute flash swap
            tx_params: TxParams = {
                'from': params.get('sender'),
                'gas': Wei(1000000)
            }
            
            tx_hash = await pool.functions.flash(
                [token0, token1],
                [amount, 0],  # Borrow amount0 of token0
                params.get('receiver'),
                callback_data
            ).transact(tx_params)
            
            return {
                'success': True,
                'transaction_hash': HexStr(tx_hash.hex()),
                'chain': chain,
                'pool_address': pool_address
            }
            
        except Exception as e:
            logger.error(f"Error executing Curve flash swap on {chain}: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def bridge_profits(
        self,
        chain: str,
        token_address: str,
        amount: int,
        target_chain: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Bridge profits to target chain
        
        Args:
            chain: Source chain name
            token_address: Token to bridge
            amount: Amount to bridge
            target_chain: Target chain name
            params: Additional parameters for bridging
            
        Returns:
            Transaction result
        """
        try:
            # Get arbitrage contract
            arbitrage = await self.get_contract(chain, 'arbitrage', 'flash_arbitrage')
            if not arbitrage is None:
                raise ValueError("Failed to get arbitrage contract")

            # Execute bridge transfer
            tx_params: TxParams = {
                'from': params.get('sender'),
                'gas': Wei(500000)  # Lower gas as it's a simpler operation
            }

            tx = await arbitrage.functions.bridgeProfits(
                token_address,
                amount,
                target_chain
            ).transact(tx_params)

            return {
                'success': True,
                'transaction_hash': HexStr(tx.hex()),
                'chain': chain,
                'target_chain': target_chain
            }

        except Exception as e:
            logger.error(f"Error bridging profits from {chain} to {target_chain}: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def send_cross_chain_message(
        self,
        source_chain: str,
        dest_chain: str,
        token: str,
        amount: int,
        beneficiary: str,
        execution_data: bytes,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Send cross-chain message via LayerZero"""
        try:
            # Get arbitrage contract
            arbitrage = await self.get_contract(source_chain, 'arbitrage', 'flash_arbitrage')
            if not arbitrage:
                raise ValueError("Failed to get arbitrage contract")

            # Get destination chain ID and path
            dest_chain_id = self.chain_registry.get_layerzero_chain_id(dest_chain)
            dest_path = self.chain_registry.get_layerzero_path(dest_chain)

            if not dest_chain_id or not dest_path:
                raise ValueError(f"Invalid destination chain configuration: {dest_chain}")

            # Get Web3 connection
            web3 = await self.chain_connector.get_connection(source_chain)
            if not web3:
                raise ValueError(f"Failed to get Web3 connection for {source_chain}")

            # Prepare transaction parameters
            sender = cast(HexAddress, params.get('sender'))
            tx_params = cast(TxParams, {
                'from': sender,
                'gas': Wei(500000),
                'value': Wei(params.get('fee', 0))
            })

            # Execute cross-chain message
            tx_hash = await arbitrage.functions.sendCrossChainMessage(
                dest_chain_id,
                dest_path,
                Web3.to_checksum_address(token),
                amount,
                Web3.to_checksum_address(beneficiary),
                execution_data
            ).transact(tx_params)

            return {
                'success': True,
                'transaction_hash': HexStr(tx_hash.hex()),
                'source_chain': source_chain,
                'destination_chain': dest_chain,
                'chain_id': dest_chain_id
            }

        except Exception as e:
            logger.error(f"Error sending cross-chain message from {source_chain} to {dest_chain}: {str(e)}")
            return {'success': False, 'error': str(e)}

    async def estimate_cross_chain_message_fee(
        self,
        source_chain: str,
        dest_chain: str,
        token: str,
        amount: int,
        beneficiary: str,
        execution_data: bytes
    ) -> Dict[str, Any]:
        """Estimate fee for cross-chain message"""
        try:
            # Get contracts
            arbitrage = await self.get_contract(source_chain, 'arbitrage', 'flash_arbitrage')
            if not arbitrage:
                raise ValueError("Failed to get arbitrage contract")

            # Get destination chain ID and path
            dest_chain_id = self.chain_registry.get_layerzero_chain_id(dest_chain)
            dest_path = self.chain_registry.get_layerzero_path(dest_chain)

            if not dest_chain_id or not dest_path:
                raise ValueError(f"Invalid destination chain configuration: {dest_chain}")

            # Get Web3 connection
            web3 = await self.chain_connector.get_connection(source_chain)
            if not web3:
                raise ValueError(f"Failed to get Web3 connection for {source_chain}")

            # Encode message payload using web3.eth.abi
            payload = web3.eth.abi.encode_abi(
                ['address', 'uint256', 'address', 'bytes'],
                [
                    Web3.to_checksum_address(token),
                    amount,
                    Web3.to_checksum_address(beneficiary),
                    execution_data
                ]
            )

            # Get LayerZero endpoint
            endpoint = await self.get_contract(source_chain, 'layerzero', 'endpoint')
            if not endpoint:
                raise ValueError("Failed to get LayerZero endpoint")

            # Estimate fees
            fees = await endpoint.functions.estimateFees(
                dest_chain_id,
                arbitrage.address,
                payload,
                False,  # Don't pay in ZRO
                b''  # No adapter params
            ).call()

            return {
                'success': True,
                'native_fee': fees[0],
                'zro_fee': fees[1],
                'source_chain': source_chain,
                'destination_chain': dest_chain
            }

        except Exception as e:
            logger.error(f"Error estimating cross-chain message fee: {str(e)}")
            return {'success': False, 'error': str(e)}

# Global instance
_contract_manager = None

def get_contract_manager() -> ContractManager:
    """Get global contract manager instance"""
    global _contract_manager
    if _contract_manager is None:
        _contract_manager = ContractManager()
    return _contract_manager 