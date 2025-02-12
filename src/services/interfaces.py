"""Abstract interfaces for external services"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from decimal import Decimal

class BlockchainServiceInterface(ABC):
    @abstractmethod
    async def get_token_balance(self, token_address: str, wallet_address: str) -> Decimal:
        """Get token balance for a wallet"""
        pass
    
    @abstractmethod
    async def estimate_gas(self, transaction: Dict[str, Any]) -> int:
        """Estimate gas for a transaction"""
        pass

class DexServiceInterface(ABC):
    @abstractmethod
    async def get_token_price(self, token_address: str, base_token: str) -> Decimal:
        """Get token price from DEX"""
        pass
    
    @abstractmethod
    async def get_liquidity_pools(self, token_address: str) -> List[Dict[str, Any]]:
        """Get liquidity pools for a token"""
        pass
    
    @abstractmethod
    async def simulate_swap(self, token_in: str, token_out: str, amount_in: Decimal) -> Dict[str, Any]:
        """Simulate a token swap"""
        pass

class DataProviderInterface(ABC):
    @abstractmethod
    async def get_token_metadata(self, token_address: str) -> Dict[str, Any]:
        """Get token metadata"""
        pass
    
    @abstractmethod
    async def get_market_data(self, token_address: str) -> Dict[str, Any]:
        """Get market data for a token"""
        pass

class ValidationServiceInterface(ABC):
    @abstractmethod
    async def validate_token(self, token_address: str) -> Dict[str, bool]:
        """Validate a token's security and legitimacy"""
        pass
    
    @abstractmethod
    async def check_contract_verification(self, token_address: str) -> bool:
        """Check if contract is verified"""
        pass

class MLServiceInterface(ABC):
    @abstractmethod
    async def predict_price_movement(self, token_data: Dict[str, Any]) -> Dict[str, float]:
        """Predict price movement for a token"""
        pass
    
    @abstractmethod
    async def analyze_sentiment(self, token_address: str) -> Dict[str, float]:
        """Analyze sentiment for a token"""
        pass 