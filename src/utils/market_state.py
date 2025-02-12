from dataclasses import dataclass, field
from typing import Optional, List, Dict
from decimal import Decimal
from datetime import datetime
import numpy as np

@dataclass
class MarketState:
    """Represents the current state of the market for a trading pair"""
    price: Decimal
    volume: Decimal
    liquidity: Decimal
    timestamp: float
    volatility: Optional[float] = None
    price_change_24h: Optional[float] = None
    volume_change_24h: Optional[float] = None
    market_cap: Optional[Decimal] = None
    tvl: Optional[Decimal] = None
    historical_prices: List[Dict[str, float]] = field(default_factory=list)
    historical_volumes: List[Dict[str, float]] = field(default_factory=list)
    max_history_length: int = 1000
    
    def __post_init__(self):
        # Convert string or float inputs to Decimal where needed
        self.price = Decimal(str(self.price))
        self.volume = Decimal(str(self.volume))
        self.liquidity = Decimal(str(self.liquidity))
        if self.market_cap is not None:
            self.market_cap = Decimal(str(self.market_cap))
        if self.tvl is not None:
            self.tvl = Decimal(str(self.tvl))
        
        # Validate inputs
        self._validate_inputs()
            
    def _validate_inputs(self):
        """Validate market state inputs"""
        if self.price <= 0:
            raise ValueError("Price must be positive")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
        if self.liquidity < 0:
            raise ValueError("Liquidity cannot be negative")
        if self.timestamp > datetime.now().timestamp():
            raise ValueError("Timestamp cannot be in the future")
            
    def update_state(self, new_price: float, new_volume: float, 
                    new_liquidity: float, timestamp: float) -> None:
        """Update market state with new data"""
        # Store historical data
        self.historical_prices.append({
            'value': float(self.price),
            'timestamp': self.timestamp
        })
        self.historical_volumes.append({
            'value': float(self.volume),
            'timestamp': self.timestamp
        })
        
        # Trim historical data if needed
        if len(self.historical_prices) > self.max_history_length:
            self.historical_prices = self.historical_prices[-self.max_history_length:]
            self.historical_volumes = self.historical_volumes[-self.max_history_length:]
        
        # Update current state
        self.price = Decimal(str(new_price))
        self.volume = Decimal(str(new_volume))
        self.liquidity = Decimal(str(new_liquidity))
        self.timestamp = timestamp
        
        # Update derived metrics
        self._update_metrics()
            
    def _update_metrics(self):
        """Update derived market metrics"""
        if len(self.historical_prices) > 1:
            # Calculate volatility
            prices = [p['value'] for p in self.historical_prices[-24:]]  # Last 24 data points
            if len(prices) > 1:
                self.volatility = float(np.std(prices) / np.mean(prices))
            
            # Calculate 24h changes
            day_ago_price = self.historical_prices[-24]['value'] if len(self.historical_prices) >= 24 else self.historical_prices[0]['value']
            day_ago_volume = self.historical_volumes[-24]['value'] if len(self.historical_volumes) >= 24 else self.historical_volumes[0]['value']
            
            self.price_change_24h = (float(self.price) - day_ago_price) / day_ago_price * 100
            self.volume_change_24h = (float(self.volume) - day_ago_volume) / day_ago_volume * 100
            
    def to_dict(self) -> dict:
        """Convert the market state to a dictionary"""
        return {
            'price': float(self.price),
            'volume': float(self.volume),
            'liquidity': float(self.liquidity),
            'timestamp': self.timestamp,
            'volatility': self.volatility,
            'price_change_24h': self.price_change_24h,
            'volume_change_24h': self.volume_change_24h,
            'market_cap': float(self.market_cap) if self.market_cap else None,
            'tvl': float(self.tvl) if self.tvl else None,
            'historical_data': {
                'prices': self.historical_prices,
                'volumes': self.historical_volumes
            }
        }
        
    def get_historical_data(self, timeframe: str = '24h') -> Dict:
        """Get historical data for a specific timeframe"""
        if timeframe == '24h':
            prices = self.historical_prices[-24:]
            volumes = self.historical_volumes[-24:]
        elif timeframe == '7d':
            prices = self.historical_prices[-168:]  # 24 * 7
            volumes = self.historical_volumes[-168:]
        else:
            prices = self.historical_prices
            volumes = self.historical_volumes
            
        return {
            'prices': prices,
            'volumes': volumes
        } 