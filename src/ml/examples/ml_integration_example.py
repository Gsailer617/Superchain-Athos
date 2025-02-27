"""Example script demonstrating how to use the ML integration with advanced models.

This script shows how to:
1. Initialize the ML integration with the base model
2. Use VAE for anomaly detection and pattern learning
3. Use SAC for reinforcement learning and strategy optimization
4. Use attention mechanisms for time series understanding
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ml.model import ArbitrageModel, MarketDataType
from src.ml.ml_integration import MLModelIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_market_data(num_samples=100, seq_length=32):
    """Create sample market data for demonstration."""
    # Create a dictionary to hold market data
    market_data = {}
    
    # Basic market features
    market_data["price"] = np.random.normal(100, 10, num_samples)
    market_data["volume_24h"] = np.random.normal(1000000, 200000, num_samples)
    market_data["liquidity"] = np.random.normal(5000000, 1000000, num_samples)
    market_data["volatility"] = np.random.normal(0.05, 0.02, num_samples)
    market_data["market_cap"] = np.random.normal(1000000000, 200000000, num_samples)
    market_data["tvl"] = np.random.normal(500000000, 100000000, num_samples)
    market_data["fees_24h"] = np.random.normal(100000, 20000, num_samples)
    market_data["gas_price"] = np.random.normal(50, 10, num_samples)
    market_data["block_time"] = np.random.normal(12, 2, num_samples)
    market_data["network_load"] = np.random.normal(0.7, 0.1, num_samples)
    market_data["pending_tx_count"] = np.random.normal(10000, 2000, num_samples)
    
    # Create time series data (e.g., price history)
    time_series = np.zeros((num_samples, seq_length, 5))
    for i in range(num_samples):
        # Generate random walk for price
        price_series = np.cumsum(np.random.normal(0, 1, seq_length))
        # Normalize
        price_series = (price_series - np.mean(price_series)) / np.std(price_series)
        
        # Generate volume series (correlated with price changes)
        price_changes = np.diff(price_series, prepend=price_series[0])
        volume_series = np.abs(price_changes) * 10 + np.random.normal(5, 1, seq_length)
        
        # Generate other features
        liquidity_series = price_series * 2 + np.random.normal(0, 0.5, seq_length)
        volatility_series = np.abs(price_changes) + np.random.normal(0.02, 0.01, seq_length)
        sentiment_series = price_series * 0.5 + np.random.normal(0, 0.5, seq_length)
        
        # Combine into time series
        time_series[i, :, 0] = price_series
        time_series[i, :, 1] = volume_series
        time_series[i, :, 2] = liquidity_series
        time_series[i, :, 3] = volatility_series
        time_series[i, :, 4] = sentiment_series
    
    market_data["time_series"] = time_series
    
    # Create market state for RL
    market_state = np.zeros((num_samples, 128))
    for i in range(num_samples):
        # Combine various features into a state representation
        market_state[i, 0] = market_data["price"][i] / 100
        market_state[i, 1] = market_data["volume_24h"][i] / 1000000
        market_state[i, 2] = market_data["liquidity"][i] / 5000000
        market_state[i, 3] = market_data["volatility"][i] / 0.05
        market_state[i, 4] = market_data["market_cap"][i] / 1000000000
        market_state[i, 5] = market_data["tvl"][i] / 500000000
        market_state[i, 6] = market_data["fees_24h"][i] / 100000
        market_state[i, 7] = market_data["gas_price"][i] / 50
        market_state[i, 8] = market_data["block_time"][i] / 12
        market_state[i, 9] = market_data["network_load"][i] / 0.7
        market_state[i, 10] = market_data["pending_tx_count"][i] / 10000
        
        # Add time series summary statistics
        ts_data = time_series[i]
        market_state[i, 11:16] = np.mean(ts_data, axis=0)
        market_state[i, 16:21] = np.std(ts_data, axis=0)
        market_state[i, 21:26] = np.max(ts_data, axis=0)
        market_state[i, 26:31] = np.min(ts_data, axis=0)
        
        # Fill the rest with random noise
        market_state[i, 31:] = np.random.normal(0, 0.1, 128 - 31)
    
    market_data["market_state"] = market_state
    
    # Add some anomalies
    anomaly_indices = np.random.choice(num_samples, size=5, replace=False)
    for idx in anomaly_indices:
        # Make price extremely high
        market_data["price"][idx] *= 5
        # Make volume extremely low
        market_data["volume_24h"][idx] /= 10
        # Make volatility extremely high
        market_data["volatility"][idx] *= 8
        
        # Reflect these changes in the time series and state
        time_series[idx, :, 0] *= 3  # Price series
        market_state[idx, 0] *= 5    # Price state
        market_state[idx, 1] /= 10   # Volume state
        market_state[idx, 3] *= 8    # Volatility state
    
    return market_data

def create_mock_arbitrage_model():
    """Create a mock arbitrage model for demonstration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ArbitrageModel(device)
    
    # We don't need to initialize the full model for this example
    # Just mock the _extract_market_features method
    def mock_extract_features(market_data):
        # Convert numpy arrays to tensors
        features = []
        for key in ["price", "volume_24h", "liquidity", "volatility", 
                   "market_cap", "tvl", "fees_24h", "gas_price", 
                   "block_time", "network_load", "pending_tx_count"]:
            if key in market_data:
                features.append(market_data[key])
        
        # Stack features
        features_array = np.column_stack(features)
        return torch.tensor(features_array, dtype=torch.float32, device=device)
    
    # Replace the method with our mock
    model._extract_market_features = mock_extract_features
    
    return model

def main():
    """Main function demonstrating ML integration."""
    logger.info("Creating sample market data...")
    market_data = create_sample_market_data(num_samples=100, seq_length=32)
    
    logger.info("Creating mock arbitrage model...")
    base_model = create_mock_arbitrage_model()
    
    logger.info("Initializing ML integration...")
    ml_integration = MLModelIntegration(base_model=base_model)
    
    # Convert numpy arrays to tensors
    device = ml_integration.device
    market_features = torch.tensor(
        np.column_stack([
            market_data["price"], 
            market_data["volume_24h"],
            market_data["liquidity"],
            market_data["volatility"],
            market_data["market_cap"],
            market_data["tvl"],
            market_data["fees_24h"],
            market_data["gas_price"],
            market_data["block_time"],
            market_data["network_load"],
            market_data["pending_tx_count"]
        ]), 
        dtype=torch.float32, 
        device=device
    )
    
    time_series = torch.tensor(
        market_data["time_series"], 
        dtype=torch.float32, 
        device=device
    )
    
    market_state = torch.tensor(
        market_data["market_state"], 
        dtype=torch.float32, 
        device=device
    )
    
    # 1. Demonstrate VAE for anomaly detection
    logger.info("Detecting market anomalies with VAE...")
    is_anomaly, anomaly_scores = ml_integration.detect_market_anomalies(market_features)
    
    num_anomalies = is_anomaly.sum().item()
    logger.info(f"Detected {num_anomalies} anomalies")
    
    if num_anomalies > 0:
        anomaly_indices = torch.where(is_anomaly)[0].cpu().numpy()
        logger.info(f"Anomaly indices: {anomaly_indices}")
        logger.info(f"Anomaly scores: {anomaly_scores[is_anomaly].cpu().numpy()}")
    
    # 2. Demonstrate conditional VAE for market generation
    logger.info("Generating market samples with conditional VAE...")
    # Use the first market state as condition
    condition = market_features[0:1, :32]  # Take first 32 features as condition
    generated_samples = ml_integration.generate_market_samples(
        condition=condition, 
        num_samples=5
    )
    logger.info(f"Generated {generated_samples.shape[0]} market samples")
    
    # 3. Demonstrate hierarchical VAE for feature extraction
    logger.info("Extracting hierarchical features from time series...")
    hierarchical_features = ml_integration.extract_hierarchical_features(time_series)
    logger.info(f"Local features shape: {hierarchical_features['local_features'].shape}")
    logger.info(f"Global features shape: {hierarchical_features['global_features'].shape}")
    
    # 4. Demonstrate SAC for trading strategy optimization
    logger.info("Optimizing trading strategy with SAC...")
    # Use the first market state
    optimal_action = ml_integration.optimize_trading_strategy(
        state=market_state[0:1], 
        deterministic=True
    )
    logger.info(f"Optimal action: {optimal_action}")
    
    # 5. Demonstrate model-based RL for trajectory planning
    logger.info("Planning trading trajectory with model-based RL...")
    best_action, best_value = ml_integration.plan_trading_trajectory(
        state=market_state[0:1],
        horizon=5
    )
    logger.info(f"Best action from planning: {best_action}")
    logger.info(f"Best value from planning: {best_value}")
    
    # 6. Demonstrate attention for time series processing
    logger.info("Processing time series with multi-scale attention...")
    attended_time_series = ml_integration.process_time_series_with_attention(time_series)
    logger.info(f"Attended time series shape: {attended_time_series.shape}")
    
    # 7. Demonstrate cross-market attention
    logger.info("Analyzing cross-market relationships...")
    # Simulate multiple markets (reshape our time series)
    primary_market = time_series[0:1]  # First sample as primary market
    other_markets = time_series[1:6].unsqueeze(0)  # Next 5 samples as other markets
    
    enhanced_primary = ml_integration.analyze_cross_market_relationships(
        primary_market=primary_market,
        other_markets=other_markets
    )
    logger.info(f"Enhanced primary market shape: {enhanced_primary.shape}")
    
    # 8. Demonstrate hierarchical time attention
    logger.info("Processing hierarchical time data...")
    hierarchical_output = ml_integration.process_hierarchical_time_data(time_series[:10])
    logger.info(f"Hierarchical output shape: {hierarchical_output.shape}")
    
    # 9. Demonstrate enhancing base model output
    logger.info("Enhancing base model output...")
    # Create a mock base model output
    base_output = {
        "market_analysis": torch.randn(100, 32, device=device),
        "path_finding": torch.randn(100, 16, device=device),
        "risk_assessment": torch.randn(100, 8, device=device),
        "execution_strategy": torch.randn(100, 16, device=device),
        "confidence": torch.ones(100, 1, device=device) * 0.8
    }
    
    # Prepare market data in the expected format
    market_data_dict = {
        "price": market_data["price"],
        "volume_24h": market_data["volume_24h"],
        "liquidity": market_data["liquidity"],
        "volatility": market_data["volatility"],
        "market_cap": market_data["market_cap"],
        "tvl": market_data["tvl"],
        "fees_24h": market_data["fees_24h"],
        "gas_price": market_data["gas_price"],
        "block_time": market_data["block_time"],
        "network_load": market_data["network_load"],
        "pending_tx_count": market_data["pending_tx_count"],
        "time_series": market_data["time_series"],
        "market_state": market_data["market_state"]
    }
    
    enhanced_output = ml_integration.enhance_base_model_output(
        market_data=market_data_dict,
        base_output=base_output
    )
    
    # Check what new keys were added
    new_keys = set(enhanced_output.keys()) - set(base_output.keys())
    logger.info(f"New keys in enhanced output: {new_keys}")
    
    # Check confidence adjustment
    if "confidence_adjustment" in enhanced_output:
        confidence_adj = enhanced_output["confidence_adjustment"].cpu().numpy()
        logger.info(f"Confidence adjustment min: {confidence_adj.min()}, max: {confidence_adj.max()}")
    
    # 10. Save models
    logger.info("Saving models...")
    os.makedirs("models", exist_ok=True)
    ml_integration.save_models(save_dir="models")
    
    logger.info("ML integration demonstration completed successfully!")

if __name__ == "__main__":
    main() 