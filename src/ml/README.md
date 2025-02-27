# Advanced Machine Learning Models for Blockchain

This directory contains advanced machine learning models for blockchain data analysis, market prediction, and strategy optimization.

## New Models

We've implemented three key advanced machine learning approaches:

### 1. Variational Autoencoders (VAE)

Located in `vae_models.py`, these models are used for:
- Market data representation learning
- Anomaly detection in market patterns
- Generating synthetic market data
- Hierarchical feature extraction from time series

Key implementations:
- `MarketVAE`: Standard VAE for market data representation and anomaly detection
- `ConditionalMarketVAE`: Generates market data conditioned on specific market states
- `HierarchicalMarketVAE`: Models market data at multiple time scales with local and global latent variables

### 2. Reinforcement Learning with Soft Actor-Critic (SAC)

Located in `rl_models.py`, these models are used for:
- Trading strategy optimization
- Market action planning
- Risk-aware decision making

Key implementations:
- `SoftActorCritic`: Off-policy actor-critic algorithm with entropy regularization
- `ModelBasedRL`: Combines world modeling with planning for more efficient learning
- `ReplayBuffer`: Experience replay for stable RL training

### 3. Multi-Scale Attention Mechanisms

Located in `attention_models.py`, these models are used for:
- Improved time series understanding
- Capturing relationships between different markets
- Processing data at multiple temporal resolutions

Key implementations:
- `MultiScaleAttention`: Processes time series at different temporal resolutions
- `TemporallyWeightedAttention`: Weighs time steps differently based on temporal distance
- `CrossMarketAttention`: Analyzes relationships between different markets/tokens
- `HierarchicalTimeAttention`: Organizes time series into hierarchical structures

## Integration

The `ml_integration.py` file provides a unified interface for using these advanced models with the existing architecture. The `MLModelIntegration` class:

1. Initializes all models with configurable parameters
2. Provides methods for using each model type
3. Enhances the output of the base model with advanced ML techniques
4. Handles model saving and loading

## Example Usage

See `examples/ml_integration_example.py` for a complete demonstration of how to use these models.

Basic usage:

```python
from src.ml.model import ArbitrageModel
from src.ml.ml_integration import MLModelIntegration

# Initialize with base model
base_model = ArbitrageModel(device)
ml_integration = MLModelIntegration(base_model=base_model)

# Use VAE for anomaly detection
is_anomaly, anomaly_scores = ml_integration.detect_market_anomalies(market_features)

# Use SAC for strategy optimization
optimal_action = ml_integration.optimize_trading_strategy(state=market_state)

# Use attention for time series processing
attended_features = ml_integration.process_time_series_with_attention(time_series)

# Enhance base model output
enhanced_output = ml_integration.enhance_base_model_output(
    market_data=market_data,
    base_output=base_output
)
```

## Configuration

The models can be configured through a JSON configuration file passed to the `MLModelIntegration` constructor:

```python
ml_integration = MLModelIntegration(
    base_model=base_model,
    config_path="config/ml_config.json"
)
```

See the `_load_config` method in `ml_integration.py` for the default configuration structure.

## Model Persistence

Models can be saved and loaded using:

```python
# Save models
ml_integration.save_models(save_dir="models")

# Load models
ml_integration.load_models(load_dir="models")
``` 