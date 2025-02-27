# MLModelIntegration Guide

This guide explains how to import and use the `MLModelIntegration` class from the blockchain project.

## Overview

The `MLModelIntegration` class provides a unified interface for using advanced machine learning models:

1. **Variational Autoencoders (VAE)** for anomaly detection and market pattern learning
2. **Soft Actor-Critic (SAC)** for reinforcement learning and strategy optimization
3. **Multi-scale attention mechanisms** for improved time series understanding

## Installation Requirements

To use the `MLModelIntegration` class, you need the following dependencies:

```bash
# Core dependencies
pip install torch                # PyTorch for deep learning
pip install numpy<2              # NumPy (version 1.x to avoid compatibility issues)
pip install torch-geometric      # PyTorch Geometric for graph neural networks
pip install matplotlib           # For visualization (optional)
```

> **Note:** There are known compatibility issues with NumPy 2.x. If you encounter errors, downgrade to NumPy 1.x with `pip install numpy<2`.

## Importing the Class

```python
from src.ml.ml_integration import MLModelIntegration
```

## Initialization

```python
# Basic initialization
ml_integration = MLModelIntegration(
    base_model=your_base_model,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# With custom configuration
ml_integration = MLModelIntegration(
    base_model=your_base_model,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    config_path="path/to/config.json"
)
```

## Key Features

### 1. Anomaly Detection with VAE

```python
# Detect anomalies in market data
is_anomaly, anomaly_scores = ml_integration.detect_market_anomalies(market_features)

# Generate market samples conditioned on market state
samples = ml_integration.generate_market_samples(
    condition=market_condition, 
    num_samples=10
)

# Extract hierarchical features from time series
features = ml_integration.extract_hierarchical_features(time_series)
```

### 2. Strategy Optimization with SAC

```python
# Get optimal trading action
optimal_action = ml_integration.optimize_trading_strategy(
    state=market_state,
    deterministic=True
)

# Plan trading trajectory
best_action, best_value = ml_integration.plan_trading_trajectory(
    state=market_state,
    horizon=5
)
```

### 3. Time Series Processing with Attention

```python
# Process time series with multi-scale attention
attended_features = ml_integration.process_time_series_with_attention(
    time_series=time_series,
    mask=attention_mask  # Optional
)

# Analyze cross-market relationships
enhanced_market = ml_integration.analyze_cross_market_relationships(
    primary_market=primary_market_data,
    other_markets=other_markets_data
)

# Process hierarchical time data
hierarchical_output = ml_integration.process_hierarchical_time_data(time_series)
```

### 4. Enhancing Base Model Output

```python
# Enhance the output of your base model
enhanced_output = ml_integration.enhance_base_model_output(
    market_data=market_data,
    base_output=base_output
)
```

## Model Persistence

```python
# Save models to disk
ml_integration.save_models(save_dir="models")

# Load models from disk
ml_integration.load_models(load_dir="models")
```

## Configuration

The `MLModelIntegration` class can be configured through a JSON file with the following structure:

```json
{
    "market_vae": {
        "input_dim": 64,
        "latent_dim": 16,
        "hidden_dims": [128, 64],
        "dropout": 0.1
    },
    "conditional_vae": {
        "input_dim": 64,
        "condition_dim": 32,
        "latent_dim": 16,
        "hidden_dims": [128, 64],
        "dropout": 0.1
    },
    "hierarchical_vae": {
        "input_dim": 64,
        "local_latent_dim": 8,
        "global_latent_dim": 16,
        "sequence_length": 32,
        "hidden_dim": 128,
        "dropout": 0.1
    },
    "sac": {
        "state_dim": 128,
        "action_dim": 16,
        "hidden_dim": 256,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "lr": 3e-4,
        "buffer_size": 100000,
        "action_space": "continuous"
    },
    "model_based_rl": {
        "state_dim": 128,
        "action_dim": 16,
        "hidden_dim": 256,
        "model_hidden_dim": 512,
        "ensemble_size": 5,
        "horizon": 5,
        "gamma": 0.99,
        "learning_rate": 3e-4
    },
    "multi_scale_attention": {
        "d_model": 256,
        "num_heads": 8,
        "scales": [1, 4, 16],
        "dropout": 0.1
    },
    "temporal_attention": {
        "d_model": 256,
        "num_heads": 8,
        "max_len": 512,
        "dropout": 0.1,
        "time_decay": 0.1
    },
    "cross_market_attention": {
        "d_model": 256,
        "num_heads": 8,
        "dropout": 0.1
    },
    "market_encoder": {
        "input_dim": 64,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 8,
        "d_ff": 1024,
        "dropout": 0.1,
        "scales": [1, 4, 16]
    },
    "hierarchical_time_attention": {
        "input_dim": 64,
        "d_model": 256,
        "num_heads": 8,
        "hierarchy_levels": 3,
        "samples_per_level": 60,
        "dropout": 0.1
    },
    "anomaly_threshold": 3.0,
    "model_paths": {
        "vae": "models/market_vae.pt",
        "sac": "models/sac_model.pt",
        "attention": "models/attention_model.pt"
    }
}
```

## Complete Example

For a complete example of using `MLModelIntegration`, see:
`src/ml/examples/ml_integration_example.py`

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'torch_geometric'**
   - Install PyTorch Geometric: `pip install torch-geometric`

2. **NumPy compatibility issues**
   - Downgrade to NumPy 1.x: `pip install numpy<2`

3. **CUDA out of memory**
   - Reduce batch sizes in configuration
   - Use CPU instead: `device=torch.device("cpu")`

4. **Model loading errors**
   - Ensure model paths in configuration are correct
   - Check if model files exist in the specified directories

### Getting Help

If you encounter issues not covered here, check:
- The docstrings in the `ml_integration.py` file
- The example script in `src/ml/examples/ml_integration_example.py`
- The project documentation in `src/ml/README.md` 