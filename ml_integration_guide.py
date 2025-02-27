"""
Guide to Using MLModelIntegration in Your Blockchain Project
===========================================================

The MLModelIntegration class provides a unified interface for using advanced machine learning models:
1. Variational Autoencoders (VAE) for anomaly detection and market pattern learning
2. Soft Actor-Critic (SAC) for reinforcement learning and strategy optimization
3. Multi-scale attention mechanisms for improved time series understanding

This guide shows how to import and use the class in your project.
"""

# Import Instructions
print("=== Import Instructions ===")
print("To import the MLModelIntegration class, use:")
print("from src.ml.ml_integration import MLModelIntegration")
print()

# Initialization
print("=== Initialization ===")
print("Initialize the MLModelIntegration with your base model:")
print("""
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
""")
print()

# Key Features
print("=== Key Features ===")
print("1. Anomaly Detection with VAE:")
print("""
# Detect anomalies in market data
is_anomaly, anomaly_scores = ml_integration.detect_market_anomalies(market_features)

# Generate market samples conditioned on market state
samples = ml_integration.generate_market_samples(
    condition=market_condition, 
    num_samples=10
)

# Extract hierarchical features from time series
features = ml_integration.extract_hierarchical_features(time_series)
""")
print()

print("2. Strategy Optimization with SAC:")
print("""
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
""")
print()

print("3. Time Series Processing with Attention:")
print("""
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
""")
print()

print("4. Enhancing Base Model Output:")
print("""
# Enhance the output of your base model
enhanced_output = ml_integration.enhance_base_model_output(
    market_data=market_data,
    base_output=base_output
)
""")
print()

# Model Persistence
print("=== Model Persistence ===")
print("""
# Save models to disk
ml_integration.save_models(save_dir="models")

# Load models from disk
ml_integration.load_models(load_dir="models")
""")
print()

# Configuration
print("=== Configuration ===")
print("The MLModelIntegration class can be configured through a JSON file with the following structure:")
print("""
{
    "market_vae": {
        "input_dim": 64,
        "latent_dim": 16,
        "hidden_dims": [128, 64],
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
    "multi_scale_attention": {
        "d_model": 256,
        "num_heads": 8,
        "scales": [1, 4, 16],
        "dropout": 0.1
    },
    "anomaly_threshold": 3.0,
    "model_paths": {
        "vae": "models/market_vae.pt",
        "sac": "models/sac_model.pt",
        "attention": "models/attention_model.pt"
    }
}
""")
print()

# Example Script
print("=== Complete Example ===")
print("For a complete example of using MLModelIntegration, see:")
print("src/ml/examples/ml_integration_example.py")
print()

print("=== Dependencies ===")
print("To use MLModelIntegration, you need the following dependencies:")
print("- PyTorch")
print("- torch_geometric")
print("- NumPy")
print("- matplotlib (for visualization)")
print()

print("Note: If you encounter NumPy compatibility issues, try downgrading to NumPy 1.x:")
print("pip install numpy<2") 