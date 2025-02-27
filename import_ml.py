from src.ml.ml_integration import MLModelIntegration

print("Successfully imported MLModelIntegration")
print("MLModelIntegration class details:", MLModelIntegration)

# Example of how to use the MLModelIntegration class
print("\nExample usage:")
print("ml_integration = MLModelIntegration(base_model=your_base_model)")
print("# Use VAE for anomaly detection")
print("is_anomaly, anomaly_scores = ml_integration.detect_market_anomalies(market_features)")
print("# Use SAC for strategy optimization")
print("optimal_action = ml_integration.optimize_trading_strategy(state=market_state)")
print("# Use attention for time series processing")
print("attended_features = ml_integration.process_time_series_with_attention(time_series)") 