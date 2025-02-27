import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Print the current working directory and Python path
print("Current working directory:", os.getcwd())
print("Python path:", sys.path)

# Try to list the src/ml directory
try:
    print("\nContents of src/ml directory:")
    for file in os.listdir("src/ml"):
        print(f"  - {file}")
except Exception as e:
    print(f"Error listing directory: {e}")

print("\nAttempting to import MLModelIntegration directly...")
try:
    # Try to import the file directly without dependencies
    with open("src/ml/ml_integration.py", "r") as f:
        print("Successfully opened ml_integration.py")
        # Print the first few lines to confirm it's the right file
        lines = f.readlines()
        print("\nFirst 10 lines of ml_integration.py:")
        for i, line in enumerate(lines[:10]):
            print(f"{i+1}: {line.strip()}")
        
        print("\nMLModelIntegration class definition found in file:", "MLModelIntegration" in "".join(lines))
except Exception as e:
    print(f"Error reading file: {e}")

print("\nTo use MLModelIntegration in your code:")
print("from src.ml.ml_integration import MLModelIntegration")
print("\nExample usage:")
print("ml_integration = MLModelIntegration(base_model=your_base_model)")
print("is_anomaly, anomaly_scores = ml_integration.detect_market_anomalies(market_features)") 