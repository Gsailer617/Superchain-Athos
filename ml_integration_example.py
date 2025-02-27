"""
Simple Example of Importing and Using MLModelIntegration
=======================================================

This script demonstrates how to import and use the MLModelIntegration class
with proper error handling for dependencies.
"""

import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function demonstrating MLModelIntegration import and usage."""
    
    # Check for dependencies
    dependencies_ok = True
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        logger.error("PyTorch is not installed. Please install it with: pip install torch")
        dependencies_ok = False
    
    try:
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")
        
        # Check for NumPy 2.x compatibility issues
        if np.__version__.startswith('2'):
            logger.warning("NumPy 2.x detected. This may cause compatibility issues with some modules.")
            logger.warning("Consider downgrading to NumPy 1.x with: pip install numpy<2")
    except ImportError:
        logger.error("NumPy is not installed. Please install it with: pip install numpy<2")
        dependencies_ok = False
    
    try:
        import torch_geometric
        logger.info(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        logger.error("PyTorch Geometric is not installed. Please install it with: pip install torch-geometric")
        dependencies_ok = False
    
    if not dependencies_ok:
        logger.error("Missing dependencies. Please install them before continuing.")
        return
    
    # Try to import MLModelIntegration
    try:
        from src.ml.ml_integration import MLModelIntegration
        logger.info("Successfully imported MLModelIntegration")
        
        # Import ArbitrageModel for demonstration
        try:
            from src.ml.model import ArbitrageModel
            logger.info("Successfully imported ArbitrageModel")
            
            # Create a mock ArbitrageModel
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            base_model = ArbitrageModel(device)
            logger.info(f"Created ArbitrageModel on device: {device}")
            
            # Initialize MLModelIntegration
            ml_integration = MLModelIntegration(
                base_model=base_model,
                device=device
            )
            logger.info("Successfully initialized MLModelIntegration")
            
            # Show available methods
            logger.info("Available methods in MLModelIntegration:")
            for method_name in dir(ml_integration):
                if not method_name.startswith('_'):
                    logger.info(f"  - {method_name}")
            
        except ImportError as e:
            logger.error(f"Could not import ArbitrageModel: {e}")
            logger.info("Creating a mock base model for demonstration...")
            
            # Create a mock base model
            class MockBaseModel:
                def __init__(self, device):
                    self.device = device
                
                def _extract_market_features(self, market_data):
                    # Mock implementation
                    return torch.zeros((1, 64), device=self.device)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            base_model = MockBaseModel(device)
            logger.info(f"Created MockBaseModel on device: {device}")
            
            # Initialize MLModelIntegration with mock model
            try:
                ml_integration = MLModelIntegration(
                    base_model=base_model,
                    device=device
                )
                logger.info("Successfully initialized MLModelIntegration with mock model")
            except Exception as e:
                logger.error(f"Error initializing MLModelIntegration with mock model: {e}")
        
    except ImportError as e:
        logger.error(f"Could not import MLModelIntegration: {e}")
        logger.info("Please ensure that the src/ml directory is in your Python path")
        logger.info("You can add it with: sys.path.append('/path/to/project')")
    except Exception as e:
        logger.error(f"Unexpected error importing MLModelIntegration: {e}")
    
    logger.info("Example script completed")

if __name__ == "__main__":
    main() 