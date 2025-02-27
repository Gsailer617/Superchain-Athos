"""Integration module for advanced machine learning models in the blockchain system.

This module provides a unified interface for using the advanced machine learning models:
1. Variational Autoencoders (VAE) for anomaly detection and market pattern learning
2. Soft Actor-Critic (SAC) and Distributional RL for reinforcement learning and strategy optimization
3. Transformer and Multi-head attention mechanisms for improved time series understanding

The integration ensures these models work seamlessly with the existing architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt

# Import the advanced models
from .vae_models import MarketVAE, ConditionalMarketVAE, HierarchicalMarketVAE
from .rl_models import (
    SoftActorCritic, ModelBasedRL, ReplayBuffer, 
    DistributionalSoftActorCritic, RainbowDQN, PrioritizedReplayBuffer
)
from .attention_models import (
    MultiScaleAttention, TemporallyWeightedAttention, 
    CrossMarketAttention, MarketAttentionEncoder,
    HierarchicalTimeAttention, FinancialTimeTransformer,
    MultiHeadCrossMarketAttention
)

# Import existing architecture components
from .model import MarketDataType, ModelOutputType, ArbitrageModel

logger = logging.getLogger(__name__)

class MLModelIntegration:
    """Integration class for advanced machine learning models.
    
    This class provides a unified interface for using the advanced ML models
    within the existing blockchain architecture.
    """
    
    def __init__(
        self,
        base_model: ArbitrageModel,
        device: torch.device = None,
        config_path: Optional[str] = None
    ):
        """Initialize the integration with the base arbitrage model.
        
        Args:
            base_model: The existing arbitrage model
            device: Computation device (CPU/GPU)
            config_path: Path to configuration file for model parameters
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.base_model = base_model
        
        # Load configuration if provided
        self.config = self._load_config(config_path)
        
        # Initialize models
        self._init_vae_models()
        self._init_rl_models()
        self._init_attention_models()
        
        # Track anomalies detected
        self.recent_anomalies = []
        self.anomaly_threshold = self.config.get("anomaly_threshold", 3.0)
        
        # Risk metrics
        self.risk_level = self.config.get("risk_level", 0.05)
        
        logger.info(f"ML integration initialized on device: {self.device}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            # VAE configuration
            "market_vae": {
                "input_dim": 64,
                "latent_dim": 16,
                "hidden_dims": [128, 64],
                "dropout": 0.1,
                "use_residual": True,
                "activation": "relu"
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
            
            # RL configuration
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
            "distributional_sac": {
                "state_dim": 128,
                "action_dim": 16,
                "hidden_dims": [256, 256],
                "atom_size": 51,
                "v_min": -10.0,
                "v_max": 10.0,
                "alpha": 0.2,
                "gamma": 0.99,
                "tau": 0.005,
                "auto_entropy_tuning": True
            },
            "rainbow_dqn": {
                "state_dim": 128,
                "action_dim": 16,
                "hidden_dims": [256, 256],
                "atom_size": 51,
                "v_min": -10.0,
                "v_max": 10.0,
                "noisy": True
            },
            
            # Attention configuration
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
            "financial_transformer": {
                "input_dim": 64,
                "d_model": 512,
                "nhead": 8,
                "num_encoder_layers": 6,
                "dim_feedforward": 2048,
                "dropout": 0.1,
                "activation": "gelu",
                "max_seq_length": 1000,
                "learned_pos_encoding": True,
                "forecast_horizon": 10
            },
            "cross_market_multi_head": {
                "market_dim": 64,
                "num_markets": 5,
                "embed_dim": 256,
                "num_heads": 8,
                "dropout": 0.1
            },
            
            # General configuration
            "anomaly_threshold": 3.0,
            "risk_level": 0.05,  # For CVaR computations
            "model_paths": {
                "vae": "models/market_vae.pt",
                "sac": "models/sac_model.pt",
                "dist_sac": "models/dist_sac_model.pt",
                "rainbow": "models/rainbow_dqn.pt",
                "attention": "models/attention_model.pt",
                "transformer": "models/transformer_model.pt"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Update default config with loaded values
                    self._update_nested_dict(default_config, loaded_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
                
        return default_config
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Update nested dictionary with another dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _init_vae_models(self):
        """Initialize VAE models."""
        # Standard Market VAE
        market_vae_config = self.config["market_vae"]
        self.market_vae = MarketVAE(
            input_dim=market_vae_config["input_dim"],
            latent_dim=market_vae_config["latent_dim"],
            hidden_dims=market_vae_config["hidden_dims"],
            dropout=market_vae_config["dropout"],
            use_residual=market_vae_config["use_residual"],
            activation=market_vae_config["activation"]
        ).to(self.device)
        
        # Conditional Market VAE
        cond_vae_config = self.config["conditional_vae"]
        self.conditional_vae = ConditionalMarketVAE(
            input_dim=cond_vae_config["input_dim"],
            condition_dim=cond_vae_config["condition_dim"],
            latent_dim=cond_vae_config["latent_dim"],
            hidden_dims=cond_vae_config["hidden_dims"],
            dropout=cond_vae_config["dropout"]
        ).to(self.device)
        
        # Hierarchical Market VAE
        hier_vae_config = self.config["hierarchical_vae"]
        self.hierarchical_vae = HierarchicalMarketVAE(
            input_dim=hier_vae_config["input_dim"],
            local_latent_dim=hier_vae_config["local_latent_dim"],
            global_latent_dim=hier_vae_config["global_latent_dim"],
            sequence_length=hier_vae_config["sequence_length"],
            hidden_dim=hier_vae_config["hidden_dim"],
            dropout=hier_vae_config["dropout"]
        ).to(self.device)
        
        # Try to load pre-trained models if available
        vae_path = self.config.get("model_paths", {}).get("vae")
        if vae_path and os.path.exists(vae_path):
            try:
                checkpoint = torch.load(vae_path, map_location=self.device)
                self.market_vae.load_state_dict(checkpoint.get("market_vae", {}))
                self.conditional_vae.load_state_dict(checkpoint.get("conditional_vae", {}))
                self.hierarchical_vae.load_state_dict(checkpoint.get("hierarchical_vae", {}))
                logger.info(f"Loaded VAE models from {vae_path}")
            except Exception as e:
                logger.warning(f"Could not load VAE models: {e}")
    
    def _init_rl_models(self):
        """Initialize reinforcement learning models."""
        # Soft Actor-Critic
        sac_config = self.config["sac"]
        self.sac = SoftActorCritic(
            state_dim=sac_config["state_dim"],
            action_dim=sac_config["action_dim"],
            hidden_dim=sac_config["hidden_dim"],
            gamma=sac_config["gamma"],
            tau=sac_config["tau"],
            alpha=sac_config["alpha"],
            lr=sac_config["lr"],
            buffer_size=sac_config["buffer_size"],
            action_space=sac_config["action_space"],
            device=self.device
        )
        
        # Model-Based RL
        mbrl_config = self.config["model_based_rl"]
        self.model_based_rl = ModelBasedRL(
            state_dim=mbrl_config["state_dim"],
            action_dim=mbrl_config["action_dim"],
            hidden_dim=mbrl_config["hidden_dim"],
            model_hidden_dim=mbrl_config["model_hidden_dim"],
            ensemble_size=mbrl_config["ensemble_size"],
            horizon=mbrl_config["horizon"],
            gamma=mbrl_config["gamma"],
            learning_rate=mbrl_config["learning_rate"],
            device=self.device
        )
        
        # Distributional Soft Actor-Critic
        dist_sac_config = self.config["distributional_sac"]
        self.dist_sac = DistributionalSoftActorCritic(
            state_dim=dist_sac_config["state_dim"],
            action_dim=dist_sac_config["action_dim"],
            hidden_dims=dist_sac_config["hidden_dims"],
            atom_size=dist_sac_config["atom_size"],
            v_min=dist_sac_config["v_min"],
            v_max=dist_sac_config["v_max"],
            alpha=dist_sac_config["alpha"],
            gamma=dist_sac_config["gamma"],
            tau=dist_sac_config["tau"],
            auto_entropy_tuning=dist_sac_config["auto_entropy_tuning"],
            device=self.device
        )
        
        # Rainbow DQN
        rainbow_config = self.config["rainbow_dqn"]
        self.rainbow_dqn = RainbowDQN(
            state_dim=rainbow_config["state_dim"],
            action_dim=rainbow_config["action_dim"],
            hidden_dims=rainbow_config["hidden_dims"],
            atom_size=rainbow_config["atom_size"],
            v_min=rainbow_config["v_min"],
            v_max=rainbow_config["v_max"],
            noisy=rainbow_config["noisy"],
            device=self.device
        )
        
        # Prioritized Replay Buffer for Rainbow DQN
        self.prioritized_buffer = PrioritizedReplayBuffer(
            capacity=sac_config["buffer_size"],
            alpha=0.6,
            beta=0.4
        )
        
        # Try to load pre-trained models if available
        rl_path = self.config.get("model_paths", {}).get("sac")
        if rl_path and os.path.exists(rl_path):
            try:
                self.sac.load(rl_path)
                logger.info(f"Loaded SAC model from {rl_path}")
            except Exception as e:
                logger.warning(f"Could not load SAC model: {e}")
                
        # Try to load distributional SAC model
        dist_sac_path = self.config.get("model_paths", {}).get("dist_sac")
        if dist_sac_path and os.path.exists(dist_sac_path):
            try:
                checkpoint = torch.load(dist_sac_path, map_location=self.device)
                self.dist_sac.load_state_dict(checkpoint)
                logger.info(f"Loaded Distributional SAC model from {dist_sac_path}")
            except Exception as e:
                logger.warning(f"Could not load Distributional SAC model: {e}")
                
        # Try to load Rainbow DQN model
        rainbow_path = self.config.get("model_paths", {}).get("rainbow")
        if rainbow_path and os.path.exists(rainbow_path):
            try:
                checkpoint = torch.load(rainbow_path, map_location=self.device)
                self.rainbow_dqn.load_state_dict(checkpoint)
                logger.info(f"Loaded Rainbow DQN model from {rainbow_path}")
            except Exception as e:
                logger.warning(f"Could not load Rainbow DQN model: {e}")
    
    def _init_attention_models(self):
        """Initialize attention models."""
        # Multi-Scale Attention Model
        msa_config = self.config["multi_scale_attention"]
        self.multi_scale_attention = MultiScaleAttention(
            input_dim=msa_config["input_dim"],
            hidden_dim=msa_config["hidden_dim"],
            output_dim=msa_config["output_dim"],
            num_heads=msa_config["num_heads"],
            num_scales=msa_config["scales"],
            dropout=msa_config["dropout"],
            device=self.device
        )
        
        # Temporal Attention
        temp_config = self.config["temporal_attention"]
        self.temporal_attention = TemporallyWeightedAttention(
            d_model=temp_config["d_model"],
            num_heads=temp_config["num_heads"],
            max_len=temp_config["max_len"],
            dropout=temp_config["dropout"],
            time_decay=temp_config["time_decay"]
        ).to(self.device)
        
        # Cross-Market Attention
        cm_config = self.config["cross_market_attention"]
        self.cross_market_attention = CrossMarketAttention(
            num_markets=cm_config["num_markets"],
            feature_dim=cm_config["feature_dim"],
            hidden_dim=cm_config["hidden_dim"],
            num_heads=cm_config["num_heads"],
            dropout=cm_config["dropout"],
            device=self.device
        )
        
        # Market Attention Encoder
        mae_config = self.config["market_encoder"]
        self.market_encoder = MarketAttentionEncoder(
            input_dim=mae_config["input_dim"],
            hidden_dim=mae_config["hidden_dim"],
            num_layers=mae_config["num_layers"],
            num_heads=mae_config["num_heads"],
            dropout=mae_config["dropout"],
            device=self.device
        )
        
        # Hierarchical Time Attention
        hta_config = self.config["hierarchical_time_attention"]
        self.hierarchical_time_attention = HierarchicalTimeAttention(
            input_dim=hta_config["input_dim"],
            hidden_dim=hta_config["hidden_dim"],
            output_dim=hta_config["output_dim"],
            num_heads=hta_config["num_heads"],
            dropout=hta_config["dropout"],
            device=self.device
        )
        
        # Financial Time Transformer
        ft_config = self.config["financial_transformer"]
        self.financial_transformer = FinancialTimeTransformer(
            input_dim=ft_config["input_dim"],
            d_model=ft_config["d_model"],
            nhead=ft_config["nhead"],
            num_encoder_layers=ft_config["num_encoder_layers"],
            dim_feedforward=ft_config["dim_feedforward"],
            dropout=ft_config["dropout"],
            activation=ft_config["activation"],
            device=self.device
        )
        
        # Multi-Head Cross-Market Attention
        mhcm_config = self.config["cross_market_multi_head"]
        self.multi_head_cross_market = MultiHeadCrossMarketAttention(
            num_markets=mhcm_config["num_markets"],
            feature_dim=mhcm_config["feature_dim"],
            num_heads=mhcm_config["num_heads"],
            dropout=mhcm_config["dropout"],
            use_linear_projection=mhcm_config["use_linear_projection"],
            use_layer_norm=mhcm_config["use_layer_norm"],
            device=self.device
        )
        
        # Try to load pre-trained models if available
        for model_name, model in [
            ("multi_scale_attention", self.multi_scale_attention),
            ("temporal_attention", self.temporal_attention),
            ("cross_market_attention", self.cross_market_attention),
            ("market_encoder", self.market_encoder),
            ("hierarchical_time_attention", self.hierarchical_time_attention),
            ("financial_transformer", self.financial_transformer),
            ("multi_head_cross_market", self.multi_head_cross_market)
        ]:
            model_path = self.config.get("model_paths", {}).get(model_name)
            if model_path and os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model.load_state_dict(checkpoint)
                    logger.info(f"Loaded {model_name} model from {model_path}")
            except Exception as e:
                    logger.warning(f"Could not load {model_name} model: {e}")
    
    def detect_market_anomalies(self, market_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect anomalies in market data using VAE.
        
        Args:
            market_data: Tensor of market data [batch_size, features]
            
        Returns:
            is_anomaly: Boolean tensor indicating anomalies
            anomaly_scores: Anomaly scores (z-scores)
        """
        self.market_vae.eval()
        with torch.no_grad():
            is_anomaly, anomaly_scores = self.market_vae.detect_anomalies(
                market_data, 
                threshold_multiplier=self.anomaly_threshold
            )
            
            # Track recent anomalies for reporting
            if is_anomaly.any():
                anomaly_indices = torch.where(is_anomaly)[0].cpu().numpy()
                for idx in anomaly_indices:
                    self.recent_anomalies.append({
                        "score": anomaly_scores[idx].item(),
                        "timestamp": time.time(),
                        "data_idx": idx
                    })
                # Keep only recent anomalies (last 100)
                self.recent_anomalies = self.recent_anomalies[-100:]
                
            return is_anomaly, anomaly_scores
    
    def generate_market_samples(
        self, 
        condition: torch.Tensor, 
        num_samples: int = 10
    ) -> torch.Tensor:
        """Generate market samples conditioned on market state.
        
        Args:
            condition: Market condition tensor
            num_samples: Number of samples to generate
            
        Returns:
            Generated market samples
        """
        self.conditional_vae.eval()
        with torch.no_grad():
            samples = self.conditional_vae.sample(
                num_samples=num_samples,
                condition=condition,
                device=self.device
            )
            return samples
    
    def extract_hierarchical_features(self, time_series: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract hierarchical features from time series data using hierarchical VAE.
        
        Args:
            time_series: Time series data [batch_size, sequence_length, features]
            
        Returns:
            Dictionary with local and global latent representations
        """
        self.hierarchical_vae.eval()
        with torch.no_grad():
            outputs = self.hierarchical_vae(time_series)
            return {
                "local_features": outputs["local_z"],
                "global_features": outputs["global_z"]
            }
    
    def optimize_trading_strategy(
        self, 
        state: torch.Tensor, 
        deterministic: bool = True
    ) -> torch.Tensor:
        """Optimize trading strategy using SAC.
        
        Args:
            state: Current market state
            deterministic: Whether to use deterministic policy
            
        Returns:
            Optimal action
        """
        return self.sac.act(state, deterministic=deterministic)
    
    def plan_trading_trajectory(
        self, 
        state: torch.Tensor, 
        horizon: int = 5
    ) -> Tuple[torch.Tensor, float]:
        """Plan trading trajectory using model-based RL.
        
        Args:
            state: Current market state
            horizon: Planning horizon
            
        Returns:
            best_action: First action in the best trajectory
            best_value: Predicted value of the best trajectory
        """
        return self.model_based_rl.plan_trajectory(state, horizon=horizon)
    
    def process_time_series_with_attention(
        self, 
        time_series: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Process time series data with multi-scale attention.
        
        Args:
            time_series: Time series data [batch_size, sequence_length, features]
            mask: Optional attention mask
            
        Returns:
            Processed time series with attention applied
        """
        # First encode the time series
        encoded = self.market_encoder(time_series, mask)
        
        # Apply multi-scale attention
        attended = self.multi_scale_attention(encoded, mask)
        
        return attended
    
    def analyze_cross_market_relationships(
        self,
        market_features: Union[List[torch.Tensor], torch.Tensor],
        return_attention_weights: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Analyze cross-market relationships using advanced attention mechanisms.
        
        Args:
            market_features: Features from multiple markets
                If tensor, shape: [batch_size, num_markets, seq_len, feature_dim]
                If list, list of tensors with shape [batch_size, seq_len, feature_dim]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Dictionary with market relationships and optionally attention weights
        """
        # Convert list to tensor if necessary
        if isinstance(market_features, list):
            # Convert each tensor to the right shape and device
            processed_features = []
            for i, features in enumerate(market_features):
                if not isinstance(features, torch.Tensor):
                    features = torch.tensor(features, dtype=torch.float32)
                # Move to device
                features = features.to(self.device)
                # Add batch dimension if needed
                if features.dim() == 2:
                    features = features.unsqueeze(0)
                processed_features.append(features)
                
            # Stack into single tensor [batch, num_markets, seq_len, feature_dim]
            market_features = torch.stack(processed_features, dim=1)
        elif not isinstance(market_features, torch.Tensor):
            market_features = torch.tensor(market_features, dtype=torch.float32, device=self.device)
            
        # Process with multi-head cross market attention
        outputs, attn_weights = self.multi_head_cross_market(
            market_features, 
            return_attention=True
        )
        
        # Process with standard cross market attention for comparison
        std_outputs, std_attn_weights = self.cross_market_attention(
            market_features,
            return_attention=True
        )
        
        result = {
            "multi_head_market_features": outputs,
            "standard_market_features": std_outputs,
        }
        
        # Add attention weights if requested
        if return_attention_weights:
            result["multi_head_attention_weights"] = attn_weights
            result["standard_attention_weights"] = std_attn_weights
            
            # Compute correlation matrix from attention weights
            # Shape: [batch_size, num_markets, num_markets]
            batch_size, num_markets = attn_weights.shape[0], attn_weights.shape[1]
            correlation_matrix = torch.zeros(batch_size, num_markets, num_markets, device=self.device)
            
            # Average attention weights across heads
            avg_attn = attn_weights.mean(dim=1)  # [batch_size, num_markets, num_markets]
            
            # Normalize to correlation-like values (-1 to 1)
            for b in range(batch_size):
                # Min-max normalize to [0, 1]
                norm_attn = (avg_attn[b] - avg_attn[b].min()) / (avg_attn[b].max() - avg_attn[b].min() + 1e-8)
                # Scale to [-1, 1]
                correlation_matrix[b] = 2 * norm_attn - 1
                
            result["market_correlation_matrix"] = correlation_matrix
            
        return result
    
    def process_hierarchical_time_data(self, time_series: torch.Tensor) -> torch.Tensor:
        """Process hierarchical time series data.
        
        Args:
            time_series: Time series data [batch_size, sequence_length, features]
            
        Returns:
            Processed time series with hierarchical attention applied
        """
        return self.hierarchical_time_attention(time_series)
    
    def enhance_base_model_output(
        self, 
        market_data: MarketDataType,
        base_output: ModelOutputType
    ) -> ModelOutputType:
        """Enhance the output of the base model with advanced ML techniques.
        
        Args:
            market_data: Market data input
            base_output: Output from the base arbitrage model
            
        Returns:
            Enhanced model output
        """
        enhanced_output = base_output.copy()
        
        try:
            # Extract features from market data
            market_features = self._prepare_market_features(market_data)
            
            # 1. Detect anomalies
            is_anomaly, anomaly_scores = self.detect_market_anomalies(market_features)
            
            # 2. Process time series with attention if available
            if "time_series" in market_data and market_data["time_series"] is not None:
                time_series = self._prepare_time_series(market_data["time_series"])
                attended_features = self.process_time_series_with_attention(time_series)
                
                # Enhance market analysis with attended features
                if "market_analysis" in enhanced_output:
                    enhanced_output["market_analysis"] = torch.cat([
                        enhanced_output["market_analysis"],
                        attended_features.mean(dim=1)  # Aggregate over time
                    ], dim=1)
            
            # 3. Optimize execution strategy with RL if applicable
            if "execution_strategy" in enhanced_output and "market_state" in market_data:
                market_state = self._prepare_market_state(market_data["market_state"])
                
                # Get optimal action from SAC
                optimal_action = self.optimize_trading_strategy(market_state)
                
                # Incorporate optimal action into execution strategy
                enhanced_output["execution_strategy"] = (
                    enhanced_output["execution_strategy"] * 0.7 + 
                    torch.tensor(optimal_action, device=self.device) * 0.3
                )
            
            # 4. Add anomaly information to output
            enhanced_output["anomaly_detected"] = is_anomaly
            enhanced_output["anomaly_scores"] = anomaly_scores
            
            # 5. Adjust confidence based on anomaly detection
            if "confidence" in enhanced_output:
                # Reduce confidence for anomalous data
                confidence_adjustment = torch.ones_like(enhanced_output["confidence"])
                if is_anomaly.any():
                    for i, is_anom in enumerate(is_anomaly):
                        if is_anom:
                            # Scale down confidence based on anomaly score
                            adjustment = 1.0 - min(0.5, anomaly_scores[i].item() / 10.0)
                            confidence_adjustment[i] = adjustment
                
                enhanced_output["confidence"] = enhanced_output["confidence"] * confidence_adjustment
                enhanced_output["confidence_adjustment"] = confidence_adjustment
            
        except Exception as e:
            logger.error(f"Error enhancing model output: {e}")
            
        return enhanced_output
    
    def _prepare_market_features(self, market_data: MarketDataType) -> torch.Tensor:
        """Prepare market features for ML models."""
        # Extract numerical features from market data
        features = []
        
        # Add price, volume, liquidity, etc.
        if hasattr(market_data, "price"):
            features.append(market_data.price)
        if hasattr(market_data, "volume_24h"):
            features.append(market_data.volume_24h)
        if hasattr(market_data, "liquidity"):
            features.append(market_data.liquidity)
        if hasattr(market_data, "volatility"):
            features.append(market_data.volatility)
        if hasattr(market_data, "market_cap"):
            features.append(market_data.market_cap)
        if hasattr(market_data, "tvl"):
            features.append(market_data.tvl)
        if hasattr(market_data, "fees_24h"):
            features.append(market_data.fees_24h)
        if hasattr(market_data, "gas_price"):
            features.append(market_data.gas_price)
        if hasattr(market_data, "block_time"):
            features.append(market_data.block_time)
        if hasattr(market_data, "network_load"):
            features.append(market_data.network_load)
        if hasattr(market_data, "pending_tx_count"):
            features.append(market_data.pending_tx_count)
        
        # If market_data is already a tensor or has a tensor representation
        if isinstance(market_data, torch.Tensor):
            return market_data
        elif hasattr(market_data, "to_tensor") and callable(getattr(market_data, "to_tensor")):
            return market_data.to_tensor().to(self.device)
        
        # Convert list to tensor if we extracted features
        if features:
            return torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Fallback: try to use the base model's feature extraction
        try:
            return self.base_model._extract_market_features(market_data).to(self.device)
        except:
            logger.warning("Could not extract market features, using dummy tensor")
            return torch.zeros((1, self.config["market_vae"]["input_dim"]), device=self.device)
    
    def _prepare_time_series(self, time_series_data: Any) -> torch.Tensor:
        """Prepare time series data for attention models."""
        if isinstance(time_series_data, torch.Tensor):
            return time_series_data.to(self.device)
        elif isinstance(time_series_data, np.ndarray):
            return torch.tensor(time_series_data, dtype=torch.float32, device=self.device)
        else:
            logger.warning("Unsupported time series data type, using dummy tensor")
            return torch.zeros((1, 32, self.config["market_encoder"]["input_dim"]), device=self.device)
    
    def _prepare_market_state(self, market_state: Any) -> torch.Tensor:
        """Prepare market state for RL models."""
        if isinstance(market_state, torch.Tensor):
            return market_state.to(self.device)
        elif isinstance(market_state, np.ndarray):
            return torch.tensor(market_state, dtype=torch.float32, device=self.device)
        else:
            logger.warning("Unsupported market state data type, using dummy tensor")
            return torch.zeros((1, self.config["sac"]["state_dim"]), device=self.device)
    
    def save_models(self, save_dir: str = "models"):
        """Save all models to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save VAE models
        vae_path = os.path.join(save_dir, "market_vae.pt")
        torch.save({
            "market_vae": self.market_vae.state_dict(),
            "conditional_vae": self.conditional_vae.state_dict(),
            "hierarchical_vae": self.hierarchical_vae.state_dict()
        }, vae_path)
        
        # Save SAC model
        sac_path = os.path.join(save_dir, "sac_model.pt")
        self.sac.save(sac_path)
        
        # Save attention models
        attention_path = os.path.join(save_dir, "attention_model.pt")
        torch.save({
            "multi_scale_attention": self.multi_scale_attention.state_dict(),
            "temporal_attention": self.temporal_attention.state_dict(),
            "cross_market_attention": self.cross_market_attention.state_dict(),
            "market_encoder": self.market_encoder.state_dict(),
            "hierarchical_time_attention": self.hierarchical_time_attention.state_dict(),
            "financial_transformer": self.financial_transformer.state_dict(),
            "cross_market_multi_head": self.multi_head_cross_market.state_dict()
        }, attention_path)
        
        logger.info(f"Saved all models to {save_dir}")
        
    def load_models(self, load_dir: str = "models"):
        """Load all models from disk."""
        # Load VAE models
        vae_path = os.path.join(load_dir, "market_vae.pt")
        if os.path.exists(vae_path):
            checkpoint = torch.load(vae_path, map_location=self.device)
            self.market_vae.load_state_dict(checkpoint["market_vae"])
            self.conditional_vae.load_state_dict(checkpoint["conditional_vae"])
            self.hierarchical_vae.load_state_dict(checkpoint["hierarchical_vae"])
            logger.info(f"Loaded VAE models from {vae_path}")
        
        # Load SAC model
        sac_path = os.path.join(load_dir, "sac_model.pt")
        if os.path.exists(sac_path):
            self.sac.load(sac_path)
            logger.info(f"Loaded SAC model from {sac_path}")
        
        # Load attention models
        attention_path = os.path.join(load_dir, "attention_model.pt")
        if os.path.exists(attention_path):
            checkpoint = torch.load(attention_path, map_location=self.device)
            self.multi_scale_attention.load_state_dict(checkpoint["multi_scale_attention"])
            self.temporal_attention.load_state_dict(checkpoint["temporal_attention"])
            self.cross_market_attention.load_state_dict(checkpoint["cross_market_attention"])
            self.market_encoder.load_state_dict(checkpoint["market_encoder"])
            self.hierarchical_time_attention.load_state_dict(checkpoint["hierarchical_time_attention"])
            self.financial_transformer.load_state_dict(checkpoint["financial_transformer"])
            self.multi_head_cross_market.load_state_dict(checkpoint["cross_market_multi_head"])
            logger.info(f"Loaded attention models from {attention_path}")
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get a report of recent anomalies."""
        return {
            "total_anomalies": len(self.recent_anomalies),
            "recent_anomalies": self.recent_anomalies,
            "threshold": self.anomaly_threshold
        } 
    
    def optimize_trading_strategy_with_risk_assessment(
        self, 
        state: torch.Tensor,
        risk_averse: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Optimize trading strategy using Distributional SAC with risk assessment.
        
        Args:
            state: Current market state tensor
            risk_averse: Whether to use risk-averse decision making
            
        Returns:
            Dictionary with action and risk metrics
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
        # Prepare for batch dimension if not present
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Get action from distributional SAC
        with torch.no_grad():
            action = self.dist_sac.select_action(state, evaluate=not self.training)
            
            # If risk-averse, compute CVaR and adjust actions
            if risk_averse:
                # Sample multiple actions
                num_samples = 10
                actions = []
                for _ in range(num_samples):
                    actions.append(self.dist_sac.select_action(state, evaluate=False))
                
                actions = torch.stack(actions)
                
                # Evaluate risk for each action
                cvars = []
                for act in actions:
                    cvar1, cvar2 = self.dist_sac.get_cvar(state, act, alpha=self.risk_level)
                    # Use minimum CVaR from both critics for conservative estimate
                    min_cvar = torch.min(cvar1, cvar2)
                    cvars.append(min_cvar)
                
                cvars = torch.stack(cvars)
                
                # Select action with highest CVaR (best worst-case outcome)
                best_idx = torch.argmax(cvars)
                action = actions[best_idx]
                cvar_value = cvars[best_idx]
                
                # Get standard Q-values for comparison
                q1, q2 = self.dist_sac.get_expected_q(state, action)
                expected_value = torch.min(q1, q2)
                
                return {
                    "action": action,
                    "expected_value": expected_value,
                    "cvar": cvar_value,
                    "risk_premium": expected_value - cvar_value
                }
            
            # For standard mode, just return the action
            return {"action": action}
        
    def discrete_strategy_optimization(
        self, 
        state: torch.Tensor,
        evaluate: bool = False
    ) -> torch.Tensor:
        """Optimize discrete trading strategy using Rainbow DQN.
        
        Args:
            state: Current market state tensor
            evaluate: Whether to use deterministic actions
            
        Returns:
            Selected action index
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            
        # Prepare for batch dimension if not present
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Reset noisy layers if exploring
        if not evaluate:
            self.rainbow_dqn.reset_noise()
            
        # Get action from Rainbow DQN
        with torch.no_grad():
            action = self.rainbow_dqn.select_action(state, evaluate=evaluate)
            
        return action
        
    def visualize_value_distribution(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        save_path: Optional[str] = None
    ) -> None:
        """Visualize the value distribution for a state-action pair.
        
        Args:
            state: State tensor
            action: Action tensor
            save_path: Optional path to save the visualization
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
            
        # Prepare for batch dimension if not present
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        # Get value distributions
        with torch.no_grad():
            q_dist1, q_dist2 = self.dist_sac.get_value_distribution(state, action)
            
            # Convert to numpy for plotting
            support = self.dist_sac.support.cpu().numpy()
            q_dist1 = q_dist1[0].cpu().numpy()
            q_dist2 = q_dist2[0].cpu().numpy()
            
            # Plot distributions
            plt.figure(figsize=(10, 6))
            plt.bar(support, q_dist1, alpha=0.5, width=0.3, label="Critic 1")
            plt.bar(support, q_dist2, alpha=0.5, width=0.3, label="Critic 2")
            
            # Add vertical line for CVaR
            cvar1, cvar2 = self.dist_sac.get_cvar(state, action, alpha=self.risk_level)
            plt.axvline(x=cvar1.item(), color='r', linestyle='--', label=f'CVaR {self.risk_level} (Critic 1)')
            plt.axvline(x=cvar2.item(), color='g', linestyle='--', label=f'CVaR {self.risk_level} (Critic 2)')
            
            # Add vertical line for expected value
            q1, q2 = self.dist_sac.get_expected_q(state, action)
            plt.axvline(x=q1.item(), color='r', label='Expected (Critic 1)')
            plt.axvline(x=q2.item(), color='g', label='Expected (Critic 2)')
            
            plt.legend()
            plt.title("Value Distribution for State-Action Pair")
            plt.xlabel("Value")
            plt.ylabel("Probability")
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()

    def transformer_time_series_analysis(
        self,
        time_series: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_predictions: bool = True,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Analyze time series data using the Financial Time Transformer.
        
        Args:
            time_series: Time series data with shape [batch_size, seq_len, features]
            mask: Optional mask for padding, shape [batch_size, seq_len]
            return_predictions: Whether to return predictions
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary with encoded features and optionally predictions and attention weights
        """
        if not isinstance(time_series, torch.Tensor):
            time_series = torch.tensor(time_series, dtype=torch.float32, device=self.device)
            
        # Add batch dimension if needed
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(0)
            
        # Move to device
        time_series = time_series.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
            
        # Process with transformer
        encoded_features, attn_weights, predictions = self.financial_transformer(
            time_series, 
            src_mask=mask,
            return_attention=return_attention,
            return_prediction=return_predictions
        )
        
        result = {
            "encoded_features": encoded_features,
        }
        
        if return_predictions:
            result["predictions"] = predictions
            
        if return_attention:
            result["attention_weights"] = attn_weights
            
            # Parse attention for interpretability
            # Extract the last layer's attention for time point importance
            last_layer_attn = attn_weights[-1]  # [batch_size, num_heads, seq_len, seq_len]
            
            # Average across heads for global importance
            avg_attn = last_layer_attn.mean(dim=1)  # [batch_size, seq_len, seq_len]
            
            # For each time step, get its attention to all other steps
            # Higher attention means that time point is important for predictions
            time_importance = avg_attn.mean(dim=2)  # [batch_size, seq_len]
            
            result["time_point_importance"] = time_importance
            
        return result 