"""Variational Autoencoder models for market data representation and anomaly detection"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MarketVAE(nn.Module):
    """Variational Autoencoder for market data representation learning and anomaly detection"""
    
    def __init__(
        self, 
        input_dim: int, 
        latent_dim: int, 
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent distribution parameters
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        
        for dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Tracking running statistics for anomaly detection
        self.register_buffer("reconstruction_error_mean", torch.tensor(0.0))
        self.register_buffer("reconstruction_error_std", torch.tensor(1.0))
        self.register_buffer("num_samples", torch.tensor(0.0))
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input data to latent distribution parameters"""
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstructed input"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and distribution parameters"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var
    
    def loss_function(
        self, 
        x: torch.Tensor, 
        x_reconstructed: torch.Tensor, 
        mu: torch.Tensor, 
        log_var: torch.Tensor,
        kld_weight: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE loss with reconstruction and KL divergence components"""
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_reconstructed, x, reduction='mean')
        
        # KL divergence
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        
        # Total loss
        loss = recon_loss + kld_weight * kld_loss
        
        return {
            'loss': loss,
            'reconstruction_loss': recon_loss,
            'kld_loss': kld_loss
        }
    
    def detect_anomalies(
        self, 
        x: torch.Tensor, 
        threshold_multiplier: float = 3.0,
        update_stats: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect anomalies based on reconstruction error
        
        Args:
            x: Input data tensor
            threshold_multiplier: Number of standard deviations for anomaly threshold
            update_stats: Whether to update running statistics
            
        Returns:
            is_anomaly: Boolean tensor indicating anomalies
            anomaly_scores: Anomaly scores (z-scores of reconstruction errors)
        """
        with torch.no_grad():
            # Get reconstruction
            x_reconstructed, _, _ = self.forward(x)
            
            # Calculate reconstruction error (MSE per sample)
            recon_error = F.mse_loss(x_reconstructed, x, reduction='none').mean(dim=1)
            
            # Update statistics if in training mode
            if update_stats and self.training:
                # Update running stats using Welford's algorithm
                n = self.num_samples
                new_n = n + x.size(0)
                
                batch_mean = recon_error.mean()
                delta = batch_mean - self.reconstruction_error_mean
                new_mean = self.reconstruction_error_mean + delta * x.size(0) / new_n
                
                self.reconstruction_error_mean = new_mean
                
                # Update variance
                batch_var = recon_error.var()
                m_a = self.reconstruction_error_std ** 2 * n
                m_b = batch_var * x.size(0)
                M2 = m_a + m_b + delta ** 2 * n * x.size(0) / new_n
                self.reconstruction_error_std = torch.sqrt(M2 / new_n)
                
                # Update count
                self.num_samples = new_n
            
            # Calculate z-scores for anomaly detection
            if self.reconstruction_error_std > 0:
                z_scores = (recon_error - self.reconstruction_error_mean) / self.reconstruction_error_std
            else:
                z_scores = torch.zeros_like(recon_error)
                
            # Detect anomalies based on threshold
            is_anomaly = z_scores > threshold_multiplier
            
            return is_anomaly, z_scores
            
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation (embedding) for input data"""
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu


class ConditionalMarketVAE(nn.Module):
    """Conditional VAE for market data that includes market conditions"""
    
    def __init__(
        self, 
        input_dim: int,
        condition_dim: int, 
        latent_dim: int, 
        hidden_dims: List[int] = [256, 128],
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Combine input and condition dimensions for encoder
        combined_input_dim = input_dim + condition_dim
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = combined_input_dim
        
        for dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent distribution parameters
        self.mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Build decoder layers (combines latent space with condition)
        decoder_layers = []
        prev_dim = latent_dim + condition_dim
        
        for dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store dimensions
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim
        
    def encode(self, x: torch.Tensor, condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input data with condition to latent distribution parameters"""
        # Concatenate input and condition
        h = torch.cat([x, condition], dim=1)
        h = self.encoder(h)
        return self.mu(h), self.log_var(h)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Decode latent vector with condition to reconstructed input"""
        # Concatenate latent vector and condition
        z_cond = torch.cat([z, condition], dim=1)
        return self.decoder(z_cond)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with condition"""
        mu, log_var = self.encode(x, condition)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z, condition)
        return x_reconstructed, mu, log_var
    
    def sample(
        self, 
        num_samples: int, 
        condition: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Generate samples conditioned on market state"""
        if device is None:
            device = next(self.parameters()).device
            
        # Repeat condition for each sample if needed
        if condition.size(0) == 1:
            condition = condition.repeat(num_samples, 1)
        elif condition.size(0) != num_samples:
            raise ValueError(f"Condition batch size ({condition.size(0)}) must be 1 or {num_samples}")
            
        # Sample from latent space
        z = torch.randn(num_samples, self.latent_dim, device=device)
        
        # Decode with condition
        return self.decode(z, condition)


class HierarchicalMarketVAE(nn.Module):
    """Hierarchical VAE for modeling market data at multiple time scales"""
    
    def __init__(
        self,
        input_dim: int,
        local_latent_dim: int = 8,
        global_latent_dim: int = 16,
        sequence_length: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.local_latent_dim = local_latent_dim
        self.global_latent_dim = global_latent_dim
        
        # Local encoder (processes each timestep individually)
        self.local_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.local_mu = nn.Linear(hidden_dim, local_latent_dim)
        self.local_log_var = nn.Linear(hidden_dim, local_latent_dim)
        
        # Global encoder (processes sequence of local latents)
        self.global_encoder = nn.Sequential(
            nn.Linear(local_latent_dim * sequence_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.global_mu = nn.Linear(hidden_dim, global_latent_dim)
        self.global_log_var = nn.Linear(hidden_dim, global_latent_dim)
        
        # Decoder: first generate local latents, then reconstruct timesteps
        self.global_to_local = nn.Sequential(
            nn.Linear(global_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, local_latent_dim * sequence_length)
        )
        
        self.local_decoder = nn.Sequential(
            nn.Linear(local_latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def encode_local(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode each timestep to local latent variables
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            
        Returns:
            local_mu: Local latent means [batch_size, sequence_length, local_latent_dim]
            local_log_var: Local latent log variances [batch_size, sequence_length, local_latent_dim]
            local_z: Sampled local latents [batch_size, sequence_length, local_latent_dim]
        """
        batch_size, seq_len, _ = x.size()
        
        # Reshape to process each timestep
        x_flat = x.reshape(-1, self.input_dim)
        h = self.local_encoder(x_flat)
        
        # Get distribution parameters
        local_mu = self.local_mu(h)
        local_log_var = self.local_log_var(h)
        
        # Sample local latents
        std = torch.exp(0.5 * local_log_var)
        eps = torch.randn_like(std)
        local_z = local_mu + eps * std
        
        # Reshape back to sequence form
        local_mu = local_mu.reshape(batch_size, seq_len, self.local_latent_dim)
        local_log_var = local_log_var.reshape(batch_size, seq_len, self.local_latent_dim)
        local_z = local_z.reshape(batch_size, seq_len, self.local_latent_dim)
        
        return local_mu, local_log_var, local_z
    
    def encode_global(self, local_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode sequence of local latents to global latent
        
        Args:
            local_z: Local latents [batch_size, sequence_length, local_latent_dim]
            
        Returns:
            global_mu: Global latent mean [batch_size, global_latent_dim]
            global_log_var: Global latent log variance [batch_size, global_latent_dim]
            global_z: Sampled global latent [batch_size, global_latent_dim]
        """
        batch_size = local_z.size(0)
        
        # Flatten sequence
        local_z_flat = local_z.reshape(batch_size, -1)
        h = self.global_encoder(local_z_flat)
        
        # Get distribution parameters
        global_mu = self.global_mu(h)
        global_log_var = self.global_log_var(h)
        
        # Sample global latent
        std = torch.exp(0.5 * global_log_var)
        eps = torch.randn_like(std)
        global_z = global_mu + eps * std
        
        return global_mu, global_log_var, global_z
    
    def decode(self, global_z: torch.Tensor) -> torch.Tensor:
        """Decode global latent to full sequence
        
        Args:
            global_z: Global latent [batch_size, global_latent_dim]
            
        Returns:
            x_reconstructed: Reconstructed sequence [batch_size, sequence_length, input_dim]
        """
        batch_size = global_z.size(0)
        
        # Generate local latents from global
        local_latents_flat = self.global_to_local(global_z)
        local_latents = local_latents_flat.reshape(batch_size, self.sequence_length, self.local_latent_dim)
        
        # Decode each timestep
        local_latents_flat = local_latents.reshape(-1, self.local_latent_dim)
        x_reconstructed_flat = self.local_decoder(local_latents_flat)
        
        # Reshape back to sequence
        x_reconstructed = x_reconstructed_flat.reshape(batch_size, self.sequence_length, self.input_dim)
        
        return x_reconstructed
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through hierarchical VAE
        
        Args:
            x: Input sequence [batch_size, sequence_length, input_dim]
            
        Returns:
            Dictionary with reconstructed data and latent parameters
        """
        # Encode to local latents
        local_mu, local_log_var, local_z = self.encode_local(x)
        
        # Encode to global latent
        global_mu, global_log_var, global_z = self.encode_global(local_z)
        
        # Decode back to original space
        x_reconstructed = self.decode(global_z)
        
        return {
            'reconstruction': x_reconstructed,
            'local_mu': local_mu,
            'local_log_var': local_log_var,
            'local_z': local_z,
            'global_mu': global_mu,
            'global_log_var': global_log_var,
            'global_z': global_z
        }
    
    def loss_function(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute hierarchical VAE loss"""
        # Unpack outputs
        x_reconstructed = outputs['reconstruction']
        local_mu = outputs['local_mu']
        local_log_var = outputs['local_log_var']
        global_mu = outputs['global_mu']
        global_log_var = outputs['global_log_var']
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_reconstructed, x, reduction='mean')
        
        # KL divergence for local latents
        local_kld = -0.5 * torch.mean(1 + local_log_var - local_mu.pow(2) - local_log_var.exp())
        
        # KL divergence for global latent
        global_kld = -0.5 * torch.mean(1 + global_log_var - global_mu.pow(2) - global_log_var.exp())
        
        # Total loss (with weighting)
        loss = recon_loss + 0.1 * local_kld + 0.01 * global_kld
        
        return {
            'loss': loss,
            'reconstruction_loss': recon_loss,
            'local_kld': local_kld,
            'global_kld': global_kld
        } 