"""
Advanced attention mechanisms for market data processing.
This module provides attention-based models for:
1. Processing time series at multiple temporal scales
2. Analyzing relationships between different markets
3. Handling hierarchical temporal structures
4. Transformer-based sequence modeling for financial data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Tuple, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models to provide
    sequence order information
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and transpose
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as buffer (persistent state)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, embedding_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism allows the model to jointly attend to 
    information from different representation subspaces
    """
    
    def __init__(
        self,
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize multi-head attention
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to include bias in projections
        """
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for queries, keys, and values
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute multi-head attention
        
        Args:
            query: Query tensor [batch_size, query_len, embed_dim]
            key: Key tensor [batch_size, key_len, embed_dim]
            value: Value tensor [batch_size, value_len, embed_dim]
            mask: Optional mask tensor [batch_size, query_len, key_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.embed_dim
        )
        
        # Final projection
        output = self.out_proj(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output

class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with multi-head self-attention and
    position-wise feed-forward network
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int, 
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        """
        Initialize transformer encoder layer
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Position-wise feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
    def forward(
        self,
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            src_mask: Optional mask for self-attention
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor and optionally attention weights
        """
        # Self-attention
        if return_attention:
            attn_output, attn_weights = self.self_attn(
                src, src, src, mask=src_mask, return_attention=True
            )
        else:
            attn_output = self.self_attn(src, src, src, mask=src_mask)
            
        # Residual connection and layer normalization
        src = src + self.dropout1(attn_output)
        src = self.norm1(src)
        
        # Feedforward network
        ff_output = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        
        # Residual connection and layer normalization
        src = src + self.dropout2(ff_output)
        src = self.norm2(src)
        
        if return_attention:
            return src, attn_weights
        return src

class FinancialTimeTransformer(nn.Module):
    """
    Transformer model for financial time series processing with
    specialized features for market data
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        max_seq_length: int = 1000,
        learned_pos_encoding: bool = True,
        forecast_horizon: Optional[int] = None
    ):
        """
        Initialize time transformer
        
        Args:
            input_dim: Dimensionality of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function
            max_seq_length: Maximum sequence length
            learned_pos_encoding: Whether to use learned positional encoding
            forecast_horizon: Number of future time steps to predict (None = same as input)
        """
        super(FinancialTimeTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        if learned_pos_encoding:
            self.pos_encoding = nn.Parameter(
                torch.zeros(max_seq_length, 1, d_model)
            )
            nn.init.normal_(self.pos_encoding, mean=0, std=0.02)
        else:
            self.pos_encoding = PositionalEncoding(
                d_model, max_seq_length, dropout
            )
        
        self.learned_pos_encoding = learned_pos_encoding
        
        # Build encoder layers
        encoder_layers = []
        for _ in range(num_encoder_layers):
            encoder_layers.append(
                TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout, activation
                )
            )
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # Output projection
        if forecast_horizon is not None:
            self.output_projection = nn.Linear(d_model, input_dim * forecast_horizon)
        else:
            self.output_projection = nn.Linear(d_model, input_dim)
        
        # Layer normalization for final output
        self.norm = nn.LayerNorm(d_model)
        
        # Financial-specific features
        self.volatility_attention = nn.Parameter(torch.ones(1, 1, d_model))
        self.temporal_decay = nn.Parameter(torch.ones(1, 1, d_model) * 0.99)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional mask tensor
            timestamps: Optional timestamp information for temporal features
            
        Returns:
            Dictionary containing output tensor and attention maps
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        if self.learned_pos_encoding:
            x = x + self.pos_encoding[:seq_len]
        else:
            # Transpose for positional encoding then transpose back
            x = x.transpose(0, 1)
            x = self.pos_encoding(x)
            x = x.transpose(0, 1)
        
        # Process temporal features if provided
        if timestamps is not None:
            # Calculate time differences (normalized)
            time_diffs = timestamps[:, 1:] - timestamps[:, :-1]
            time_diffs = F.pad(time_diffs, (1, 0), value=0)
            time_diffs = time_diffs / (time_diffs.mean() + 1e-8)
        
            # Create time embeddings
            time_embedding = torch.exp(-time_diffs.unsqueeze(-1) * self.temporal_decay)
            x = x * time_embedding
        
        # Store attention maps for visualization
        attention_maps = []
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x, attn = layer(x, src_mask=mask, return_attention=True)
            attention_maps.append(attn.detach())
            
        # Apply final normalization
        x = self.norm(x)
        
        # Financial-specific feature: apply volatility-based attention
        # Compute simple proxy for volatility
        if x.size(1) > 1:
            volatility_proxy = torch.std(x, dim=1, keepdim=True)
            volatility_attn = torch.sigmoid(volatility_proxy @ self.volatility_attention)
            x = x * volatility_attn
        
        # Apply output projection
        outputs = self.output_projection(x)
        
        # Reshape output if forecasting multiple steps
        if self.forecast_horizon is not None:
            outputs = outputs.view(batch_size, seq_len, self.forecast_horizon, self.input_dim)
        
        return {
            "output": outputs,
            "attention_maps": attention_maps,
            "hidden_states": x
        }

class MultiHeadCrossMarketAttention(nn.Module):
    """
    Multi-head attention for analyzing relationships between multiple markets
    """
    
    def __init__(
        self,
        market_dim: int,
        num_markets: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head cross-market attention
        
        Args:
            market_dim: Dimensionality of market features
            num_markets: Number of markets to analyze
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadCrossMarketAttention, self).__init__()
        
        self.market_dim = market_dim
        self.num_markets = num_markets
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Projections for each market
        self.market_projections = nn.ModuleList([
            nn.Linear(market_dim, embed_dim)
            for _ in range(num_markets)
        ])
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim, num_heads, dropout
        )
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, market_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(market_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
    def forward(
        self,
        markets: List[torch.Tensor],
        focus_market_idx: int = 0,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process multiple markets with cross-market attention
        
        Args:
            markets: List of market tensors [batch_size, seq_len, market_dim]
            focus_market_idx: Index of the market to focus on
            mask: Optional attention mask
               
        Returns:
            Dictionary with enhanced features and attention maps
        """
        assert len(markets) == self.num_markets, f"Expected {self.num_markets} markets, got {len(markets)}"
        
        batch_size, seq_len, _ = markets[0].shape
        
        # Project each market to embedding space
        market_embeddings = [
            proj(market) for proj, market in zip(self.market_projections, markets)
        ]
        
        # Get the focus market
        focus_market = market_embeddings[focus_market_idx]
        
        # Combine all other markets for keys and values
        all_markets = torch.cat([
            emb.unsqueeze(1) for emb in market_embeddings
        ], dim=1)  # [batch_size, num_markets, seq_len, embed_dim]
        
        # Reshape for attention
        all_markets = all_markets.view(batch_size * self.num_markets, seq_len, self.embed_dim)
        focus_market_expanded = focus_market.unsqueeze(1).repeat(1, self.num_markets, 1, 1)
        focus_market_expanded = focus_market_expanded.view(batch_size * self.num_markets, seq_len, self.embed_dim)
            
        # Apply attention
        attn_output, attn_weights = self.attention(
            focus_market_expanded, all_markets, all_markets,
            mask=mask, return_attention=True
        )
        
        # Reshape attention output
        attn_output = attn_output.view(batch_size, self.num_markets, seq_len, self.embed_dim)
        attn_weights = attn_weights.view(batch_size, self.num_markets, self.num_heads, seq_len, seq_len)
        
        # Average attention across heads for visualization
        attn_weights_avg = attn_weights.mean(dim=2)
                
        # Extract enhanced features for focus market
        enhanced_focus = attn_output[:, focus_market_idx]
        
        # Add residual connection and normalization
        enhanced_focus = focus_market + enhanced_focus
        enhanced_focus = self.norm1(enhanced_focus)
        
        # Apply feed-forward network
        ffn_output = self.ffn(enhanced_focus)
        enhanced_focus = enhanced_focus + ffn_output
        enhanced_focus = self.norm1(enhanced_focus)
        
        # Project back to original market dimension
        output = self.output_projection(enhanced_focus)
            
        # Final residual connection
        output = markets[focus_market_idx] + output
        output = self.norm2(output)
        
        return {
            "enhanced_features": output,
            "attention_weights": attn_weights_avg,
            "market_influence": attn_weights_avg.mean(dim=-1),  # Average influence per market
            "all_market_enhanced": attn_output.view(batch_size, self.num_markets, seq_len, self.embed_dim)
        } 