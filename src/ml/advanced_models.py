"""Advanced neural network architectures"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, GINConv
from typing import List, Tuple, Optional
import math

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_linear(out)

class TransformerBlock(nn.Module):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Multi-head attention
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed-forward network
        fed_forward = self.ff(x)
        x = self.norm2(x + self.dropout(fed_forward))
        
        return x

class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for sequence modeling"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()
        
        layers = []
        num_levels = len(hidden_dims)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else hidden_dims[i-1]
            out_channels = hidden_dims[i]
            
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels,
                        kernel_size,
                        dilation=dilation,
                        padding=(kernel_size-1) * dilation
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
        Returns:
            Output tensor of shape [batch_size, sequence_length, hidden_dims[-1]]
        """
        # Convert to channel-first format
        x = x.transpose(1, 2)
        x = self.network(x)
        # Convert back to sequence-first format
        return x.transpose(1, 2)

class HybridGNNTransformer(nn.Module):
    """Hybrid model combining GNN and Transformer for market graph analysis"""
    
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GATConv(
                node_dim if i == 0 else hidden_dim,
                hidden_dim,
                heads=num_heads,
                dropout=dropout,
                edge_dim=edge_dim
            )
            for i in range(num_layers)
        ])
        
        # Transformer layers for temporal processing
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim,
                num_heads,
                hidden_dim * 4,
                dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Graph attention processing
        for gat in self.gat_layers:
            x = gat(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
        
        # Reshape for transformer
        if batch is not None:
            x = torch.stack([
                x[batch == i]
                for i in range(batch.max() + 1)
            ])
        
        # Transformer processing
        for transformer in self.transformer_layers:
            x = transformer(x)
        
        # Global pooling and prediction
        x = x.mean(dim=1)  # Global average pooling
        return self.output_net(x)

class AdaptiveEnsemble(nn.Module):
    """Adaptive ensemble with learnable weights"""
    
    def __init__(
        self,
        models: List[nn.Module],
        input_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.models = nn.ModuleList(models)
        
        # Weight network
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(models)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get adaptive weights
        weights = self.weight_net(x)
        
        # Get predictions from each model
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred.unsqueeze(-1))
        
        # Stack predictions
        predictions = torch.cat(predictions, dim=-1)
        
        # Weighted sum
        return torch.sum(predictions * weights.unsqueeze(1), dim=-1)

class MetaLearningModel(nn.Module):
    """Meta-learning model for quick adaptation to market changes"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_tasks: int,
        adaptation_lr: float = 0.01
    ):
        super().__init__()
        
        self.adaptation_lr = adaptation_lr
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Task-specific layers
        self.task_nets = nn.ModuleList([
            nn.Linear(hidden_dim, 1)
            for _ in range(num_tasks)
        ])
        
    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: int = 5
    ):
        """Quick adaptation to new task"""
        # Create a copy of task-specific parameters
        adapted_params = [
            net.weight.clone().requires_grad_(),
            net.bias.clone().requires_grad_()
            for net in self.task_nets
        ]
        
        optimizer = torch.optim.SGD(adapted_params, lr=self.adaptation_lr)
        
        for _ in range(num_steps):
            # Forward pass
            features = self.feature_net(support_x)
            predictions = []
            for w, b in zip(adapted_params[::2], adapted_params[1::2]):
                pred = F.linear(features, w, b)
                predictions.append(pred)
            predictions = torch.cat(predictions, dim=1)
            
            # Compute loss and adapt
            loss = F.mse_loss(predictions, support_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_params
    
    def forward(
        self,
        x: torch.Tensor,
        adapted_params: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        features = self.feature_net(x)
        
        if adapted_params is None:
            # Normal forward pass
            predictions = []
            for net in self.task_nets:
                pred = net(features)
                predictions.append(pred)
            return torch.cat(predictions, dim=1)
        else:
            # Forward pass with adapted parameters
            predictions = []
            for w, b in zip(adapted_params[::2], adapted_params[1::2]):
                pred = F.linear(features, w, b)
                predictions.append(pred)
            return torch.cat(predictions, dim=1) 