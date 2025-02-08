import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from typing import Dict, List, Tuple, Optional, TypedDict, Union, Any, Mapping
import logging
from torch_geometric.data import Data, Batch
import networkx as nx
import os
import numpy as np
from dataclasses import dataclass

# Define core types
class MarketDataType(TypedDict):
    token_pair: Tuple[str, str]
    price: float
    volume_24h: float
    liquidity: float
    volatility: float
    market_cap: float
    tvl: float
    fees_24h: float
    gas_price: float
    block_time: float
    network_load: float
    pending_tx_count: float
    primary_protocol: Optional[str]
    protocols: List[str]
    tokens: List[Dict[str, Any]]
    pools: List[Dict[str, Any]]
    actual_market: Optional[torch.Tensor]  # For validation
    actual_path: Optional[torch.Tensor]    # For validation
    actual_risk: Optional[torch.Tensor]    # For validation
    actual_execution: Optional[torch.Tensor]  # For validation

@dataclass
class TrainingBatchType:
    """Type definition for training batch data"""
    features: Dict[str, torch.Tensor]
    labels: Dict[str, torch.Tensor]

# Type alias for model outputs
ModelOutputType = Dict[str, torch.Tensor]

logger = logging.getLogger(__name__)

class MarketAnalysisNetwork(nn.Module):
    """Neural network for market analysis"""
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 16)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PathFindingNetwork(nn.Module):
    """Graph neural network for path finding"""
    
    def __init__(self, in_channels: int = 16, out_channels: int = 32):
        super().__init__()
        self.conv = GCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            improved=True,
            cached=True
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        return self.conv(data.x, data.edge_index)

class RiskAssessmentNetwork(nn.Module):
    """Neural network for risk assessment"""
    
    def __init__(self, input_dim: int = 48):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ExecutionStrategyNetwork(nn.Module):
    """Neural network for execution strategy with attention"""
    
    def __init__(self, input_dim: int = 56):
        super().__init__()
        # Multi-head attention for path analysis
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # [execution_timing, gas_price, slippage_tolerance, confidence]
        )
        
        # Protocol-specific heads
        self.protocol_heads = nn.ModuleDict({
            'uniswap': nn.Linear(32, 4),
            'sushiswap': nn.Linear(32, 4),
            'balancer': nn.Linear(32, 4)
        })
        
    def forward(self, x: torch.Tensor, protocol: Optional[str] = None) -> torch.Tensor:
        # Apply attention
        x_reshaped = x.unsqueeze(0)  # Add sequence dimension
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x = attn_out.squeeze(0)  # Remove sequence dimension
        
        # Main network
        features = self.network[:-1](x)  # Get features before last layer
        
        # Protocol-specific output if specified
        if protocol and protocol in self.protocol_heads:
            return self.protocol_heads[protocol](features)
            
        # Default output
        return self.network[-1](features)

class ArbitrageModel(nn.Module):
    """Complete arbitrage model combining all neural networks"""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        
        # Initialize networks with all original capabilities
        self.market_analyzer = MarketAnalysisNetwork().to(device)
        self.path_finder = PathFindingNetwork().to(device)
        self.risk_analyzer = RiskAssessmentNetwork().to(device)
        self.execution_strategist = ExecutionStrategyNetwork().to(device)
        
        # Token economics embeddings
        self.token_embeddings = nn.Embedding(
            num_embeddings=10000,  # Max number of tokens
            embedding_dim=32
        ).to(device)
        
        # Protocol embeddings
        self.protocol_embeddings = nn.Embedding(
            num_embeddings=100,  # Max number of protocols
            embedding_dim=16
        ).to(device)
        
        # Training configuration
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        self.gradient_clip_val = 1.0
        self.early_stopping_patience = 10
        self.scheduler_factor = 0.5
        self.scheduler_patience = 5
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            verbose=True
        )
        
        # Early stopping
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_state = None
        
        # Checkpointing
        self.checkpoint_dir = 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Performance tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def forward(self, market_data: MarketDataType) -> Optional[ModelOutputType]:
        """Forward pass with all original capabilities"""
        try:
            # Extract features
            market_features = self._extract_market_features(market_data)
            
            # Get token embeddings
            token_ids = self._get_token_ids(market_data['token_pair'])
            token_embeds = self.token_embeddings(token_ids)
            
            # Get protocol embeddings
            protocol_ids = self._get_protocol_ids(market_data.get('protocols', []))
            protocol_embeds = self.protocol_embeddings(protocol_ids)
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.gradient_clip_val
            )
            
            # Forward passes through networks
            market_analysis = self.market_analyzer(
                torch.cat([market_features, token_embeds.mean(0)])
            )
            
            graph_data = self._build_graph_data(market_data)
            path_embeddings = self.path_finder(graph_data)
            
            combined_features = torch.cat([
                market_analysis,
                path_embeddings.mean(dim=0),
                market_features,
                protocol_embeds.mean(0)
            ])
            risk_assessment = self.risk_analyzer(combined_features)
            
            strategy_features = torch.cat([
                market_analysis,
                risk_assessment,
                path_embeddings.mean(dim=0),
                protocol_embeds.mean(0)
            ])
            
            # Get protocol-specific execution strategy if available
            protocol = market_data.get('primary_protocol')
            execution_strategy = self.execution_strategist(
                strategy_features,
                protocol=protocol
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                market_analysis,
                risk_assessment,
                execution_strategy
            )
            
            return {
                'market_analysis': market_analysis,
                'path_embeddings': path_embeddings,
                'risk_assessment': risk_assessment,
                'execution_strategy': execution_strategy,
                'confidence': confidence,
                'token_embeddings': token_embeds,
                'protocol_embeddings': protocol_embeds
            }
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            return None
            
    def training_step(
        self,
        batch: TrainingBatchType,
        batch_idx: int
    ) -> Optional[ModelOutputType]:
        """Single training step with advanced features"""
        try:
            # Forward pass
            predictions = self(batch.features)
            if predictions is None:
                return None
                
            # Calculate losses
            market_loss = self._calculate_market_loss(
                predictions['market_analysis'],
                batch.labels['market']
            )
            path_loss = self._calculate_path_loss(
                predictions['path_embeddings'],
                batch.labels['path']
            )
            risk_loss = self._calculate_risk_loss(
                predictions['risk_assessment'],
                batch.labels['risk']
            )
            execution_loss = self._calculate_execution_loss(
                predictions['execution_strategy'],
                batch.labels['execution']
            )
            
            # Weighted total loss
            total_loss = (
                0.3 * market_loss +
                0.3 * path_loss +
                0.2 * risk_loss +
                0.2 * execution_loss
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.gradient_clip_val
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # Track metrics
            self.train_losses.append(float(total_loss))
            self.learning_rates.append(
                self.optimizer.param_groups[0]['lr']
            )
            
            return {
                'loss': total_loss,
                'market_loss': market_loss,
                'path_loss': path_loss,
                'risk_loss': risk_loss,
                'execution_loss': execution_loss
            }
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            return None
            
    def validation_step(
        self,
        batch: TrainingBatchType,
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Single validation step"""
        try:
            with torch.no_grad():
                # Forward pass
                predictions = self(batch.features)
                
                # Calculate losses
                market_loss = self._calculate_market_loss(
                    predictions['market_analysis'],
                    batch.labels['market']
                )
                path_loss = self._calculate_path_loss(
                    predictions['path_embeddings'],
                    batch.labels['path']
                )
                risk_loss = self._calculate_risk_loss(
                    predictions['risk_assessment'],
                    batch.labels['risk']
                )
                execution_loss = self._calculate_execution_loss(
                    predictions['execution_strategy'],
                    batch.labels['execution']
                )
                
                # Total loss
                total_loss = (
                    0.3 * market_loss +
                    0.3 * path_loss +
                    0.2 * risk_loss +
                    0.2 * execution_loss
                )
                
                # Track validation loss
                self.val_losses.append(float(total_loss))
                
                return {
                    'val_loss': total_loss,
                    'val_market_loss': market_loss,
                    'val_path_loss': path_loss,
                    'val_risk_loss': risk_loss,
                    'val_execution_loss': execution_loss
                }
                
        except Exception as e:
            logger.error(f"Error in validation step: {str(e)}")
            return None
            
    def on_validation_epoch_end(self) -> None:
        """Handle end of validation epoch"""
        try:
            # Calculate average validation loss
            avg_val_loss = sum(self.val_losses[-100:]) / len(self.val_losses[-100:])
            
            # Learning rate scheduling
            self.scheduler.step(avg_val_loss)
            
            # Early stopping check
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.patience_counter = 0
                self.best_model_state = self.state_dict()
                self._save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                
            # Check if should stop
            if self.patience_counter >= self.early_stopping_patience:
                logger.info("Early stopping triggered")
                if self.best_model_state is not None:
                    self.load_state_dict(self.best_model_state)
                
        except Exception as e:
            logger.error(f"Error in validation epoch end: {str(e)}")
            
    def _save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint"""
        try:
            checkpoint = {
                'model_state': self.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'best_loss': self.best_loss,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates
            }
            
            path = os.path.join(self.checkpoint_dir, filename)
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint"""
        try:
            path = os.path.join(self.checkpoint_dir, filename)
            if not os.path.exists(path):
                logger.warning(f"No checkpoint found at {path}")
                return
                
            checkpoint = torch.load(path, map_location=self.device)
            
            self.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            
            self.best_loss = checkpoint['best_loss']
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.learning_rates = checkpoint['learning_rates']
            
            logger.info(f"Checkpoint loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            
    def plot_training_progress(self) -> None:
        """Plot training metrics"""
        try:
            import matplotlib.pyplot as plt
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Plot losses
            ax1.plot(self.train_losses, label='Train Loss')
            ax1.plot(self.val_losses, label='Val Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # Plot learning rate
            ax2.plot(self.learning_rates)
            ax2.set_title('Learning Rate')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Learning Rate')
            
            plt.tight_layout()
            plt.savefig('training_progress.png')
            plt.close()
            
            logger.info("Training progress plot saved")
            
        except Exception as e:
            logger.error(f"Error plotting training progress: {str(e)}")
            
    def get_learning_curves(self) -> Dict[str, List[float]]:
        """Get learning curves data"""
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'learning_rate': self.learning_rates
        }

    def _extract_market_features(self, market_data: MarketDataType) -> torch.Tensor:
        """Extract and normalize market features"""
        try:
            features = []
            
            # Price features
            features.extend([
                float(market_data.get('price', 0)),
                float(market_data.get('price_change_24h', 0)),
                float(market_data.get('volume_24h', 0)),
                float(market_data.get('liquidity', 0))
            ])
            
            # Market metrics
            features.extend([
                float(market_data.get('volatility', 0)),
                float(market_data.get('market_cap', 0)),
                float(market_data.get('tvl', 0)),
                float(market_data.get('fees_24h', 0))
            ])
            
            # Network metrics
            features.extend([
                float(market_data.get('gas_price', 0)),
                float(market_data.get('block_time', 0)),
                float(market_data.get('network_load', 0)),
                float(market_data.get('pending_tx_count', 0))
            ])
            
            # Convert to tensor and normalize
            features_tensor = torch.tensor(features, dtype=torch.float32)
            return self._normalize_features(features_tensor)
            
        except Exception as e:
            logger.error(f"Error extracting market features: {str(e)}")
            return torch.zeros(32, dtype=torch.float32)
            
    def _build_graph_data(self, market_data: MarketDataType) -> Data:
        """Build graph data structure for GNN"""
        try:
            G = nx.DiGraph()
            
            # Add nodes (tokens)
            for token in market_data.get('tokens', []):
                G.add_node(token['address'], features=token)
                
            # Add edges (pools)
            for pool in market_data.get('pools', []):
                G.add_edge(
                    pool['token0'],
                    pool['token1'],
                    features=pool
                )
                
            # Convert to PyTorch Geometric Data
            x = []  # Node features
            edge_index = []  # Edge indices
            edge_attr = []  # Edge features
            
            for node in G.nodes():
                node_features = self._extract_token_features(G.nodes[node]['features'])
                x.append(node_features)
                
            for edge in G.edges():
                edge_index.append([
                    list(G.nodes()).index(edge[0]),
                    list(G.nodes()).index(edge[1])
                ])
                edge_features = self._extract_pool_features(G.edges[edge]['features'])
                edge_attr.append(edge_features)
                
            return Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                edge_attr=torch.tensor(edge_attr, dtype=torch.float32)
            )
            
        except Exception as e:
            logger.error(f"Error building graph data: {str(e)}")
            return Data(
                x=torch.zeros((1, 16)),
                edge_index=torch.zeros((2, 1), dtype=torch.long),
                edge_attr=torch.zeros((1, 8))
            )
            
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Normalize features using running statistics"""
        try:
            if not hasattr(self, 'feature_stats'):
                self.feature_stats = {
                    'mean': features.mean(dim=0),
                    'std': features.std(dim=0),
                    'n_samples': 1
                }
            else:
                # Update running statistics
                n = self.feature_stats['n_samples']
                new_mean = self.feature_stats['mean'] * (n/(n+1)) + features.mean(dim=0) * (1/(n+1))
                new_std = torch.sqrt(
                    self.feature_stats['std']**2 * (n/(n+1)) +
                    features.std(dim=0)**2 * (1/(n+1))
                )
                
                self.feature_stats.update({
                    'mean': new_mean,
                    'std': new_std,
                    'n_samples': n + 1
                })
                
            # Normalize features
            normalized = (features - self.feature_stats['mean']) / (self.feature_stats['std'] + 1e-8)
            return torch.clamp(normalized, -5, 5)  # Clip to prevent outliers
            
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return features
            
    def _calculate_confidence(
        self,
        market_analysis: torch.Tensor,
        risk_assessment: torch.Tensor,
        execution_strategy: torch.Tensor
    ) -> torch.Tensor:
        """Calculate overall confidence score"""
        try:
            # Market confidence (30% weight)
            market_conf = torch.sigmoid(market_analysis.mean())
            
            # Risk confidence (40% weight)
            risk_conf = 1 - torch.sigmoid(risk_assessment.mean())
            
            # Execution confidence (30% weight)
            exec_conf = torch.sigmoid(execution_strategy.mean())
            
            # Weighted average
            confidence = (
                0.3 * market_conf +
                0.4 * risk_conf +
                0.3 * exec_conf
            )
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return torch.tensor(0.0)

    def save_model(self, path: str) -> None:
        """Save model state to disk
        
        Args:
            path: Path to save model state
        """
        try:
            state = {
                'market_analyzer': self.market_analyzer.state_dict(),
                'path_finder': self.path_finder.state_dict(),
                'risk_analyzer': self.risk_analyzer.state_dict(),
                'execution_strategist': self.execution_strategist.state_dict(),
                'feature_stats': self.feature_stats if hasattr(self, 'feature_stats') else None
            }
            torch.save(state, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def load_model(self, path: str) -> None:
        """Load model state from disk
        
        Args:
            path: Path to load model state from
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"No model found at {path}")
                return
            
            state = torch.load(path, map_location=self.device)
            
            self.market_analyzer.load_state_dict(state['market_analyzer'])
            self.path_finder.load_state_dict(state['path_finder'])
            self.risk_analyzer.load_state_dict(state['risk_analyzer'])
            self.execution_strategist.load_state_dict(state['execution_strategist'])
            
            if state['feature_stats'] is not None:
                self.feature_stats = state['feature_stats']
            
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def evaluate_model(
        self,
        validation_data: List[MarketDataType]
    ) -> Dict[str, float]:
        """Evaluate model performance on validation data
        
        Args:
            validation_data: List of market data points for validation
            
        Returns:
            Dict containing evaluation metrics:
                - market_error: Market analysis mean squared error
                - path_accuracy: Path finding accuracy
                - risk_auc: Risk assessment AUC score
                - execution_error: Execution strategy mean squared error
                - overall_score: Combined performance score
        """
        try:
            self.eval()  # Set to evaluation mode
            
            market_errors = []
            path_accuracies = []
            risk_scores = []
            execution_errors = []
            
            with torch.no_grad():
                for data in validation_data:
                    # Get predictions
                    predictions = self(data)
                    
                    if predictions is None:
                        continue
                    
                    # Calculate metrics
                    market_error = self._calculate_market_error(
                        predictions['market_analysis'],
                        data['actual_market']
                    )
                    path_accuracy = self._calculate_path_accuracy(
                        predictions['path_embeddings'],
                        data['actual_path']
                    )
                    risk_auc = self._calculate_risk_auc(
                        predictions['risk_assessment'],
                        data['actual_risk']
                    )
                    execution_error = self._calculate_execution_error(
                        predictions['execution_strategy'],
                        data['actual_execution']
                    )
                    
                    market_errors.append(market_error)
                    path_accuracies.append(path_accuracy)
                    risk_scores.append(risk_auc)
                    execution_errors.append(execution_error)
            
            # Calculate average metrics
            metrics = {
                'market_error': float(np.mean(market_errors)),
                'path_accuracy': float(np.mean(path_accuracies)),
                'risk_auc': float(np.mean(risk_scores)),
                'execution_error': float(np.mean(execution_errors))
            }
            
            # Calculate overall score
            metrics['overall_score'] = self._calculate_overall_score(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}
        finally:
            self.train()  # Set back to training mode

    def _calculate_market_error(
        self,
        predictions: torch.Tensor,
        actual: Optional[torch.Tensor]
    ) -> float:
        """Calculate market analysis mean squared error"""
        if actual is None:
            return 0.0
        return float(torch.mean((predictions - actual) ** 2))

    def _calculate_path_accuracy(
        self,
        predictions: torch.Tensor,
        actual: Optional[torch.Tensor]
    ) -> float:
        """Calculate path finding accuracy"""
        if actual is None:
            return 0.0
        pred_paths = torch.argmax(predictions, dim=1)
        actual_paths = torch.argmax(actual, dim=1)
        return float(torch.mean((pred_paths == actual_paths).float()))

    def _calculate_risk_auc(
        self,
        predictions: torch.Tensor,
        actual: Optional[torch.Tensor]
    ) -> float:
        """Calculate risk assessment AUC score"""
        if actual is None:
            return 0.0
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(
                actual.cpu().numpy(),
                predictions.cpu().numpy()
            ))
        except Exception:
            return 0.0

    def _calculate_execution_error(
        self,
        predictions: torch.Tensor,
        actual: Optional[torch.Tensor]
    ) -> float:
        """Calculate execution strategy mean squared error"""
        if actual is None:
            return 0.0
        return float(torch.mean((predictions - actual) ** 2))

    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall model performance score"""
        weights = {
            'market_error': -0.3,  # Negative because lower is better
            'path_accuracy': 0.3,
            'risk_auc': 0.2,
            'execution_error': -0.2  # Negative because lower is better
        }
        
        score = sum(
            metrics[key] * weight
            for key, weight in weights.items()
        )
        
        return float(score)

    def _get_token_ids(self, token_pair: Tuple[str, str]) -> torch.Tensor:
        """Convert token addresses to IDs for embedding"""
        try:
            # Simple hash-based conversion
            ids = [
                int(int(token, 16) % 10000)
                for token in token_pair
            ]
            return torch.tensor(ids, device=self.device)
        except Exception:
            return torch.zeros(2, device=self.device)
            
    def _get_protocol_ids(self, protocols: List[str]) -> torch.Tensor:
        """Convert protocol names to IDs for embedding"""
        try:
            # Simple hash-based conversion
            ids = [
                int(hash(protocol) % 100)
                for protocol in protocols
            ]
            return torch.tensor(ids, device=self.device)
        except Exception:
            return torch.zeros(1, device=self.device) 