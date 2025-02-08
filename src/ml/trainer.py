import torch
import logging
from typing import Dict, List, Optional, Any, TypedDict, Union, cast
from dataclasses import dataclass, field
import time
import statistics
import os
from torch.optim import Adam
import torch.nn as nn
from torch.nn.parameter import Parameter

from ..core.types import MarketDataType, OpportunityType

logger = logging.getLogger(__name__)

class PredictionOutput(TypedDict):
    market_analysis: torch.Tensor
    path_embeddings: torch.Tensor
    risk_assessment: torch.Tensor
    execution_strategy: torch.Tensor

class LabelData(TypedDict):
    market_targets: torch.Tensor
    path_targets: torch.Tensor
    risk_targets: torch.Tensor
    execution_targets: torch.Tensor
    market: torch.Tensor
    path: torch.Tensor
    risk: torch.Tensor
    execution: torch.Tensor

@dataclass
class TrainingBatch:
    """Training batch data structure"""
    features: torch.Tensor
    labels: Dict[str, torch.Tensor]
    market_conditions: List[MarketDataType]
    timestamps: List[float]
    execution_results: List[Dict[str, Any]] = field(default_factory=list)
    protocol_data: List[str] = field(default_factory=list)
    indices: List[int] = field(default_factory=list)

class TrainingManager:
    """Manages model training and data collection"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        buffer_size: int = 10000,
        batch_size: int = 32
    ):
        self.model = model
        self.device = device
        self.max_buffer_size = buffer_size
        self.batch_size = batch_size
        
        # Unified buffer for all training data
        self.replay_buffer = {
            'features': [],
            'labels': [],
            'market_conditions': [],
            'timestamps': [],
            'priorities': [],  # For prioritized experience replay
            'protocol_data': [],  # Protocol-specific data
            'token_data': [],  # Token-specific data
            'path_data': [],  # Path finding data
            'execution_results': []  # Execution results
        }
        
        # Unified metrics tracking
        self.metrics = {
            # Task-specific metrics
            'market_metrics': {
                'loss': [],
                'accuracy': [],
                'validation': []
            },
            'path_metrics': {
                'loss': [],
                'accuracy': [],
                'validation': []
            },
            'risk_metrics': {
                'loss': [],
                'accuracy': [],
                'validation': []
            },
            'execution_metrics': {
                'loss': [],
                'accuracy': [],
                'validation': []
            },
            # Protocol-specific metrics
            'protocol_metrics': {},
            # Overall metrics
            'success_rate': [],
            'profit_history': [],
            'learning_rate': [],
            'validation_loss': []
        }
        
        # Task weights for multi-task learning
        self.task_weights = {
            'market': 0.3,
            'path': 0.3,
            'risk': 0.2,
            'execution': 0.2
        }
        
        # Protocol-specific loss functions
        self.protocol_loss_fns = {
            'uniswap': self._uniswap_loss,
            'sushiswap': self._sushiswap_loss,
            'balancer': self._balancer_loss
        }
        
        # Curriculum learning configuration
        self.curriculum_stages = [
            {
                'name': 'basic',
                'difficulty': 0.5,
                'min_samples': 1000
            },
            {
                'name': 'intermediate',
                'difficulty': 0.75,
                'min_samples': 5000
            },
            {
                'name': 'advanced',
                'difficulty': 1.0,
                'min_samples': 10000
            }
        ]
        self.current_stage = 0
        
        # Online learning configuration
        self.online_update_frequency = 100
        self.online_batch_size = 16
        self.samples_since_update = 0
        
        # Cross-validation configuration
        self.n_folds = 5
        self.current_fold = 0
        self.cross_val_results = []
        
        # Hyperparameter optimization configuration
        self.hp_search_space = {
            'learning_rate': [1e-4, 3e-4, 1e-3],
            'batch_size': [16, 32, 64],
            'weight_decay': [1e-5, 1e-4, 1e-3]
        }
        self.best_hp_config = None
        self.best_hp_score = float('inf')
        
        # Initialize optimizer as a regular attribute, not through __setattr__
        self._optimizer: Optional[Adam] = None
        
    @property
    def optimizer(self) -> Optional[Adam]:
        return self._optimizer
        
    @optimizer.setter
    def optimizer(self, value: Adam) -> None:
        self._optimizer = value
        
    async def train_step(self, batch: TrainingBatch) -> Dict[str, float]:
        """Enhanced training step with all original capabilities"""
        try:
            # Apply curriculum difficulty
            batch = self._apply_curriculum(batch)
            
            # Update priorities based on loss
            losses = await self._calculate_sample_losses(batch)
            self._update_priorities(batch, losses)
            
            # Multi-task learning
            metrics = {}
            for task, weight in self.task_weights.items():
                task_loss = await self._calculate_task_loss(batch, task)
                metrics[f"{task}_loss"] = float(task_loss)
                metrics[f"{task}_weight"] = weight
                
            # Protocol-specific losses
            for protocol in batch.protocol_data:
                if protocol in self.protocol_loss_fns:
                    protocol_loss = self.protocol_loss_fns[protocol](batch)
                    metrics[f"{protocol}_loss"] = float(protocol_loss)
                    
            # Update historical performance
            self._update_performance_history(metrics)
            
            # Online learning check
            self.samples_since_update += len(batch.features)
            if self.samples_since_update >= self.online_update_frequency:
                await self._perform_online_update()
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            return {}
            
    def get_training_batch(self) -> Optional[TrainingBatch]:
        """Get batch with prioritized sampling"""
        try:
            if len(self.replay_buffer['features']) < self.batch_size:
                return None
                
            # Sample based on priorities
            priorities = torch.tensor(self.replay_buffer['priorities'])
            probs = priorities / priorities.sum()
            indices = torch.multinomial(probs, self.batch_size)
            
            return TrainingBatch(
                features=torch.stack([
                    self.replay_buffer['features'][i] for i in indices
                ]),
                labels={
                    key: torch.stack([
                        self.replay_buffer['labels'][i][key] 
                        for i in indices
                    ]) for key in self.replay_buffer['labels'][0].keys()
                },
                market_conditions=[
                    self.replay_buffer['market_conditions'][i] for i in indices
                ],
                timestamps=[
                    self.replay_buffer['timestamps'][i] for i in indices
                ],
                execution_results=[
                    self.replay_buffer['execution_results'][i] for i in indices
                ],
                indices=indices.tolist()
            )
            
        except Exception as e:
            logger.error(f"Error getting training batch: {str(e)}")
            return None
            
    def store_experience(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        market_conditions: MarketDataType,
        priority: float = 1.0
    ) -> None:
        """Store experience with priority"""
        try:
            # Add to replay buffer
            self.replay_buffer['features'].append(features)
            self.replay_buffer['labels'].append(labels)
            self.replay_buffer['market_conditions'].append(market_conditions)
            self.replay_buffer['timestamps'].append(time.time())
            self.replay_buffer['priorities'].append(priority)
            
            # Cleanup if buffer is full
            if len(self.replay_buffer['features']) > self.max_buffer_size:
                # Remove samples with lowest priority
                priorities = torch.tensor(self.replay_buffer['priorities'])
                _, indices = torch.sort(priorities)
                keep_indices = indices[-self.max_buffer_size:]
                
                for key in self.replay_buffer:
                    self.replay_buffer[key] = [
                        self.replay_buffer[key][i] for i in keep_indices
                    ]
                    
        except Exception as e:
            logger.error(f"Error storing experience: {str(e)}")
            
    def _verify_batch_quality(self, batch: TrainingBatch) -> bool:
        """Verify that the training batch contains meaningful data"""
        try:
            # Check for empty or None values
            if not batch or not all([
                len(batch.features) > 0,
                len(batch.labels) > 0,
                len(batch.market_conditions) > 0
            ]):
                return False
                
            # Verify feature statistics
            if torch.isnan(batch.features).any():
                logger.warning("Found NaN values in features")
                return False
                
            # Verify labels are reasonable
            if not (0 <= torch.mean(batch.labels) <= 1):
                logger.warning("Labels outside expected range")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error in batch verification: {str(e)}")
            return False
            
    def _update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update tracking metrics for learning progress"""
        try:
            # Update loss history
            for key in ['market_loss', 'path_loss', 'risk_loss', 'execution_loss']:
                if key in metrics:
                    self.metrics[key].append(metrics[key])
                    
            # Cleanup old metrics
            max_history = 1000
            for key in self.metrics:
                if len(self.metrics[key]) > max_history:
                    self.metrics[key] = self.metrics[key][-max_history:]
                    
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            
    def _calculate_market_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate market analysis loss"""
        return torch.nn.functional.mse_loss(
            predictions['market_analysis'],
            labels['market_targets']
        )
        
    def _calculate_path_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate path finding loss"""
        return torch.nn.functional.cross_entropy(
            predictions['path_embeddings'],
            labels['path_targets']
        )
        
    def _calculate_risk_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate risk assessment loss"""
        return torch.nn.functional.binary_cross_entropy(
            predictions['risk_assessment'],
            labels['risk_targets']
        )
        
    def _calculate_execution_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate execution strategy loss"""
        return torch.nn.functional.mse_loss(
            predictions['execution_strategy'],
            labels['execution_targets']
        )

    def save_training_state(self, path: str) -> None:
        """Save training state to disk
        
        Args:
            path: Path to save training state
        """
        try:
            state = {
                'replay_buffer': self.replay_buffer,
                'metrics': self.metrics,
                'batch_size': self.batch_size,
                'max_buffer_size': self.max_buffer_size
            }
            torch.save(state, path)
            logger.info(f"Training state saved to {path}")
        except Exception as e:
            logger.error(f"Error saving training state: {str(e)}")

    def load_training_state(self, path: str) -> None:
        """Load training state from disk
        
        Args:
            path: Path to load training state from
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"No training state found at {path}")
                return
            
            state = torch.load(path)
            
            self.replay_buffer = state['replay_buffer']
            self.metrics = state['metrics']
            self.batch_size = state['batch_size']
            self.max_buffer_size = state['max_buffer_size']
            
            logger.info(f"Training state loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading training state: {str(e)}")

    async def validation_step(
        self,
        validation_data: List[MarketDataType]
    ) -> Dict[str, float]:
        """Perform validation step"""
        try:
            # Convert validation data to appropriate format
            batch = self._prepare_validation_batch(validation_data)
            if batch is None:
                return {}
                
            # Get model evaluation metrics
            metrics = self.model.evaluate_model(validation_data)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in validation step: {str(e)}")
            return {}

    def _apply_curriculum(self, batch: TrainingBatch) -> TrainingBatch:
        """Apply curriculum learning difficulty"""
        try:
            # Check if should advance curriculum
            if (len(self.replay_buffer['features']) >=
                self.curriculum_stages[self.current_stage]['min_samples']):
                self.current_stage = min(
                    self.current_stage + 1,
                    len(self.curriculum_stages) - 1
                )
                
            # Apply difficulty
            difficulty = self.curriculum_stages[self.current_stage]['difficulty']
            
            # Adjust batch based on difficulty
            batch.features = batch.features * difficulty
            
            return batch
            
        except Exception as e:
            logger.error(f"Error applying curriculum: {str(e)}")
            return batch
            
    async def _calculate_sample_losses(
        self,
        batch: TrainingBatch
    ) -> torch.Tensor:
        """Calculate loss for each sample"""
        try:
            with torch.no_grad():
                predictions = self.model(batch.features)
                
                sample_losses = []
                for i in range(len(batch.features)):
                    loss = self.model._calculate_market_loss(
                        predictions['market_analysis'][i:i+1],
                        batch.labels['market'][i:i+1]
                    )
                    sample_losses.append(loss)
                    
                return torch.stack(sample_losses)
                
        except Exception as e:
            logger.error(f"Error calculating sample losses: {str(e)}")
            return torch.ones(len(batch.features))
            
    def _update_priorities(
        self,
        batch: TrainingBatch,
        losses: torch.Tensor
    ) -> None:
        """Update replay buffer priorities"""
        try:
            # Convert losses to priorities
            priorities = losses.abs().cpu().numpy()
            
            # Update priorities in buffer
            for i, idx in enumerate(batch.indices):
                self.replay_buffer['priorities'][idx] = float(priorities[i])
                
        except Exception as e:
            logger.error(f"Error updating priorities: {str(e)}")
            
    async def _perform_online_update(self) -> None:
        """Perform online learning update"""
        try:
            # Get small batch of recent samples
            recent_indices = range(
                max(0, len(self.replay_buffer['features']) - self.online_batch_size),
                len(self.replay_buffer['features'])
            )
            
            batch = TrainingBatch(
                features=torch.stack([
                    self.replay_buffer['features'][i] for i in recent_indices
                ]),
                labels=torch.stack([
                    self.replay_buffer['labels'][i] for i in recent_indices
                ]),
                market_conditions=[
                    self.replay_buffer['market_conditions'][i]
                    for i in recent_indices
                ],
                timestamps=[
                    self.replay_buffer['timestamps'][i] for i in recent_indices
                ]
            )
            
            # Perform update
            self.model.training_step(batch, 0)
            
            self.samples_since_update = 0
            
        except Exception as e:
            logger.error(f"Error in online update: {str(e)}")
            
    async def perform_cross_validation(self) -> Dict[str, float]:
        """Perform k-fold cross validation"""
        try:
            # Split data into folds
            fold_size = len(self.replay_buffer['features']) // self.n_folds
            
            metrics = []
            for fold in range(self.n_folds):
                # Get validation indices
                val_start = fold * fold_size
                val_end = (fold + 1) * fold_size
                
                # Split data
                train_data = {
                    'features': self.replay_buffer['features'][:val_start] +
                               self.replay_buffer['features'][val_end:],
                    'labels': self.replay_buffer['labels'][:val_start] +
                             self.replay_buffer['labels'][val_end:]
                }
                
                val_data = {
                    'features': self.replay_buffer['features'][val_start:val_end],
                    'labels': self.replay_buffer['labels'][val_start:val_end]
                }
                
                # Train on fold
                fold_metrics = await self._train_fold(train_data, val_data)
                metrics.append(fold_metrics)
                
            # Calculate average metrics
            avg_metrics = {}
            for key in metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in metrics) / len(metrics)
                
            self.cross_val_results.append(avg_metrics)
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Error in cross validation: {str(e)}")
            return {}
            
    async def optimize_hyperparameters(self) -> Dict[str, Any]:
        """Perform hyperparameter optimization"""
        try:
            # Generate hyperparameter combinations
            hp_configs = self._generate_hp_configs()
            
            for config in hp_configs:
                # Apply config
                self._apply_hp_config(config)
                
                # Evaluate config
                metrics = await self.perform_cross_validation()
                score = metrics['val_loss']
                
                # Update best config
                if score < self.best_hp_score:
                    self.best_hp_score = score
                    self.best_hp_config = config
                    
            # Apply best config
            self._apply_hp_config(self.best_hp_config)
            
            return {
                'best_config': self.best_hp_config,
                'best_score': self.best_hp_score
            }
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            return {}
            
    async def _train_fold(
        self,
        train_data: Dict[str, List],
        val_data: Dict[str, List]
    ) -> Dict[str, float]:
        """Train on single cross-validation fold"""
        try:
            # Create temporary buffers
            old_buffer = self.replay_buffer.copy()
            self.replay_buffer = train_data
            
            # Train for one epoch
            metrics = []
            for i in range(0, len(train_data['features']), self.batch_size):
                batch = self.get_training_batch()
                if batch is not None:
                    step_metrics = await self.train_step(batch)
                    metrics.append(step_metrics)
                    
            # Validate
            val_metrics = await self.validation_step(TrainingBatch(
                features=torch.stack(val_data['features']),
                labels=torch.stack(val_data['labels']),
                market_conditions=[],
                timestamps=[]
            ))
            
            # Restore buffer
            self.replay_buffer = old_buffer
            
            return val_metrics
            
        except Exception as e:
            logger.error(f"Error training fold: {str(e)}")
            return {}
            
    def _generate_hp_configs(self) -> List[Dict[str, Any]]:
        """Generate hyperparameter configurations"""
        configs = []
        for lr in self.hp_search_space['learning_rate']:
            for bs in self.hp_search_space['batch_size']:
                for wd in self.hp_search_space['weight_decay']:
                    configs.append({
                        'learning_rate': lr,
                        'batch_size': bs,
                        'weight_decay': wd
                    })
        return configs
        
    def _apply_hp_config(self, config: Optional[Dict[str, Any]]) -> None:
        """Apply hyperparameter configuration"""
        try:
            if config is None:
                return
                
            self.model.learning_rate = config['learning_rate']
            self.batch_size = config['batch_size']
            
            # Update optimizer
            self.model.optimizer = Adam(
                self.model.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
            
        except Exception as e:
            logger.error(f"Error applying HP config: {str(e)}")

    async def _calculate_task_loss(
        self,
        batch: TrainingBatch,
        task: str
    ) -> torch.Tensor:
        """Calculate loss for specific task"""
        try:
            predictions = cast(Dict[str, torch.Tensor], self.model(batch.features))
            
            if task == 'market':
                return self._calculate_market_loss(predictions, batch.labels)
            elif task == 'path':
                return self._calculate_path_loss(predictions, batch.labels)
            elif task == 'risk':
                return self._calculate_risk_loss(predictions, batch.labels)
            elif task == 'execution':
                return self._calculate_execution_loss(predictions, batch.labels)
            else:
                return torch.tensor(0.0, device=self.device)
                
        except Exception as e:
            logger.error(f"Error calculating task loss: {str(e)}")
            return torch.tensor(0.0, device=self.device)
            
    def _uniswap_loss(self, batch: TrainingBatch) -> torch.Tensor:
        """Uniswap-specific loss calculation"""
        try:
            predictions = self.model(
                batch.features,
                protocol='uniswap'
            )
            # Implement Uniswap-specific loss
            return torch.tensor(0.0, device=self.device)
        except Exception:
            return torch.tensor(0.0, device=self.device)
            
    def _sushiswap_loss(self, batch: TrainingBatch) -> torch.Tensor:
        """Sushiswap-specific loss calculation"""
        try:
            predictions = self.model(
                batch.features,
                protocol='sushiswap'
            )
            # Implement Sushiswap-specific loss
            return torch.tensor(0.0, device=self.device)
        except Exception:
            return torch.tensor(0.0, device=self.device)
            
    def _balancer_loss(self, batch: TrainingBatch) -> torch.Tensor:
        """Balancer-specific loss calculation"""
        try:
            predictions = self.model(
                batch.features,
                protocol='balancer'
            )
            # Implement Balancer-specific loss
            return torch.tensor(0.0, device=self.device)
        except Exception:
            return torch.tensor(0.0, device=self.device)
            
    def _update_performance_history(self, metrics: Dict[str, float]) -> None:
        """Update historical performance tracking"""
        try:
            # Update task metrics
            for task in self.task_weights.keys():
                if f"{task}_loss" in metrics:
                    self.metrics[f"{task}_metrics"].append(
                        metrics[f"{task}_loss"]
                    )
                    
            # Update protocol metrics
            for protocol in self.protocol_loss_fns.keys():
                if f"{protocol}_loss" in metrics:
                    if protocol not in self.metrics['protocol_metrics']:
                        self.metrics['protocol_metrics'][protocol] = []
                    self.metrics['protocol_metrics'][protocol].append(
                        metrics[f"{protocol}_loss"]
                    )
                    
            # Cleanup old history
            self._cleanup_performance_history()
            
        except Exception as e:
            logger.error(f"Error updating performance history: {str(e)}")
            
    def _cleanup_performance_history(self) -> None:
        """Cleanup historical performance data"""
        try:
            max_history = 1000
            
            # Cleanup task metrics
            for task in self.task_weights.keys():
                key = f"{task}_metrics"
                if len(self.metrics[key]) > max_history:
                    self.metrics[key] = \
                        self.metrics[key][-max_history:]
                    
            # Cleanup protocol metrics
            for protocol in self.protocol_loss_fns.keys():
                if protocol in self.metrics['protocol_metrics']:
                    if len(self.metrics['protocol_metrics'][protocol]) > max_history:
                        self.metrics['protocol_metrics'][protocol] = \
                            self.metrics['protocol_metrics'][protocol][-max_history:]
                            
        except Exception as e:
            logger.error(f"Error cleaning up performance history: {str(e)}")

    def _prepare_validation_batch(
        self,
        validation_data: List[MarketDataType]
    ) -> Optional[TrainingBatch]:
        """Convert validation data to training batch format"""
        try:
            if not validation_data:
                return None
                
            # Extract features from validation data
            features = []
            labels = {
                'market_targets': [],
                'path_targets': [],
                'risk_targets': [],
                'execution_targets': []
            }
            
            for data in validation_data:
                # Extract features from market data
                feature = self._extract_features(data)
                features.append(feature)
                
                # Extract validation targets if available
                if data.get('actual_market') is not None:
                    labels['market_targets'].append(data['actual_market'])
                if data.get('actual_path') is not None:
                    labels['path_targets'].append(data['actual_path'])
                if data.get('actual_risk') is not None:
                    labels['risk_targets'].append(data['actual_risk'])
                if data.get('actual_execution') is not None:
                    labels['execution_targets'].append(data['actual_execution'])
            
            # Convert lists to tensors
            features_tensor = torch.stack(features)
            labels_tensor = {
                key: torch.stack(values) if values else torch.zeros(len(features), device=self.device)
                for key, values in labels.items()
            }
            
            return TrainingBatch(
                features=features_tensor,
                labels=labels_tensor,
                market_conditions=validation_data,
                timestamps=[time.time() for _ in validation_data],
                execution_results=[]
            )
            
        except Exception as e:
            logger.error(f"Error preparing validation batch: {str(e)}")
            return None
            
    def _extract_features(self, market_data: MarketDataType) -> torch.Tensor:
        """Extract features from market data"""
        try:
            # Convert market data to tensor format
            features = [
                float(market_data.get('price', 0)),
                float(market_data.get('volume_24h', 0)),
                float(market_data.get('liquidity', 0)),
                float(market_data.get('volatility', 0)),
                float(market_data.get('market_cap', 0)),
                float(market_data.get('tvl', 0)),
                float(market_data.get('fees_24h', 0)),
                float(market_data.get('gas_price', 0)),
                float(market_data.get('block_time', 0)),
                float(market_data.get('network_load', 0)),
                float(market_data.get('pending_tx_count', 0))
            ]
            return torch.tensor(features, device=self.device)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return torch.zeros(11, device=self.device)  # Return zero tensor with expected feature size 