import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from collections import deque
import numpy as np
import os
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class TrainingManager:
    def __init__(self, model: nn.Module, device: torch.device,
                 buffer_size: int = 10000,
                 batch_size: int = 64,
                 learning_rate: float = 0.001,
                 checkpoint_dir: str = 'checkpoints',
                 patience: int = 5,
                 min_delta: float = 0.001):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.min_delta = min_delta
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Initialize experience replay buffer
        self.replay_buffer = {
            'state': deque(maxlen=buffer_size),
            'action': deque(maxlen=buffer_size),
            'reward': deque(maxlen=buffer_size),
            'next_state': deque(maxlen=buffer_size),
            'done': deque(maxlen=buffer_size)
        }
        
        # Training metrics
        self.metrics = {
            'loss_history': [],
            'reward_history': [],
            'success_rate': []
        }
        
        # Early stopping variables
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_model_path = None
        
    def store_experience(self, state: torch.Tensor, action: torch.Tensor,
                        reward: float, next_state: torch.Tensor, done: bool):
        """Store experience in replay buffer"""
        self.replay_buffer['state'].append(state)
        self.replay_buffer['action'].append(action)
        self.replay_buffer['reward'].append(reward)
        self.replay_buffer['next_state'].append(next_state)
        self.replay_buffer['done'].append(done)
        
    def get_training_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get a batch of experiences for training"""
        if len(self.replay_buffer['state']) < self.batch_size:
            return None
            
        # Sample random indices
        indices = np.random.choice(
            len(self.replay_buffer['state']),
            self.batch_size,
            replace=False
        )
        
        # Prepare batch
        batch = {
            'state': torch.stack([self.replay_buffer['state'][i] for i in indices]).to(self.device),
            'action': torch.stack([self.replay_buffer['action'][i] for i in indices]).to(self.device),
            'reward': torch.tensor([self.replay_buffer['reward'][i] for i in indices], device=self.device),
            'next_state': torch.stack([self.replay_buffer['next_state'][i] for i in indices]).to(self.device),
            'done': torch.tensor([self.replay_buffer['done'][i] for i in indices], device=self.device)
        }
        
        return batch
        
    async def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step with early stopping"""
        try:
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch['state'])
            
            # Calculate loss
            loss = self._calculate_loss(predictions, batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update learning rate
            self.scheduler.step(loss)
            
            # Update metrics
            current_loss = loss.item()
            self.metrics['loss_history'].append(current_loss)
            self.metrics['reward_history'].append(batch['reward'].mean().item())
            
            # Check for improvement
            if self._is_improvement(current_loss):
                self.best_loss = current_loss
                self.patience_counter = 0
                await self.save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                
            # Check for early stopping
            if self.patience_counter >= self.patience:
                logger.info("Early stopping triggered")
                await self.load_best_model()
                
            return {
                'loss': current_loss,
                'avg_reward': batch['reward'].mean().item(),
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            return {'loss': float('inf'), 'avg_reward': 0.0, 'learning_rate': self.learning_rate}
            
    def _is_improvement(self, current_loss: float) -> bool:
        """Check if current loss is an improvement"""
        return current_loss < (self.best_loss - self.min_delta)
            
    def _calculate_loss(self, predictions: torch.Tensor,
                       batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate the loss for training"""
        criterion = nn.MSELoss()
        target = batch['reward'].unsqueeze(1)
        return criterion(predictions, target)
        
    async def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, filename)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'metrics': self.metrics,
                'best_loss': self.best_loss,
                'timestamp': datetime.now().isoformat()
            }, checkpoint_path)
            
            self.best_model_path = checkpoint_path
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            
    async def load_best_model(self):
        """Load the best model checkpoint"""
        try:
            if self.best_model_path and os.path.exists(self.best_model_path):
                checkpoint = torch.load(self.best_model_path)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(f"Loaded best model from {self.best_model_path}")
                
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
        
    def get_metrics(self) -> Dict[str, List[float]]:
        """Get current training metrics"""
        return self.metrics
        
    def update_success_rate(self, success_rate: float):
        """Update the success rate metric"""
        self.metrics['success_rate'].append(success_rate)
        
    def save_metrics(self, filename: str = 'training_metrics.json'):
        """Save training metrics to file"""
        try:
            metrics_path = os.path.join(self.checkpoint_dir, filename)
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f)
            logger.info(f"Saved metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")