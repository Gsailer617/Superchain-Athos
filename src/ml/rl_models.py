"""
Reinforcement Learning models for trading strategy optimization.
This module includes implementations of state-of-the-art RL algorithms:
- Soft Actor-Critic (SAC): For continuous action spaces
- Distributional RL: For better risk assessment with value distribution learning
- Rainbow DQN: For discrete action spaces with multiple enhancements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import random
import logging
from collections import deque, namedtuple
import math

logger = logging.getLogger(__name__)

# For prioritized experience replay
Experience = namedtuple('Experience', 
                        ('state', 'action', 'reward', 'next_state', 'done', 'priority'))

class NoisyLinear(nn.Module):
    """Noisy Linear layer for exploration in Rainbow DQN"""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.sample_noise()
    
    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
        
    def sample_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(x, 
                            self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class ReplayBuffer:
    """Experience replay buffer for reinforcement learning"""
    
    def __init__(
        self, 
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: torch.device = torch.device("cpu")
    ):
        self.capacity = capacity
        self.device = device
        
        # Storage for experiences
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((capacity, 1), dtype=torch.bool, device=device)
        
        self.ptr = 0
        self.size = 0
        
    def add(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        reward: float, 
        next_state: torch.Tensor,
        done: bool
    ):
        """Add experience to buffer"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = torch.tensor([reward], device=self.device)
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = torch.tensor([done], device=self.device)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of experiences"""
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices]
        }
        
    def __len__(self) -> int:
        return self.size


class SoftActorCritic(nn.Module):
    """Soft Actor-Critic implementation for trading strategy optimization
    
    SAC is an off-policy actor-critic algorithm that:
    1. Uses entropy regularization for exploration
    2. Leverages twin Q-networks for reduced bias
    3. Adapts to both discrete and continuous action spaces
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        lr: float = 3e-4,
        buffer_size: int = 100000,
        action_space: str = "continuous",
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim, self.device)
        
        # Initialize policy network (actor)
        if action_space == "continuous":
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            
            self.mean = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Linear(hidden_dim, action_dim)
        else:  # discrete actions
            self.actor = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        # Twin Q-networks (critics)
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Target networks
        self.critic1_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.critic2_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Copy parameters to target networks
        self._copy_weights(self.critic1, self.critic1_target)
        self._copy_weights(self.critic2, self.critic2_target)
        
        # Entropy coefficient
        self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.target_entropy = -action_dim  # Heuristic value
        
        # Optimizers
        if action_space == "continuous":
            self.actor_optimizer = torch.optim.Adam(
                list(self.actor.parameters()) + 
                list(self.mean.parameters()) + 
                list(self.log_std.parameters()), 
                lr=lr
            )
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
            
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
        # Move to device
        self.to(self.device)
    
    def _copy_weights(self, source: nn.Module, target: nn.Module):
        """Copy weights from source to target network"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(source_param.data)
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network parameters: θ_target = τ*θ_source + (1-τ)*θ_target"""
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
            
    def act(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Select action given the current state"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
                
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            if self.action_space == "continuous":
                features = self.actor(state)
                mean = self.mean(features)
                
                if deterministic:
                    # Use mean directly for deterministic actions
                    action = torch.tanh(mean)
                else:
                    # Sample from distribution for exploration
                    log_std = self.log_std(features)
                    log_std = torch.clamp(log_std, -20, 2)
                    std = torch.exp(log_std)
                    
                    normal = torch.distributions.Normal(mean, std)
                    x_t = normal.rsample()  # Reparameterized sample
                    action = torch.tanh(x_t)
            else:
                logits = self.actor(state)
                
                if deterministic:
                    # Select highest probability action
                    action = torch.argmax(logits, dim=1, keepdim=True)
                    
                    # One-hot encode action
                    action_one_hot = torch.zeros_like(logits)
                    action_one_hot.scatter_(1, action, 1.0)
                    action = action_one_hot
                else:
                    # Sample from softmax distribution
                    probs = F.softmax(logits, dim=-1)
                    action_dist = torch.distributions.Categorical(probs)
                    action_idx = action_dist.sample().unsqueeze(1)
                    
                    # One-hot encode action
                    action = torch.zeros_like(logits)
                    action.scatter_(1, action_idx, 1.0)
                    
            return action.cpu().numpy() if action.shape[0] == 1 else action.squeeze(0).cpu().numpy()
            
    def evaluate(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate state value using critic networks"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
                
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # Get action for state
            if self.action_space == "continuous":
                features = self.actor(state)
                mean = self.mean(features)
                action = torch.tanh(mean)
            else:
                logits = self.actor(state)
                action = F.softmax(logits, dim=-1)
            
            # Evaluate Q-values
            state_action = torch.cat([state, action], dim=1)
            q1 = self.critic1(state_action)
            q2 = self.critic2(state_action)
            
            # Return minimum Q-value as conservative estimate
            return torch.min(q1, q2)
    
    def sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from current policy with log probability"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        if self.action_space == "continuous":
            features = self.actor(state)
            mean = self.mean(features)
            log_std = self.log_std(features)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # Reparameterized sample
            action = torch.tanh(x_t)
            
            # Calculate log probability, using change of variables formula for tanh
            log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=1, keepdim=True)
            
            return action, log_prob, mean
        else:
            logits = self.actor(state)
            probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(probs)
            action_idx = action_dist.sample().unsqueeze(1)
            
            # One-hot encode action
            action = torch.zeros_like(logits)
            action.scatter_(1, action_idx, 1.0)
            
            # Calculate log probability
            log_prob = torch.log(probs.gather(1, action_idx) + 1e-6)
            
            return action, log_prob, probs
            
    def update(self, batch_size: int) -> Dict[str, float]:
        """Update the model parameters using batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return {
                'actor_loss': 0.0,
                'critic1_loss': 0.0,
                'critic2_loss': 0.0,
                'alpha_loss': 0.0
            }
            
        # Sample from replay buffer
        batch = self.replay_buffer.sample(batch_size)
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        with torch.no_grad():
            # Sample next actions and their log probs from current policy
            next_actions, next_log_probs, _ = self.sample_action(next_states)
            
            # Compute target Q-values
            next_state_actions = torch.cat([next_states, next_actions], dim=1)
            next_q1 = self.critic1_target(next_state_actions)
            next_q2 = self.critic2_target(next_state_actions)
            next_q = torch.min(next_q1, next_q2) - self.log_alpha.exp() * next_log_probs
            
            # Target Q = r + γ * next_q * (1 - done)
            target_q = rewards + self.gamma * (1 - dones.float()) * next_q
            
        # Update critic networks
        state_actions = torch.cat([states, actions], dim=1)
        
        # Current Q estimates
        current_q1 = self.critic1(state_actions)
        current_q2 = self.critic2(state_actions)
        
        # Compute MSE loss for both critics
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update first critic
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        # Update second critic
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor network
        new_actions, log_probs, _ = self.sample_action(states)
        new_state_actions = torch.cat([states, new_actions], dim=1)
        q1 = self.critic1(new_state_actions)
        q2 = self.critic2(new_state_actions)
        min_q = torch.min(q1, q2)
        
        # Actor loss = -E[Q - α*log_π]
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_probs - min_q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha (entropy coefficient)
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item()
        }
        
    def store_transition(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        reward: float, 
        next_state: torch.Tensor,
        done: bool
    ):
        """Store transition in replay buffer"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action).to(self.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).to(self.device)
            
        self.replay_buffer.add(state, action, reward, next_state, done)
        
    def save(self, path: str):
        """Save model parameters"""
        torch.save({
            'actor': self.actor.state_dict(),
            'mean': self.mean.state_dict() if self.action_space == "continuous" else None,
            'log_std': self.log_std.state_dict() if self.action_space == "continuous" else None,
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic1_target': self.critic1_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, path)
        
    def load(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        if self.action_space == "continuous":
            self.mean.load_state_dict(checkpoint['mean'])
            self.log_std.load_state_dict(checkpoint['log_std'])
            
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target'])
        self.log_alpha = checkpoint['log_alpha']
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        

class ModelBasedRL(nn.Module):
    """Model-based reinforcement learning for market prediction and strategy optimization
    
    Combines a world model for simulation with a policy network for action selection.
    Allows planning multiple steps ahead using the model.
    """
    
    def __init__(
        self, 
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        model_hidden_dim: int = 512,
        ensemble_size: int = 5,
        horizon: int = 5,
        gamma: float = 0.99,
        learning_rate: float = 3e-4,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.gamma = gamma
        self.ensemble_size = ensemble_size
        
        # World model ensemble (for reducing model bias)
        self.dynamics_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + action_dim, model_hidden_dim),
                nn.ReLU(),
                nn.Linear(model_hidden_dim, model_hidden_dim),
                nn.ReLU(),
                nn.Linear(model_hidden_dim, state_dim + 1)  # +1 for reward prediction
            )
            for _ in range(ensemble_size)
        ])
        
        # Reward model (optional - can also use the output from dynamics model)
        self.reward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Linear(hidden_dim, action_dim)
        
        # Value function
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Optimizers
        self.dynamics_optimizer = torch.optim.Adam(
            list(self.dynamics_models.parameters()) + list(self.reward_model.parameters()),
            lr=learning_rate
        )
        
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + 
            list(self.action_mean.parameters()) + 
            list(self.action_log_std.parameters()),
            lr=learning_rate
        )
        
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(),
            lr=learning_rate
        )
        
        # Experience buffer for model training
        self.model_buffer = []
        
        # Move to device
        self.to(self.device)
        
    def predict_next_state(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        use_ensemble: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next state and reward using dynamics model"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.FloatTensor(action).to(self.device)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        # Concatenate state and action
        state_action = torch.cat([state, action], dim=1)
        
        if use_ensemble:
            # Get predictions from all models
            next_states = []
            rewards = []
            
            for model in self.dynamics_models:
                output = model(state_action)
                delta_state = output[:, :-1]  # State change prediction
                reward = output[:, -1:]  # Reward prediction
                
                next_state = state + delta_state
                
                next_states.append(next_state)
                rewards.append(reward)
            
            # Stack predictions
            next_states = torch.stack(next_states)
            rewards = torch.stack(rewards)
            
            # Get mean and variance
            mean_next_state = next_states.mean(dim=0)
            var_next_state = next_states.var(dim=0)
            mean_reward = rewards.mean(dim=0)
            
            return mean_next_state, mean_reward, var_next_state
        else:
            # Use a single model (for faster computation)
            model_idx = np.random.randint(0, self.ensemble_size)
            output = self.dynamics_models[model_idx](state_action)
            
            delta_state = output[:, :-1]
            reward = output[:, -1:]
            
            next_state = state + delta_state
            
            return next_state, reward, torch.zeros_like(next_state)
    
    def plan_trajectory(
        self, 
        state: torch.Tensor, 
        horizon: Optional[int] = None,
        num_samples: int = 100,
        top_k: int = 10,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """Plan trajectory using model predictive control
        
        Args:
            state: Current state
            horizon: Planning horizon (number of steps to look ahead)
            num_samples: Number of action sequences to sample
            top_k: Number of best trajectories to refine
            temperature: Temperature for sampling actions
            
        Returns:
            best_action: First action in the best trajectory
            best_value: Predicted value of the best trajectory
        """
        if horizon is None:
            horizon = self.horizon
            
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.device)
            
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Sample initial action sequences
        batch_size = num_samples
        action_dim = self.action_dim
        
        # Use current policy to initialize action sequences
        policy_features = self.policy(state)
        action_mean = self.action_mean(policy_features)
        action_log_std = self.action_log_std(policy_features)
        action_log_std = torch.clamp(action_log_std, -20, 2)
        action_std = torch.exp(action_log_std) * temperature
        
        # Sample action sequences
        action_sequences = []
        for _ in range(horizon):
            normal_dist = torch.distributions.Normal(action_mean, action_std)
            actions = normal_dist.sample(torch.Size([batch_size]))
            actions = torch.tanh(actions)  # Bound actions
            action_sequences.append(actions)
            
        action_sequences = torch.stack(action_sequences, dim=1)  # [batch_size, horizon, action_dim]
        
        # Evaluate action sequences using model
        returns = torch.zeros(batch_size, 1, device=self.device)
        states = state.repeat(batch_size, 1)
        
        # Simulate trajectories
        for t in range(horizon):
            actions = action_sequences[:, t]
            next_states, rewards, _ = self.predict_next_state(states, actions, use_ensemble=False)
            
            # Update returns
            returns += (self.gamma ** t) * rewards
            
            # Update states
            states = next_states
        
        # Add final state value
        final_values = self.value(states)
        returns += (self.gamma ** horizon) * final_values
        
        # Select top-k trajectories
        _, top_indices = torch.topk(returns.squeeze(), k=top_k)
        best_action = action_sequences[top_indices[0], 0]
        best_value = returns[top_indices[0]].item()
        
        return best_action, best_value
        
    def act(self, state: torch.Tensor, deterministic: bool = False, plan: bool = False) -> torch.Tensor:
        """Select action given the current state"""
        if plan:
            # Use planning to select action
            best_action, _ = self.plan_trajectory(state)
            return best_action.cpu().numpy()
        
        # Use policy directly
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
                
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            policy_features = self.policy(state)
            mean = self.action_mean(policy_features)
            
            if deterministic:
                # Use mean directly for deterministic actions
                action = torch.tanh(mean)
            else:
                # Sample from distribution for exploration
                log_std = self.action_log_std(policy_features)
                log_std = torch.clamp(log_std, -20, 2)
                std = torch.exp(log_std)
                
                normal = torch.distributions.Normal(mean, std)
                x_t = normal.rsample()  # Reparameterized sample
                action = torch.tanh(x_t)
                
            return action.cpu().numpy()[0]
            
    def update_model(self, state_batch: torch.Tensor, action_batch: torch.Tensor, 
                    next_state_batch: torch.Tensor, reward_batch: torch.Tensor) -> float:
        """Update dynamics model using collected experience"""
        # Concatenate state and action
        state_action = torch.cat([state_batch, action_batch], dim=1)
        
        # Target is the state difference and reward
        delta_state = next_state_batch - state_batch
        targets = torch.cat([delta_state, reward_batch], dim=1)
        
        # Update each model in the ensemble
        total_loss = 0.0
        for model in self.dynamics_models:
            # Model prediction
            predictions = model(state_action)
            
            # MSE loss
            loss = F.mse_loss(predictions, targets)
            
            total_loss += loss.item()
            
            # Update model
            self.dynamics_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.dynamics_optimizer.step()
            
        return total_loss / self.ensemble_size
        
    def update_policy(self, state_batch: torch.Tensor) -> float:
        """Update policy using model-based planning"""
        # Get policy distribution
        policy_features = self.policy(state_batch)
        mean = self.action_mean(policy_features)
        log_std = self.action_log_std(policy_features)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        
        normal = torch.distributions.Normal(mean, std)
        actions = normal.rsample()  # Reparameterized sample
        actions = torch.tanh(actions)
        
        # Predict next states and rewards
        next_states, rewards, _ = self.predict_next_state(state_batch, actions, use_ensemble=True)
        
        # Calculate value of next states
        next_values = self.value(next_states).detach()
        
        # Calculate policy objective (maximize)
        policy_objective = rewards + self.gamma * next_values
        policy_loss = -policy_objective.mean()
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item()
        
    def update_value(self, state_batch: torch.Tensor) -> float:
        """Update value function for better planning"""
        with torch.no_grad():
            # Get policy distribution
            policy_features = self.policy(state_batch)
            mean = self.action_mean(policy_features)
            log_std = self.action_log_std(policy_features)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
            
            normal = torch.distributions.Normal(mean, std)
            actions = normal.rsample()  # Reparameterized sample
            actions = torch.tanh(actions)
            
            # Predict next states and rewards
            next_states, rewards, _ = self.predict_next_state(state_batch, actions, use_ensemble=True)
            
            # Calculate target values
            next_values = self.value(next_states).detach()
            target_values = rewards + self.gamma * next_values
            
        # Current value estimates
        current_values = self.value(state_batch)
        
        # Value loss
        value_loss = F.mse_loss(current_values, target_values)
        
        # Update value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        return value_loss.item()
        
    def update(self, state_batch: torch.Tensor, action_batch: torch.Tensor, 
              next_state_batch: torch.Tensor, reward_batch: torch.Tensor) -> Dict[str, float]:
        """Update all components of the model-based RL system"""
        # Update dynamics model
        model_loss = self.update_model(state_batch, action_batch, next_state_batch, reward_batch)
        
        # Update policy
        policy_loss = self.update_policy(state_batch)
        
        # Update value function
        value_loss = self.update_value(state_batch)
        
        return {
            'model_loss': model_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss
        }
        
    def save(self, path: str):
        """Save model parameters"""
        torch.save({
            'dynamics_models': [model.state_dict() for model in self.dynamics_models],
            'reward_model': self.reward_model.state_dict(),
            'policy': self.policy.state_dict(),
            'action_mean': self.action_mean.state_dict(),
            'action_log_std': self.action_log_std.state_dict(),
            'value': self.value.state_dict(),
            'dynamics_optimizer': self.dynamics_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        """Load model parameters"""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, state_dict in enumerate(checkpoint['dynamics_models']):
            self.dynamics_models[i].load_state_dict(state_dict)
            
        self.reward_model.load_state_dict(checkpoint['reward_model'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.action_mean.load_state_dict(checkpoint['action_mean'])
        self.action_log_std.load_state_dict(checkpoint['action_log_std'])
        self.value.load_state_dict(checkpoint['value'])
        
        self.dynamics_optimizer.load_state_dict(checkpoint['dynamics_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for more efficient learning"""
    
    def __init__(
        self, 
        capacity: int, 
        alpha: float = 0.6, 
        beta: float = 0.4, 
        beta_annealing: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        Initialize a prioritized replay buffer
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent (0 = uniform sampling, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_annealing: Rate to anneal beta towards 1
            epsilon: Small value to avoid zero priority
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon
        self.max_priority = 1.0
        
    def add(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor, 
        reward: float, 
        next_state: torch.Tensor, 
        done: bool
    ) -> None:
        """Add a new experience to memory with max priority"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = Experience(state, action, reward, next_state, done, self.max_priority)
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int, device: torch.device) -> Tuple:
        """Sample a batch of experiences based on priorities"""
        if len(self.memory) < batch_size:
            indices = np.random.choice(len(self.memory), batch_size, replace=True)
        else:
            probs = self.priorities[:len(self.memory)] ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
            
        # Calculate importance sampling weights
        weights = (len(self.memory) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Anneal beta towards 1
        self.beta = min(1.0, self.beta + self.beta_annealing)
        
        # Get experiences
        experiences = [self.memory[idx] for idx in indices]
        
        # Convert to tensors
        states = torch.stack([e.state for e in experiences]).to(device)
        actions = torch.stack([e.action for e in experiences]).to(device)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32).to(device)
        next_states = torch.stack([e.next_state for e in experiences]).to(device)
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        
        return states, actions, rewards, next_states, dones, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self) -> int:
        return len(self.memory)

class RainbowDQN(nn.Module):
    """
    Rainbow DQN combining multiple improvements to DQN:
    - Prioritized Experience Replay
    - Dueling Network Architecture
    - Noisy Networks for Exploration
    - Distributional RL
    - Multi-step learning
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        atom_size: int = 51,  # Number of atoms for distributional Q-learning
        v_min: float = -10.0,  # Minimum value for support
        v_max: float = 10.0,   # Maximum value for support
        noisy: bool = True,    # Use noisy networks for exploration
        device: torch.device = None
    ):
        """
        Initialize Rainbow DQN model
        
        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            hidden_dims: Hidden layer dimensions
            atom_size: Number of atoms for distributional Q-learning
            v_min: Minimum value for support
            v_max: Maximum value for support
            noisy: Whether to use noisy linear layers
            device: Device to run the model on
        """
        super(RainbowDQN, self).__init__()
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.noisy = noisy
        
        # Support for distributional RL
        self.register_buffer("support", torch.linspace(v_min, v_max, atom_size))
        self.delta_z = (v_max - v_min) / (atom_size - 1)
        
        # Feature layer
        feature_layers = []
        prev_dim = state_dim
        
        for dim in hidden_dims:
            if noisy:
                feature_layers.append(NoisyLinear(prev_dim, dim))
            else:
                feature_layers.append(nn.Linear(prev_dim, dim))
            feature_layers.append(nn.ReLU())
            prev_dim = dim
            
        self.feature_layer = nn.Sequential(*feature_layers)
        
        # Dueling architecture
        if noisy:
            self.advantage_layer = NoisyLinear(hidden_dims[-1], action_dim * atom_size)
            self.value_layer = NoisyLinear(hidden_dims[-1], atom_size)
        else:
            self.advantage_layer = nn.Linear(hidden_dims[-1], action_dim * atom_size)
            self.value_layer = nn.Linear(hidden_dims[-1], atom_size)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            state: State tensor
            
        Returns:
            Tensor of shape (batch_size, action_dim, atom_size) with distributional Q-values
        """
        features = self.feature_layer(state)
        
        # Dueling architecture
        advantage = self.advantage_layer(features).view(-1, self.action_dim, self.atom_size)
        value = self.value_layer(features).view(-1, 1, self.atom_size)
        
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probabilities
        q_dist = F.softmax(q_atoms, dim=2)
        
        return q_dist
    
    def reset_noise(self) -> None:
        """Reset noise for exploration"""
        if not self.noisy:
            return
            
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.sample_noise()
                
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get expected Q-values by computing the expectation of the value distribution
        
        Args:
            state: State tensor
            
        Returns:
            Tensor of shape (batch_size, action_dim) with expected Q-values
        """
        q_dist = self.forward(state)
        support = self.support.expand_as(q_dist)
        q_values = torch.sum(q_dist * support, dim=2)
        
        return q_values
    
    def select_action(self, state: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """
        Select action based on epsilon-greedy policy
        
        Args:
            state: State tensor
            evaluate: Whether to evaluate (no exploration)
            
        Returns:
            Selected action
        """
        if evaluate:
            q_values = self.get_q_values(state)
            action = q_values.argmax(dim=1)
        else:
            # For noisy networks, exploration is built-in via noise
            q_values = self.get_q_values(state)
            action = q_values.argmax(dim=1)
            
        return action
        
class DistributionalSoftActorCritic(nn.Module):
    """
    Distributional Soft Actor-Critic for continuous action spaces with
    value distribution learning for better risk assessment
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        atom_size: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        auto_entropy_tuning: bool = True,
        device: torch.device = None,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        """
        Initialize Distributional SAC
        
        Args:
            state_dim: Dimensionality of state space
            action_dim: Dimensionality of action space
            hidden_dims: Hidden layer dimensions
            atom_size: Number of atoms for value distribution
            v_min: Minimum value for support
            v_max: Maximum value for support
            alpha: Temperature parameter for entropy
            gamma: Discount factor
            tau: Soft update factor
            auto_entropy_tuning: Whether to automatically tune entropy
            device: Device to run the model on
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
        """
        super(DistributionalSoftActorCritic, self).__init__()
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (atom_size - 1)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.auto_entropy_tuning = auto_entropy_tuning
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Register value distribution support
        self.register_buffer("support", torch.linspace(v_min, v_max, atom_size))
        
        # Policy network (Actor)
        self.policy = PolicyNetwork(
            state_dim, 
            action_dim, 
            hidden_dims,
            log_std_min,
            log_std_max
        ).to(device)
        
        # Distributional Q networks (Critics)
        self.critic1 = DistributionalQNetwork(state_dim, action_dim, hidden_dims, atom_size).to(device)
        self.critic2 = DistributionalQNetwork(state_dim, action_dim, hidden_dims, atom_size).to(device)
        
        # Target networks
        self.critic1_target = DistributionalQNetwork(state_dim, action_dim, hidden_dims, atom_size).to(device)
        self.critic2_target = DistributionalQNetwork(state_dim, action_dim, hidden_dims, atom_size).to(device)
        
        # Copy parameters to targets
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Automatic entropy tuning
        if auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.tensor(action_dim, device=device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
    
    def select_action(self, state: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        """
        Select action based on current policy
        
        Args:
            state: State tensor
            evaluate: Whether to evaluate (no exploration)
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            if evaluate:
                mean, _ = self.policy(state)
                return mean
            else:
                action, _, _ = self.policy.sample(state)
                return action
                
    def get_value_distribution(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get value distribution for state-action pair
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Two distributions from both critics
        """
        q_dist1 = self.critic1(state, action)
        q_dist2 = self.critic2(state, action)
        
        return q_dist1, q_dist2
    
    def get_expected_q(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get expected Q-values by computing expectation of distribution
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Expected Q-values from both critics
        """
        q_dist1, q_dist2 = self.get_value_distribution(state, action)
        
        q1 = torch.sum(self.support.expand_as(q_dist1) * q_dist1, dim=1)
        q2 = torch.sum(self.support.expand_as(q_dist2) * q_dist2, dim=1)
        
        return q1, q2
    
    def get_cvar(self, state: torch.Tensor, action: torch.Tensor, alpha: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Conditional Value at Risk (CVaR) for risk assessment
        
        Args:
            state: State tensor
            action: Action tensor
            alpha: Risk level (lower means more risk-averse)
            
        Returns:
            CVaR values from both critics
        """
        q_dist1, q_dist2 = self.get_value_distribution(state, action)
        
        # Compute cumulative probabilities
        cum_probs1 = torch.cumsum(q_dist1, dim=1)
        cum_probs2 = torch.cumsum(q_dist2, dim=1)
        
        # Find cutoff index for alpha
        var_indices1 = torch.sum(cum_probs1 < alpha, dim=1)
        var_indices2 = torch.sum(cum_probs2 < alpha, dim=1)
        
        # Compute CVaR as the mean of values below VaR
        cvar1 = torch.zeros_like(var_indices1, dtype=torch.float32)
        cvar2 = torch.zeros_like(var_indices2, dtype=torch.float32)
        
        for i in range(state.shape[0]):
            idx1 = var_indices1[i].item()
            idx2 = var_indices2[i].item()
            
            if idx1 > 0:
                cvar1[i] = torch.sum(self.support[:idx1] * q_dist1[i, :idx1]) / q_dist1[i, :idx1].sum()
            else:
                cvar1[i] = self.v_min
                
            if idx2 > 0:
                cvar2[i] = torch.sum(self.support[:idx2] * q_dist2[i, :idx2]) / q_dist2[i, :idx2].sum()
            else:
                cvar2[i] = self.v_min
        
        return cvar1, cvar2

class PolicyNetwork(nn.Module):
    """
    Policy Network for the Distributional SAC agent that outputs
    a distribution over actions
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super(PolicyNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        self.feature_layer = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action distribution parameters
        
        Args:
            state: State tensor
            
        Returns:
            Mean and log standard deviation of the action distribution
        """
        features = self.feature_layer(state)
        
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from the distribution
        
        Args:
            state: State tensor
            
        Returns:
            Action sample, log probability, and mean action
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Create normal distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Sample using reparameterization trick
        x_t = normal.rsample()
        
        # Apply tanh squashing for bounded actions
        action = torch.tanh(x_t)
        
        # Compute log probability with change of variables for tanh
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob, mean
        
class DistributionalQNetwork(nn.Module):
    """
    Q-Network that outputs a distribution over values instead of a point estimate
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        atom_size: int = 51
    ):
        super(DistributionalQNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim + action_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            prev_dim = dim
            
        layers.append(nn.Linear(hidden_dims[-1], atom_size))
        
        self.q_net = nn.Sequential(*layers)
        self.atom_size = atom_size
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to get value distribution
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Distribution over Q-values
        """
        x = torch.cat([state, action], dim=1)
        logits = self.q_net(x)
        probs = F.softmax(logits, dim=1)
        
        return probs 