from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic, Protocol
from torch import Tensor
from ..nn import _Module
from ..optim import Optimizer

Experience = Tuple[Tensor, Tensor, float, Tensor, bool]  # (state, action, reward, next_state, done)
Policy = TypeVar('Policy', bound=_Module)
Value = TypeVar('Value', bound=_Module)

class ReplayBuffer:
    """Experience replay buffer for RL training.
    
    Stores and samples transitions for off-policy learning.
    Implements prioritized experience replay.
    """
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        use_priorities: bool = True
    ) -> None: ...
    
    def add(self, experience: Experience) -> None: ...
    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]: ...
    def update_priorities(self, indices: Tensor, priorities: Tensor) -> None: ...
    def clear(self) -> None: ...

class RLAgent(Generic[Policy, Value]):
    """Base class for reinforcement learning agents.
    
    Implements core RL functionality with support for various
    algorithms and architectures.
    """
    policy_net: Policy
    value_net: Optional[Value]
    target_net: Optional[Policy]
    optimizer: Optimizer
    replay_buffer: ReplayBuffer
    
    def __init__(
        self,
        policy_net: Policy,
        value_net: Optional[Value] = None,
        replay_buffer_size: int = 1000000,
        batch_size: int = 128,
        discount_factor: float = 0.99,
        update_freq: int = 1000
    ) -> None: ...
    
    def select_action(self, state: Tensor, explore: bool = True) -> Tensor: ...
    def update(self, batch: Tuple[Tensor, ...]) -> Dict[str, float]: ...
    def train(self, env: Any, num_episodes: int) -> List[float]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

class PPOAgent(RLAgent[Policy, Value]):
    """Proximal Policy Optimization agent.
    
    Implements PPO with clipped objective and value function estimation.
    Supports continuous action spaces.
    """
    def __init__(
        self,
        policy_net: Policy,
        value_net: Value,
        clip_ratio: float = 0.2,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ) -> None: ...
    
    def compute_advantages(
        self,
        rewards: Tensor,
        values: Tensor,
        dones: Tensor,
        gamma: float = 0.99,
        lambda_: float = 0.95
    ) -> Tensor: ...
    
    def update_policy(
        self,
        states: Tensor,
        actions: Tensor,
        old_log_probs: Tensor,
        advantages: Tensor,
        num_epochs: int = 10
    ) -> Dict[str, float]: ...

class SACAgent(RLAgent[Policy, Value]):
    """Soft Actor-Critic agent.
    
    Implements SAC with automatic temperature tuning and
    twin Q-functions for reduced variance.
    """
    def __init__(
        self,
        policy_net: Policy,
        q_net1: Value,
        q_net2: Value,
        alpha_lr: float = 3e-4,
        policy_lr: float = 3e-4,
        q_lr: float = 3e-4,
        target_entropy: Optional[float] = None
    ) -> None: ...
    
    def update_critic(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor
    ) -> Dict[str, float]: ...
    
    def update_actor_and_alpha(
        self,
        states: Tensor
    ) -> Dict[str, float]: ...

class TD3Agent(RLAgent[Policy, Value]):
    """Twin Delayed Deep Deterministic Policy Gradient agent.
    
    Implements TD3 with twin critics, delayed policy updates,
    and target policy smoothing for stable learning.
    
    Attributes:
        actor: Policy network for action selection
        critic1: First Q-network for value estimation
        critic2: Second Q-network for value estimation
        target_actor: Target policy network
        target_critic1: Target Q-network 1
        target_critic2: Target Q-network 2
    """
    actor: Policy
    critic1: Value
    critic2: Value
    target_actor: Policy
    target_critic1: Value
    target_critic2: Value
    
    def __init__(
        self,
        actor: Policy,
        critic1: Value,
        critic2: Value,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        policy_delay: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        tau: float = 0.005
    ) -> None: ...
    
    def select_action(
        self,
        state: Tensor,
        noise_scale: float = 0.1
    ) -> Tensor: ...
    
    def update_critics(
        self,
        states: Tensor,
        actions: Tensor,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor
    ) -> Dict[str, float]: ...
    
    def update_actor(
        self,
        states: Tensor
    ) -> Dict[str, float]: ...
    
    def update_target_networks(self) -> None: ...
    
    def add_exploration_noise(
        self,
        action: Tensor,
        noise_scale: float,
        noise_type: str = 'gaussian'
    ) -> Tensor: ...
    
    def compute_target_actions(
        self,
        next_states: Tensor
    ) -> Tensor: ...
    
    def train_step(
        self,
        batch: Experience,
        step: int
    ) -> Dict[str, float]: ... 