"""
PPO-based Meta-Controller for EFCA-v2.1

Implements Proximal Policy Optimization for the meta-controller
as recommended in specification Section 3:

"PPO (Proximal Policy Optimization) is preferred over standard
Actor-Critic for the Meta-Controller to ensure monotonic improvement."

This implementation provides stable meta-learning with:
- Clipped surrogate objective
- Value function learning
- Advantage estimation (GAE)
- Multiple optimization epochs per update
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple
import numpy as np


class PPOMetaController(nn.Module):
    """
    PPO-based Meta-Controller with clamped outputs.
    
    Combines PPO algorithm with tanh-clamped actions for safe
    parameter updates via ClampedDeltaAdapter.
    """
    
    def __init__(self, config: dict):
        """
        Initialize PPO Meta-Controller.
        
        Args:
            config: Configuration dictionary with:
                - input_dim: Probe network output dimension
                - output_dim: Number of parameters to control
                - hidden_dim: Hidden layer size
                - lr: Learning rate for PPO
                - clip_epsilon: PPO clipping parameter (default: 0.2)
                - gae_lambda: GAE lambda parameter (default: 0.95)
                - value_coef: Value loss coefficient (default: 0.5)
                - entropy_coef: Entropy bonus coefficient (default: 0.01)
        """
        super().__init__()
        
        self.input_dim = config.get('input_dim', 64)
        self.output_dim = config.get('output_dim', 1)
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # PPO hyperparameters
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.mini_batch_size = config.get('mini_batch_size', 64)
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Tanh()  # Clamped output [-1, 1]
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Optimizer
        lr = config.get('lr', 3e-4)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: get action and value.
        
        Args:
            state: Probe network output (B, input_dim)
        
        Returns:
            action: Meta-controller action (B, output_dim)
            value: State value estimate (B, 1)
        """
        action = self.actor(state)
        value = self.critic(state)
        return action, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        For continuous control, we use the actor network output directly
        with optional noise for exploration.
        
        Args:
            state: Probe network output
            deterministic: If True, return mean action without noise
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: Value estimate
        """
        with torch.no_grad():
            action_mean, value = self.forward(state)
        
        if deterministic:
            return action_mean, torch.zeros_like(action_mean), value
        
        # Add small Gaussian noise for exploration
        action_std = 0.1
        noise = torch.randn_like(action_mean) * action_std
        action = torch.clamp(action_mean + noise, -1.0, 1.0)
        
        # Compute log probability (simple Gaussian)
        log_prob = -0.5 * ((action - action_mean) / action_std) ** 2 - np.log(action_std * np.sqrt(2 * np.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        log_prob: torch.Tensor,
        done: bool
    ):
        """Store transition in experience buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value: torch.Tensor, gamma: float = 0.99) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            next_value: Value estimate for next state
            gamma: Discount factor
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        values = self.values + [next_value]
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.cat(advantages)
        returns = advantages + torch.cat(self.values)
        
        return advantages, returns
    
    def update(self, next_value: torch.Tensor, gamma: float = 0.99) -> dict:
        """
        Update policy using PPO algorithm.
        
        Implements clipped surrogate objective for safe policy updates.
        
        Args:
            next_value: Value estimate for final state
            gamma: Discount factor
        
        Returns:
            dict: Training statistics
        """
        if len(self.states) == 0:
            return {}
        
        # Compute advantages
        advantages, returns = self.compute_gae(next_value, gamma)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        old_states = torch.cat(self.states)
        old_actions = torch.cat(self.actions)
        old_log_probs = torch.cat(self.log_probs)
        
        # PPO update epochs
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.ppo_epochs):
            # Get current policy output
            action_mean, values = self.forward(old_states)
            
            # Recompute log probs for current policy
            action_std = 0.1
            log_probs = -0.5 * ((old_actions - action_mean) / action_std) ** 2 - np.log(action_std * np.sqrt(2 * np.pi))
            log_probs = log_probs.sum(dim=-1, keepdim=True)
            
            # Policy loss with PPO clipping
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages.unsqueeze(-1)
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.unsqueeze(-1)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = self.value_coef * nn.functional.mse_loss(values, returns.unsqueeze(-1))
            
            # Entropy bonus (for exploration)
            entropy = -self.entropy_coef * log_probs.mean()
            
            # Total loss
            loss = policy_loss + value_loss - entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        # Clear buffer
        self.clear_buffer()
        
        return {
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'entropy': total_entropy / self.ppo_epochs
        }
    
    def clear_buffer(self):
        """Clear experience buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
