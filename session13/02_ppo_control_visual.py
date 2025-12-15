"""
Proximal Policy Optimization (PPO) for Visual Control
======================================================

This demo illustrates PPO algorithm for learning from pixel observations:
1. CNN encoder for visual feature extraction
2. Actor-Critic architecture with shared features
3. Clipped surrogate objective for stable policy updates
4. Generalized Advantage Estimation (GAE)
5. Visual control task learning

Lecture 13: Visual Policy Learning - From Imitation to Foundation Models
Prof. David Olivieri - UVigo - VIAR25/26

Key Concepts Demonstrated:
--------------------------
- Policy gradient methods for visual control
- Trust region optimization via clipping
- Value function learning for variance reduction
- Advantage estimation for policy improvement
- CNN architectures for processing pixel observations
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class VisualWrapper(gym.Wrapper):
    """
    Wrapper to convert any Gym environment to visual observations.
    Renders the environment and returns RGB frames as observations.
    """
    def __init__(self, env, img_size=(84, 84)):
        super().__init__(env)
        self.img_size = img_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(img_size[0], img_size[1], 3), 
            dtype=np.uint8
        )
        
    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        obs = self._get_visual_obs()
        return obs, info
    
    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self._get_visual_obs()
        return obs, reward, terminated, truncated, info
    
    def _get_visual_obs(self):
        """Render environment and return RGB observation."""
        # Get RGB array from rendering
        frame = self.env.render()
        if frame is None:
            # Fallback if render returns None
            frame = np.zeros((*self.img_size, 3), dtype=np.uint8)
        
        # Resize if needed
        if frame.shape[:2] != self.img_size:
            import cv2
            frame = cv2.resize(frame, self.img_size)
        
        return frame.astype(np.uint8)


def make_visual_env(env_name: str = "CartPole-v1", img_size: int = 84, 
                    stack_frames: int = 4, grayscale: bool = True):
    """
    Create a visual control environment with preprocessing.
    
    Args:
        env_name: Base environment name
        img_size: Size to resize images
        stack_frames: Number of frames to stack
        grayscale: Whether to convert to grayscale
    
    Returns:
        env: Wrapped environment with visual observations
    """
    # Create base environment with rgb_array rendering
    env = gym.make(env_name, render_mode="rgb_array")
    
    # Add visual wrapper
    env = VisualWrapper(env, img_size=(img_size, img_size))
    
    # Convert to grayscale if specified
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=False)
    
    # Stack frames for temporal information
    env = FrameStack(env, num_stack=stack_frames)
    
    return env


class CNNEncoder(nn.Module):
    """
    Convolutional Neural Network encoder for processing visual observations.
    Inspired by Nature DQN architecture.
    """
    def __init__(self, input_channels: int = 4, output_dim: int = 256):
        super(CNNEncoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        
        # Calculate size after convolutions (for 84x84 input)
        def conv_output_size(size, kernel_size, stride, padding=0):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        h = conv_output_size(84, 8, 4)  # After conv1
        h = conv_output_size(h, 4, 2)   # After conv2
        h = conv_output_size(h, 3, 1)   # After conv3
        
        conv_output_size_flat = h * h * 64
        
        # Fully connected layer
        self.fc = nn.Linear(conv_output_size_flat, output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Orthogonal initialization for better training stability."""
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN encoder.
        
        Args:
            x: Visual observations [batch_size, channels, height, width]
        
        Returns:
            features: Encoded features [batch_size, output_dim]
        """
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0
        
        # Convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layer
        features = F.relu(self.fc(x))
        
        return features


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Shares CNN encoder between actor and critic for efficiency.
    """
    def __init__(self, input_channels: int, num_actions: int, 
                 feature_dim: int = 256, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # Shared CNN encoder
        self.encoder = CNNEncoder(input_channels, feature_dim)
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize heads
        self._init_heads()
    
    def _init_heads(self):
        """Initialize actor and critic heads."""
        for module in [self.actor, self.critic]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=0.01)
                    nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic network.
        
        Args:
            obs: Visual observations
        
        Returns:
            action_logits: Logits over actions
            value: State value estimate
        """
        features = self.encoder(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample action from policy.
        
        Args:
            obs: Visual observation
            deterministic: If True, select argmax action
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_logits, value = self.forward(obs)
        probs = F.softmax(action_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)
        
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, action_log_prob, value.squeeze(-1)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate actions for PPO update.
        
        Args:
            obs: Visual observations
            actions: Actions to evaluate
        
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Policy entropy
        """
        action_logits, values = self.forward(obs)
        
        probs = F.softmax(action_logits, dim=-1)
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        action_log_probs = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(-1)
        
        return action_log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """
    Buffer for storing trajectories during PPO rollout phase.
    """
    def __init__(self, buffer_size: int, obs_shape: Tuple, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        
        # Storage
        self.observations = torch.zeros((buffer_size, *obs_shape), dtype=torch.uint8)
        self.actions = torch.zeros(buffer_size, dtype=torch.long)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32)
        self.values = torch.zeros(buffer_size, dtype=torch.float32)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32)
        
        self.ptr = 0
        self.path_start_idx = 0
    
    def add(self, obs, action, reward, value, log_prob, done):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = torch.from_numpy(obs)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float = 0.99, 
                                       gae_lambda: float = 0.95):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE).
        
        Args:
            last_value: Value of the last state (for bootstrapping)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        
        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        advantages = torch.zeros_like(self.rewards)
        last_gae_lambda = 0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_value = last_value
                next_non_terminal = 1.0 - self.dones[t]
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - self.dones[t]
            
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            advantages[t] = last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
        
        returns = advantages + self.values
        
        return returns, advantages
    
    def get(self):
        """Get all data from buffer."""
        return (
            self.observations.to(self.device),
            self.actions.to(self.device),
            self.log_probs.to(self.device),
            self.values.to(self.device),
            self.rewards.to(self.device),
            self.dones.to(self.device)
        )


class PPOTrainer:
    """
    Proximal Policy Optimization trainer.
    """
    def __init__(self, policy: ActorCritic, learning_rate: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate, eps=1e-5)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
    
    def update(self, rollout_buffer: RolloutBuffer, last_value: float,
               num_epochs: int = 4, batch_size: int = 256) -> Dict:
        """
        Perform PPO update using collected rollouts.
        
        Args:
            rollout_buffer: Buffer with collected trajectories
            last_value: Value of last state for bootstrapping
            num_epochs: Number of optimization epochs
            batch_size: Mini-batch size
        
        Returns:
            stats: Dictionary with training statistics
        """
        # Compute returns and advantages
        returns, advantages = rollout_buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get data from buffer
        obs, actions, old_log_probs, old_values, _, _ = rollout_buffer.get()
        returns = returns.to(device)
        advantages = advantages.to(device)
        
        # Training statistics
        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'approx_kl': [],
            'clip_fraction': []
        }
        
        # Multiple epochs of optimization
        for epoch in range(num_epochs):
            # Generate random mini-batches
            indices = torch.randperm(rollout_buffer.buffer_size)
            
            for start_idx in range(0, rollout_buffer.buffer_size, batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # Mini-batch data
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions with current policy
                new_log_probs, new_values, entropy = self.policy.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped for stability)
                value_pred_clipped = old_values[batch_indices] + torch.clamp(
                    new_values - old_values[batch_indices],
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss_unclipped = (new_values - batch_returns) ** 2
                value_loss_clipped = (value_pred_clipped - batch_returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Statistics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - ratio.log()).mean()
                    clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean()
                
                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(-entropy_loss.item())
                stats['total_loss'].append(loss.item())
                stats['approx_kl'].append(approx_kl.item())
                stats['clip_fraction'].append(clip_fraction.item())
        
        # Average statistics
        for key in stats:
            stats[key] = np.mean(stats[key])
        
        return stats


class PPOAgent:
    """
    Main PPO agent for training and evaluation.
    """
    def __init__(self, env, policy: ActorCritic, trainer: PPOTrainer,
                 rollout_steps: int = 2048):
        self.env = env
        self.policy = policy
        self.trainer = trainer
        self.rollout_steps = rollout_steps
        
        # Get observation shape
        obs, _ = env.reset()
        obs_array = np.array(obs)
        self.obs_shape = obs_array.shape
        
        # Create rollout buffer
        self.buffer = RolloutBuffer(rollout_steps, self.obs_shape, device)
        
        # Training statistics
        self.episode_returns = []
        self.episode_lengths = []
    
    def collect_rollouts(self) -> Tuple[RolloutBuffer, float]:
        """
        Collect rollouts for PPO update.
        
        Returns:
            buffer: Filled rollout buffer
            last_value: Value of last state
        """
        obs, _ = self.env.reset()
        obs_array = np.array(obs)
        episode_return = 0
        episode_length = 0
        
        for step in range(self.rollout_steps):
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs_tensor)
                action = action.item()
                log_prob = log_prob.item()
                value = value.item()
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs_array = np.array(next_obs)
            done = terminated or truncated
            
            # Store transition
            self.buffer.add(obs_array, action, reward, value, log_prob, done)
            
            episode_return += reward
            episode_length += 1
            
            # Handle episode end
            if done:
                self.episode_returns.append(episode_return)
                self.episode_lengths.append(episode_length)
                
                obs, _ = self.env.reset()
                obs_array = np.array(obs)
                episode_return = 0
                episode_length = 0
            else:
                obs_array = next_obs_array
        
        # Get value of last state for bootstrapping
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
        with torch.no_grad():
            _, last_value = self.policy.forward(obs_tensor)
            last_value = last_value.item()
        
        return self.buffer, last_value
    
    def train(self, num_updates: int = 100, num_epochs: int = 4, 
              batch_size: int = 256, eval_freq: int = 10) -> Dict:
        """
        Train the PPO agent.
        
        Args:
            num_updates: Number of PPO updates
            num_epochs: Epochs per update
            batch_size: Mini-batch size
            eval_freq: Evaluation frequency
        
        Returns:
            history: Training history
        """
        print(f"\n{'='*80}")
        print(f"Starting PPO Training")
        print(f"{'='*80}")
        print(f"Updates: {num_updates}")
        print(f"Rollout steps: {self.rollout_steps}")
        print(f"Epochs per update: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*80}\n")
        
        history = {
            'update': [],
            'mean_return': [],
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'approx_kl': [],
            'clip_fraction': []
        }
        
        for update in range(num_updates):
            # Collect rollouts
            rollout_buffer, last_value = self.collect_rollouts()
            
            # PPO update
            stats = self.trainer.update(
                rollout_buffer, last_value, num_epochs, batch_size
            )
            
            # Record history
            if len(self.episode_returns) > 0:
                recent_returns = self.episode_returns[-10:] if len(self.episode_returns) >= 10 else self.episode_returns
                mean_return = np.mean(recent_returns)
                
                history['update'].append(update)
                history['mean_return'].append(mean_return)
                for key in ['policy_loss', 'value_loss', 'entropy', 'total_loss', 'approx_kl', 'clip_fraction']:
                    history[key].append(stats[key])
                
                # Print progress
                if (update + 1) % eval_freq == 0:
                    print(f"Update {update + 1:3d}/{num_updates} | "
                          f"Return: {mean_return:7.2f} | "
                          f"Policy Loss: {stats['policy_loss']:.4f} | "
                          f"Value Loss: {stats['value_loss']:.4f} | "
                          f"Entropy: {stats['entropy']:.4f}")
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}\n")
        
        return history
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> Dict:
        """
        Evaluate the trained policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            deterministic: Use deterministic policy
        
        Returns:
            results: Evaluation results
        """
        returns = []
        lengths = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs_array = np.array(obs)
            episode_return = 0
            episode_length = 0
            done = False
            
            while not done:
                obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    action, _, _ = self.policy.get_action(obs_tensor, deterministic)
                    action = action.item()
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                obs_array = np.array(obs)
                done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
            
            returns.append(episode_return)
            lengths.append(episode_length)
        
        results = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'mean_length': np.mean(lengths),
            'returns': returns
        }
        
        return results


class Visualizer:
    """
    Visualization utilities for PPO analysis.
    """
    @staticmethod
    def plot_training_curves(history: Dict, save_path: str = None):
        """Plot PPO training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        metrics = [
            ('mean_return', 'Episode Return', '#2ecc71'),
            ('policy_loss', 'Policy Loss', '#3498db'),
            ('value_loss', 'Value Loss', '#e74c3c'),
            ('entropy', 'Entropy', '#f39c12'),
            ('approx_kl', 'Approximate KL Divergence', '#9b59b6'),
            ('clip_fraction', 'Clip Fraction', '#1abc9c')
        ]
        
        for idx, (key, title, color) in enumerate(metrics):
            if key in history and len(history[key]) > 0:
                axes[idx].plot(history['update'], history[key], 
                             color=color, linewidth=2, alpha=0.8)
                axes[idx].set_xlabel('Update', fontsize=11)
                axes[idx].set_ylabel(title, fontsize=11)
                axes[idx].set_title(title, fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle('PPO Training Curves', fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_comparison(ppo_results: Dict, random_results: Dict, save_path: str = None):
        """Compare PPO vs random policy."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        policies = ['PPO', 'Random']
        means = [ppo_results['mean_return'], random_results['mean_return']]
        stds = [ppo_results['std_return'], random_results['std_return']]
        colors = ['#2ecc71', '#e74c3c']
        
        x_pos = np.arange(len(policies))
        axes[0].bar(x_pos, means, yerr=stds, color=colors, alpha=0.7,
                   capsize=10, edgecolor='black', linewidth=1.5)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(policies, fontsize=12)
        axes[0].set_ylabel('Average Return', fontsize=12)
        axes[0].set_title('Policy Performance Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Distribution plot
        axes[1].hist(ppo_results['returns'], bins=15, alpha=0.6, 
                    label='PPO', color='#2ecc71', edgecolor='black')
        axes[1].hist(random_results['returns'], bins=15, alpha=0.6,
                    label='Random', color='#e74c3c', edgecolor='black')
        axes[1].set_xlabel('Episode Return', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Return Distribution', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Main PPO demonstration.
    """
    print("\n" + "="*80)
    print(" PROXIMAL POLICY OPTIMIZATION (PPO) FOR VISUAL CONTROL ".center(80, "="))
    print("="*80 + "\n")
    
    # ==================== Configuration ====================
    ENV_NAME = "CartPole-v1"
    IMG_SIZE = 84
    STACK_FRAMES = 4
    GRAYSCALE = True
    
    ROLLOUT_STEPS = 2048
    NUM_UPDATES = 100
    NUM_EPOCHS = 4
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4
    
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPSILON = 0.2
    VALUE_COEF = 0.5
    ENTROPY_COEF = 0.01
    
    NUM_EVAL_EPISODES = 20
    
    # ==================== Create Environment ====================
    print("Creating visual environment...")
    env = make_visual_env(ENV_NAME, IMG_SIZE, STACK_FRAMES, GRAYSCALE)
    
    obs, _ = env.reset()
    obs_array = np.array(obs)
    input_channels = obs_array.shape[0] if len(obs_array.shape) == 3 else 1
    num_actions = env.action_space.n
    
    print(f"Environment: {ENV_NAME}")
    print(f"Observation shape: {obs_array.shape}")
    print(f"Input channels: {input_channels}")
    print(f"Number of actions: {num_actions}")
    print(f"Device: {device}\n")
    
    # ==================== Create Policy and Trainer ====================
    print("Initializing policy and trainer...")
    policy = ActorCritic(input_channels, num_actions).to(device)
    
    trainer = PPOTrainer(
        policy,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_epsilon=CLIP_EPSILON,
        value_coef=VALUE_COEF,
        entropy_coef=ENTROPY_COEF
    )
    
    # ==================== Train PPO Agent ====================
    print("\n" + "="*80)
    print(" TRAINING PPO AGENT ".center(80))
    print("="*80)
    
    agent = PPOAgent(env, policy, trainer, ROLLOUT_STEPS)
    history = agent.train(
        num_updates=NUM_UPDATES,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        eval_freq=10
    )
    
    # ==================== Evaluate Trained Policy ====================
    print("\n" + "="*80)
    print(" EVALUATING TRAINED POLICY ".center(80))
    print("="*80 + "\n")
    
    ppo_results = agent.evaluate(num_episodes=NUM_EVAL_EPISODES)
    print(f"PPO Policy - Mean Return: {ppo_results['mean_return']:.2f} ± {ppo_results['std_return']:.2f}")
    print(f"PPO Policy - Mean Length: {ppo_results['mean_length']:.2f}")
    
    # Evaluate random baseline
    print("\nEvaluating random baseline...")
    random_returns = []
    for _ in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset()
        episode_return = 0
        done = False
        while not done:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        random_returns.append(episode_return)
    
    random_results = {
        'mean_return': np.mean(random_returns),
        'std_return': np.std(random_returns),
        'returns': random_returns
    }
    print(f"Random Policy - Mean Return: {random_results['mean_return']:.2f} ± {random_results['std_return']:.2f}")
    
    # ==================== Visualizations ====================
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATIONS ".center(80))
    print("="*80 + "\n")
    
    visualizer = Visualizer()
    
    print("Plotting training curves...")
    visualizer.plot_training_curves(history, save_path='ppo_training_curves.png')
    
    print("Plotting performance comparison...")
    visualizer.plot_comparison(ppo_results, random_results, save_path='ppo_performance_comparison.png')
    
    # ==================== Summary ====================
    print("\n" + "="*80)
    print(" FINAL SUMMARY ".center(80, "="))
    print("="*80 + "\n")
    
    print("Training Configuration:")
    print(f"  - Total updates: {NUM_UPDATES}")
    print(f"  - Rollout steps per update: {ROLLOUT_STEPS}")
    print(f"  - Total environment steps: {NUM_UPDATES * ROLLOUT_STEPS:,}")
    print()
    
    print("Performance Results:")
    print(f"  - PPO Policy:    {ppo_results['mean_return']:6.2f} ± {ppo_results['std_return']:5.2f}")
    print(f"  - Random Policy: {random_results['mean_return']:6.2f} ± {random_results['std_return']:5.2f}")
    print()
    
    improvement = ((ppo_results['mean_return'] - random_results['mean_return']) / 
                   random_results['mean_return'] * 100)
    print(f"Improvement over random: {improvement:.1f}%")
    print()
    
    print("Key PPO Mechanisms Demonstrated:")
    print("  1. CNN encoder for visual feature extraction")
    print("  2. Actor-Critic architecture with shared features")
    print("  3. Clipped surrogate objective for stable updates")
    print("  4. Generalized Advantage Estimation (GAE)")
    print("  5. Multiple epochs of mini-batch optimization")
    print()
    
    print("PPO Advantages:")
    print("  - Sample efficient through multiple epochs")
    print("  - Stable training via trust region clipping")
    print("  - Works well with visual observations")
    print("  - Easy to implement and tune")
    
    print("\n" + "="*80)
    print(" DEMO COMPLETE ".center(80, "="))
    print("="*80 + "\n")
    
    env.close()


if __name__ == "__main__":
    main()