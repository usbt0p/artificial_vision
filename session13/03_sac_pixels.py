"""
Soft Actor-Critic (SAC) for Visual Control
===========================================

This demo illustrates SAC algorithm for learning from pixel observations:
1. Twin Q-networks to reduce overestimation bias
2. Stochastic policy with entropy regularization
3. Automatic entropy temperature tuning
4. Off-policy learning with replay buffer
5. Visual representation learning via CNN encoders

Lecture 13: Visual Policy Learning - From Imitation to Foundation Models
Prof. David Olivieri - UVigo - VIAR25/26

Key Concepts Demonstrated:
--------------------------
- Maximum entropy reinforcement learning
- Twin Q-networks (Double Q-learning)
- Automatic temperature adjustment
- Off-policy learning from pixels
- Soft policy improvement
- Replay buffer for sample efficiency

Note: This implements discrete SAC for compatibility with CartPole.
For continuous control, the policy would use a Gaussian distribution
with reparameterization trick.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import FrameStack, GrayScaleObservation
import matplotlib.pyplot as plt
from collections import deque
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings

warnings.filterwarnings("ignore")

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
            low=0, high=255, shape=(img_size[0], img_size[1], 3), dtype=np.uint8
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
        frame = self.env.render()
        if frame is None:
            frame = np.zeros((*self.img_size, 3), dtype=np.uint8)

        if frame.shape[:2] != self.img_size:
            import cv2

            frame = cv2.resize(frame, self.img_size)

        return frame.astype(np.uint8)


def make_visual_env(
    env_name: str = "CartPole-v1",
    img_size: int = 84,
    stack_frames: int = 4,
    grayscale: bool = True,
):
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
    env = gym.make(env_name, render_mode="rgb_array")
    env = VisualWrapper(env, img_size=(img_size, img_size))

    if grayscale:
        env = GrayScaleObservation(env, keep_dim=False)

    env = FrameStack(env, num_stack=stack_frames)

    return env


class CNNEncoder(nn.Module):
    """
    Convolutional Neural Network encoder for processing visual observations.
    Used by both Q-networks and policy network.
    """

    def __init__(self, input_channels: int = 4, output_dim: int = 256):
        super(CNNEncoder, self).__init__()

        # Convolutional layers (Nature DQN architecture)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        # Calculate output size for 84x84 input
        def conv_output_size(size, kernel_size, stride, padding=0):
            return (size + 2 * padding - kernel_size) // stride + 1

        h = conv_output_size(84, 8, 4)
        h = conv_output_size(h, 4, 2)
        h = conv_output_size(h, 3, 1)

        conv_output_size_flat = h * h * 64

        # Fully connected layer
        self.fc = nn.Linear(conv_output_size_flat, output_dim)

        # Layer normalization for stability
        self.ln = nn.LayerNorm(output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier initialization."""
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN encoder.

        Args:
            x: Visual observations [batch_size, channels, height, width]

        Returns:
            features: Encoded features [batch_size, output_dim]
        """
        # Normalize pixel values
        x = x.float() / 255.0

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected layer with layer norm
        x = self.fc(x)
        x = self.ln(x)
        x = F.relu(x)

        return x


class QNetwork(nn.Module):
    """
    Q-Network for estimating action values.
    Uses CNN encoder followed by MLP.
    """

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super(QNetwork, self).__init__()

        self.encoder = CNNEncoder(input_channels, feature_dim)

        # Q-value head
        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self._init_heads()

    def _init_heads(self):
        """Initialize Q-value head."""
        for layer in self.q_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-values.

        Args:
            obs: Visual observations

        Returns:
            q_values: Q-values for each action [batch_size, num_actions]
        """
        features = self.encoder(obs)
        q_values = self.q_head(features)
        return q_values


class SACPolicy(nn.Module):
    """
    Stochastic policy network for SAC (discrete actions).
    Outputs action probabilities via softmax.
    """

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super(SACPolicy, self).__init__()

        self.encoder = CNNEncoder(input_channels, feature_dim)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self._init_heads()

    def _init_heads(self):
        """Initialize policy head."""
        for layer in self.policy_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action logits.

        Args:
            obs: Visual observations

        Returns:
            logits: Action logits [batch_size, num_actions]
        """
        features = self.encoder(obs)
        logits = self.policy_head(features)
        return logits

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample action from policy.

        Args:
            obs: Visual observation
            deterministic: If True, select argmax action

        Returns:
            action: Selected action
            log_prob: Log probability of action
            probs: Action probabilities
        """
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = torch.multinomial(probs, 1).squeeze(-1)

        # Get log prob of selected action
        action_log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)

        return action, action_log_prob, probs

    def evaluate_actions(self, obs: torch.Tensor):
        """
        Get action distribution for SAC update.

        Args:
            obs: Visual observations

        Returns:
            probs: Action probabilities
            log_probs: Log probabilities
            entropy: Policy entropy
        """
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Entropy of the policy
        entropy = -(probs * log_probs).sum(-1)

        return probs, log_probs, entropy


class ReplayBuffer:
    """
    Replay buffer for off-policy learning.
    Stores transitions and samples mini-batches.
    """

    def __init__(self, capacity: int, obs_shape: Tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Storage
        self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.next_observations = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.float32)

    def add(self, obs, action, reward, next_obs, done):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = torch.from_numpy(obs)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = torch.from_numpy(next_obs)
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        indices = torch.randint(0, self.size, (batch_size,))

        return (
            self.observations[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_observations[indices].to(self.device),
            self.dones[indices].to(self.device),
        )

    def __len__(self):
        return self.size


class SACTrainer:
    """
    Soft Actor-Critic trainer for discrete actions.
    Implements the SAC algorithm with automatic entropy tuning.
    """

    def __init__(
        self,
        policy: SACPolicy,
        q_network1: QNetwork,
        q_network2: QNetwork,
        num_actions: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: Optional[float] = None,
        auto_entropy_tuning: bool = True,
    ):
        self.policy = policy
        self.q_network1 = q_network1
        self.q_network2 = q_network2
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau

        # Create target Q-networks
        self.target_q_network1 = QNetwork(
            q_network1.encoder.conv1.in_channels, num_actions
        ).to(device)
        self.target_q_network2 = QNetwork(
            q_network2.encoder.conv1.in_channels, num_actions
        ).to(device)

        # Initialize target networks
        self.target_q_network1.load_state_dict(q_network1.state_dict())
        self.target_q_network2.load_state_dict(q_network2.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.q1_optimizer = optim.Adam(q_network1.parameters(), lr=learning_rate)
        self.q2_optimizer = optim.Adam(q_network2.parameters(), lr=learning_rate)

        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            # Target entropy is -log(1/|A|) * 0.98 for discrete actions
            if target_entropy is None:
                self.target_entropy = -np.log(1.0 / num_actions) * 0.98
            else:
                self.target_entropy = target_entropy

            # Learnable temperature parameter
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)
        else:
            self.log_alpha = torch.log(torch.tensor(0.2, device=device))

    @property
    def alpha(self):
        """Get current entropy temperature."""
        return self.log_alpha.exp()

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> Dict:
        """
        Perform SAC update.

        Args:
            replay_buffer: Replay buffer to sample from
            batch_size: Mini-batch size

        Returns:
            stats: Dictionary with training statistics
        """
        # Sample from replay buffer
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)

        # ==================== Update Q-networks ====================
        with torch.no_grad():
            # Get next action probabilities and log probs from policy
            next_probs, next_log_probs, _ = self.policy.evaluate_actions(next_obs)

            # Compute target Q-values using both target networks (take minimum)
            target_q1_values = self.target_q_network1(next_obs)
            target_q2_values = self.target_q_network2(next_obs)
            target_q_values = torch.min(target_q1_values, target_q2_values)

            # Soft Q-value: Q - α * log π(a|s)
            # For discrete actions, we take expectation over actions
            soft_q_values = (
                next_probs * (target_q_values - self.alpha * next_log_probs)
            ).sum(dim=-1)

            # Compute target
            q_target = rewards + (1 - dones) * self.gamma * soft_q_values

        # Current Q-values
        q1_values = self.q_network1(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q2_values = self.q_network2(obs).gather(-1, actions.unsqueeze(-1)).squeeze(-1)

        # Q-network losses
        q1_loss = F.mse_loss(q1_values, q_target)
        q2_loss = F.mse_loss(q2_values, q_target)

        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # ==================== Update Policy ====================
        # Get current policy distribution
        probs, log_probs, entropy = self.policy.evaluate_actions(obs)

        # Current Q-values for policy update
        q1_policy = self.q_network1(obs)
        q2_policy = self.q_network2(obs)
        q_policy = torch.min(q1_policy, q2_policy)

        # Policy loss: maximize Q - α * log π
        # For discrete actions: E_a~π[Q(s,a) - α * log π(a|s)]
        policy_loss = (probs * (self.alpha * log_probs - q_policy)).sum(dim=-1).mean()

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ==================== Update Temperature ====================
        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy_tuning:
            # Temperature loss: E[-α * (log π + target_entropy)]
            alpha_loss = (
                -(self.log_alpha * (log_probs + self.target_entropy).detach() * probs)
                .sum(dim=-1)
                .mean()
            )

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # ==================== Update Target Networks ====================
        self._soft_update(self.q_network1, self.target_q_network1)
        self._soft_update(self.q_network2, self.target_q_network2)

        # Return statistics
        stats = {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item() if self.auto_entropy_tuning else 0.0,
            "alpha": self.alpha.item(),
            "entropy": entropy.mean().item(),
            "q1_values": q1_values.mean().item(),
            "q2_values": q2_values.mean().item(),
        }

        return stats

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class SACAgent:
    """
    SAC agent for training and evaluation.
    """

    def __init__(
        self,
        env,
        policy: SACPolicy,
        trainer: SACTrainer,
        buffer_capacity: int = 100000,
        warmup_steps: int = 1000,
    ):
        self.env = env
        self.policy = policy
        self.trainer = trainer
        self.warmup_steps = warmup_steps

        # Get observation shape
        obs, _ = env.reset()
        obs_array = np.array(obs)
        self.obs_shape = obs_array.shape

        # Create replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity, self.obs_shape, device)

        # Training statistics
        self.episode_returns = []
        self.episode_lengths = []
        self.total_steps = 0

    def collect_random_data(self):
        """Collect random data for warmup."""
        print(f"Collecting {self.warmup_steps} random transitions for warmup...")

        obs, _ = self.env.reset()
        obs_array = np.array(obs)

        for step in range(self.warmup_steps):
            action = self.env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs_array = np.array(next_obs)
            done = terminated or truncated

            self.replay_buffer.add(obs_array, action, reward, next_obs_array, done)

            if done:
                obs, _ = self.env.reset()
                obs_array = np.array(obs)
            else:
                obs_array = next_obs_array

            if (step + 1) % 500 == 0:
                print(f"  Collected {step + 1}/{self.warmup_steps} transitions")

        print(f"Warmup complete! Buffer size: {len(self.replay_buffer)}\n")

    def train(
        self,
        num_steps: int = 100000,
        batch_size: int = 256,
        updates_per_step: int = 1,
        eval_freq: int = 5000,
    ) -> Dict:
        """
        Train the SAC agent.

        Args:
            num_steps: Total training steps
            batch_size: Mini-batch size
            updates_per_step: Number of gradient updates per environment step
            eval_freq: Evaluation frequency

        Returns:
            history: Training history
        """
        print(f"\n{'='*80}")
        print(f"Starting SAC Training")
        print(f"{'='*80}")
        print(f"Total steps: {num_steps:,}")
        print(f"Batch size: {batch_size}")
        print(f"Updates per step: {updates_per_step}")
        print(f"{'='*80}\n")

        # Warmup
        if len(self.replay_buffer) < self.warmup_steps:
            self.collect_random_data()

        history = {
            "step": [],
            "mean_return": [],
            "q1_loss": [],
            "q2_loss": [],
            "policy_loss": [],
            "alpha": [],
            "entropy": [],
        }

        obs, _ = self.env.reset()
        obs_array = np.array(obs)
        episode_return = 0
        episode_length = 0

        for step in range(num_steps):
            # Select action
            obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = self.policy.get_action(obs_tensor, deterministic=False)
                action = action.item()

            # Step environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs_array = np.array(next_obs)
            done = terminated or truncated

            # Store transition
            self.replay_buffer.add(obs_array, action, reward, next_obs_array, done)

            episode_return += reward
            episode_length += 1
            self.total_steps += 1

            # Update networks
            if len(self.replay_buffer) >= batch_size:
                for _ in range(updates_per_step):
                    stats = self.trainer.update(self.replay_buffer, batch_size)

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

            # Logging
            if (step + 1) % eval_freq == 0 and len(self.episode_returns) > 0:
                recent_returns = (
                    self.episode_returns[-10:]
                    if len(self.episode_returns) >= 10
                    else self.episode_returns
                )
                mean_return = np.mean(recent_returns)

                history["step"].append(step + 1)
                history["mean_return"].append(mean_return)
                history["q1_loss"].append(stats["q1_loss"])
                history["q2_loss"].append(stats["q2_loss"])
                history["policy_loss"].append(stats["policy_loss"])
                history["alpha"].append(stats["alpha"])
                history["entropy"].append(stats["entropy"])

                print(
                    f"Step {step + 1:6d}/{num_steps} | "
                    f"Return: {mean_return:7.2f} | "
                    f"Q1 Loss: {stats['q1_loss']:.4f} | "
                    f"Policy Loss: {stats['policy_loss']:.4f} | "
                    f"Alpha: {stats['alpha']:.4f} | "
                    f"Entropy: {stats['entropy']:.4f}"
                )

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
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_length": np.mean(lengths),
            "returns": returns,
        }

        return results


class Visualizer:
    """
    Visualization utilities for SAC analysis.
    """

    @staticmethod
    def plot_training_curves(history: Dict, save_path: str = None):
        """Plot SAC training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        metrics = [
            ("mean_return", "Episode Return", "#2ecc71"),
            ("q1_loss", "Q1 Loss", "#3498db"),
            ("q2_loss", "Q2 Loss", "#e74c3c"),
            ("policy_loss", "Policy Loss", "#f39c12"),
            ("alpha", "Temperature (α)", "#9b59b6"),
            ("entropy", "Policy Entropy", "#1abc9c"),
        ]

        for idx, (key, title, color) in enumerate(metrics):
            if key in history and len(history[key]) > 0:
                axes[idx].plot(
                    history["step"], history[key], color=color, linewidth=2, alpha=0.8
                )
                axes[idx].set_xlabel("Training Steps", fontsize=11)
                axes[idx].set_ylabel(title, fontsize=11)
                axes[idx].set_title(title, fontsize=12, fontweight="bold")
                axes[idx].grid(True, alpha=0.3)

        plt.suptitle("SAC Training Curves", fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_comparison(
        sac_results: Dict,
        ppo_results: Dict,
        random_results: Dict,
        save_path: str = None,
    ):
        """Compare SAC vs PPO vs random."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar plot
        policies = ["SAC", "PPO", "Random"]
        means = [
            sac_results["mean_return"],
            ppo_results["mean_return"],
            random_results["mean_return"],
        ]
        stds = [
            sac_results["std_return"],
            ppo_results["std_return"],
            random_results["std_return"],
        ]
        colors = ["#9b59b6", "#2ecc71", "#e74c3c"]

        x_pos = np.arange(len(policies))
        axes[0].bar(
            x_pos,
            means,
            yerr=stds,
            color=colors,
            alpha=0.7,
            capsize=10,
            edgecolor="black",
            linewidth=1.5,
        )
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(policies, fontsize=12)
        axes[0].set_ylabel("Average Return", fontsize=12)
        axes[0].set_title(
            "Policy Performance Comparison", fontsize=14, fontweight="bold"
        )
        axes[0].grid(True, alpha=0.3, axis="y")

        # Distribution plot
        axes[1].hist(
            sac_results["returns"],
            bins=15,
            alpha=0.6,
            label="SAC",
            color="#9b59b6",
            edgecolor="black",
        )
        axes[1].hist(
            ppo_results["returns"],
            bins=15,
            alpha=0.6,
            label="PPO",
            color="#2ecc71",
            edgecolor="black",
        )
        axes[1].hist(
            random_results["returns"],
            bins=15,
            alpha=0.6,
            label="Random",
            color="#e74c3c",
            edgecolor="black",
        )
        axes[1].set_xlabel("Episode Return", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title("Return Distribution", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def main():
    """
    Main SAC demonstration.
    """
    print("\n" + "=" * 80)
    print(" SOFT ACTOR-CRITIC (SAC) FOR VISUAL CONTROL ".center(80, "="))
    print("=" * 80 + "\n")

    # ==================== Configuration ====================
    ENV_NAME = "CartPole-v1"
    IMG_SIZE = 84
    STACK_FRAMES = 4
    GRAYSCALE = True

    BUFFER_CAPACITY = 100000
    WARMUP_STEPS = 1000
    NUM_STEPS = 50000
    BATCH_SIZE = 256
    UPDATES_PER_STEP = 1
    LEARNING_RATE = 3e-4

    GAMMA = 0.99
    TAU = 0.005

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

    # ==================== Create Networks ====================
    print("Initializing networks...")
    policy = SACPolicy(input_channels, num_actions).to(device)
    q_network1 = QNetwork(input_channels, num_actions).to(device)
    q_network2 = QNetwork(input_channels, num_actions).to(device)

    trainer = SACTrainer(
        policy,
        q_network1,
        q_network2,
        num_actions,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        tau=TAU,
        auto_entropy_tuning=True,
    )

    # ==================== Train SAC Agent ====================
    print("\n" + "=" * 80)
    print(" TRAINING SAC AGENT ".center(80))
    print("=" * 80)

    agent = SACAgent(env, policy, trainer, BUFFER_CAPACITY, WARMUP_STEPS)
    history = agent.train(
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        updates_per_step=UPDATES_PER_STEP,
        eval_freq=5000,
    )

    # ==================== Evaluate ====================
    print("\n" + "=" * 80)
    print(" EVALUATING POLICIES ".center(80))
    print("=" * 80 + "\n")

    # SAC
    sac_results = agent.evaluate(num_episodes=NUM_EVAL_EPISODES)
    print(
        f"SAC Policy - Mean Return: {sac_results['mean_return']:.2f} ± {sac_results['std_return']:.2f}"
    )

    # Random baseline
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
        "mean_return": np.mean(random_returns),
        "std_return": np.std(random_returns),
        "returns": random_returns,
    }
    print(
        f"Random Policy - Mean Return: {random_results['mean_return']:.2f} ± {random_results['std_return']:.2f}"
    )

    # Mock PPO results for comparison (or load from previous demo)
    ppo_results = {
        "mean_return": 200.0,  # Placeholder
        "std_return": 50.0,
        "returns": [200.0] * NUM_EVAL_EPISODES,
    }

    # ==================== Visualizations ====================
    print("\n" + "=" * 80)
    print(" GENERATING VISUALIZATIONS ".center(80))
    print("=" * 80 + "\n")

    visualizer = Visualizer()

    print("Plotting training curves...")
    visualizer.plot_training_curves(history, save_path="sac_training_curves.png")

    print("Plotting performance comparison...")
    visualizer.plot_comparison(
        sac_results,
        ppo_results,
        random_results,
        save_path="sac_performance_comparison.png",
    )

    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print(" FINAL SUMMARY ".center(80, "="))
    print("=" * 80 + "\n")

    print("Training Configuration:")
    print(f"  - Total steps: {NUM_STEPS:,}")
    print(f"  - Buffer capacity: {BUFFER_CAPACITY:,}")
    print(f"  - Warmup steps: {WARMUP_STEPS:,}")
    print()

    print("Performance Results:")
    print(
        f"  - SAC Policy:    {sac_results['mean_return']:6.2f} ± {sac_results['std_return']:5.2f}"
    )
    print(
        f"  - Random Policy: {random_results['mean_return']:6.2f} ± {random_results['std_return']:5.2f}"
    )
    print()

    improvement = (
        (sac_results["mean_return"] - random_results["mean_return"])
        / random_results["mean_return"]
        * 100
    )
    print(f"Improvement over random: {improvement:.1f}%")
    print()

    print("Key SAC Mechanisms Demonstrated:")
    print("  1. Twin Q-networks to reduce overestimation bias")
    print("  2. Stochastic policy with maximum entropy objective")
    print("  3. Automatic entropy temperature tuning")
    print("  4. Off-policy learning with replay buffer")
    print("  5. Soft policy improvement for exploration")
    print()

    print("SAC vs PPO Comparison:")
    print("  - SAC: Off-policy, sample efficient, automatic exploration")
    print("  - PPO: On-policy, stable, better for some tasks")
    print("  - SAC: Better for continuous control (adapted here for discrete)")
    print("  - PPO: Often simpler to implement and tune")

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE ".center(80, "="))
    print("=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    main()
