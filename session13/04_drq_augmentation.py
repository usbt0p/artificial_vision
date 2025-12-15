"""
DrQ: Data-Regularized Q-Learning with Image Augmentation
=========================================================

This demo illustrates DrQ algorithm for sample-efficient visual RL:
1. Random crop and translate augmentations
2. Q-learning with augmented observations
3. Regularization through data augmentation
4. Comparison: with vs. without augmentation
5. Sample efficiency analysis

Lecture 13: Visual Policy Learning - From Imitation to Foundation Models
Prof. David Olivieri - UVigo - VIAR25/26

Key Concepts Demonstrated:
--------------------------
- Data augmentation as implicit regularization
- Random crop for visual invariance
- Q-learning with augmented data
- Sample efficiency improvements
- Simple yet powerful approach to visual RL
- Ablation studies on augmentation strategies

Reference: Kostrikov et al., "Image Augmentation Is All You Need" (ICLR 2021)
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
    """
    env = gym.make(env_name, render_mode="rgb_array")
    env = VisualWrapper(env, img_size=(img_size, img_size))

    if grayscale:
        env = GrayScaleObservation(env, keep_dim=False)

    env = FrameStack(env, num_stack=stack_frames)

    return env


class ImageAugmentation:
    """
    Image augmentation techniques for visual RL.
    Implements random crop, shift, and other transformations.
    """

    def __init__(self, img_size: int = 84, pad: int = 4):
        self.img_size = img_size
        self.pad = pad
        self.padded_size = img_size + 2 * pad

    def random_crop(self, images: torch.Tensor) -> torch.Tensor:
        """
        Random crop augmentation (the key DrQ augmentation).

        Args:
            images: Batch of images [batch_size, channels, height, width]

        Returns:
            cropped: Randomly cropped images
        """
        batch_size, channels, h, w = images.shape

        # Pad images
        padded = F.pad(
            images, (self.pad, self.pad, self.pad, self.pad), mode="replicate"
        )

        # Random crop coordinates
        crop_max = self.padded_size - self.img_size
        w1 = torch.randint(0, crop_max + 1, (batch_size,))
        h1 = torch.randint(0, crop_max + 1, (batch_size,))

        # Crop each image in the batch
        cropped = torch.zeros_like(images)
        for i in range(batch_size):
            cropped[i] = padded[
                i, :, h1[i] : h1[i] + self.img_size, w1[i] : w1[i] + self.img_size
            ]

        return cropped

    def random_shift(self, images: torch.Tensor, max_shift: int = 4) -> torch.Tensor:
        """
        Random translation/shift augmentation.

        Args:
            images: Batch of images [batch_size, channels, height, width]
            max_shift: Maximum shift in pixels

        Returns:
            shifted: Randomly shifted images
        """
        batch_size, channels, h, w = images.shape

        # Pad images
        padded = F.pad(
            images, (max_shift, max_shift, max_shift, max_shift), mode="replicate"
        )

        # Random shift amounts
        shift_h = torch.randint(-max_shift, max_shift + 1, (batch_size,))
        shift_w = torch.randint(-max_shift, max_shift + 1, (batch_size,))

        # Shift each image
        shifted = torch.zeros_like(images)
        for i in range(batch_size):
            h_start = max_shift + shift_h[i]
            w_start = max_shift + shift_w[i]
            shifted[i] = padded[i, :, h_start : h_start + h, w_start : w_start + w]

        return shifted

    def random_intensity(
        self, images: torch.Tensor, scale: float = 0.05
    ) -> torch.Tensor:
        """
        Random intensity/brightness augmentation.

        Args:
            images: Batch of images [batch_size, channels, height, width]
            scale: Scale of intensity variation

        Returns:
            augmented: Images with random intensity
        """
        # Random intensity multiplier per image
        batch_size = images.shape[0]
        intensity = 1.0 + scale * (
            2.0 * torch.rand(batch_size, 1, 1, 1, device=images.device) - 1.0
        )

        augmented = images * intensity
        augmented = torch.clamp(augmented, 0, 255)

        return augmented

    def no_augmentation(self, images: torch.Tensor) -> torch.Tensor:
        """No augmentation (identity function for comparison)."""
        return images

    def apply(self, images: torch.Tensor, aug_type: str = "crop") -> torch.Tensor:
        """
        Apply specified augmentation.

        Args:
            images: Batch of images
            aug_type: Type of augmentation ('crop', 'shift', 'intensity', 'none')

        Returns:
            augmented: Augmented images
        """
        if aug_type == "crop":
            return self.random_crop(images)
        elif aug_type == "shift":
            return self.random_shift(images)
        elif aug_type == "intensity":
            return self.random_intensity(images)
        elif aug_type == "none":
            return self.no_augmentation(images)
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")


class CNNEncoder(nn.Module):
    """
    Convolutional Neural Network encoder for processing visual observations.
    """

    def __init__(self, input_channels: int = 4, output_dim: int = 256):
        super(CNNEncoder, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        # Calculate output size
        def conv_output_size(size, kernel_size, stride, padding=0):
            return (size + 2 * padding - kernel_size) // stride + 1

        h = conv_output_size(84, 8, 4)
        h = conv_output_size(h, 4, 2)
        h = conv_output_size(h, 3, 1)

        conv_output_size_flat = h * h * 64

        # Fully connected layer
        self.fc = nn.Linear(conv_output_size_flat, output_dim)

        # Layer normalization
        self.ln = nn.LayerNorm(output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier initialization."""
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN encoder."""
        # Normalize pixel values
        x = x.float() / 255.0

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.reshape(x.size(0), -1)

        # Fully connected with layer norm
        x = self.fc(x)
        x = self.ln(x)
        x = F.relu(x)

        return x


class DrQNetwork(nn.Module):
    """
    Q-Network for DrQ algorithm.
    Processes augmented visual observations.
    """

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super(DrQNetwork, self).__init__()

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
            q_values: Q-values for each action
        """
        features = self.encoder(obs)
        q_values = self.q_head(features)
        return q_values


class ReplayBuffer:
    """
    Replay buffer for off-policy learning.
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


class DrQTrainer:
    """
    DrQ trainer with data augmentation.
    """

    def __init__(
        self,
        q_network: DrQNetwork,
        augmentation: ImageAugmentation,
        num_actions: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        num_augmentations: int = 2,
        aug_type: str = "crop",
        use_augmentation: bool = True,
    ):
        self.q_network = q_network
        self.augmentation = augmentation
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.num_augmentations = num_augmentations
        self.aug_type = aug_type
        self.use_augmentation = use_augmentation

        # Create target network
        self.target_q_network = DrQNetwork(
            q_network.encoder.conv1.in_channels, num_actions
        ).to(device)
        self.target_q_network.load_state_dict(q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

        # Epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, obs: torch.Tensor, eval_mode: bool = False) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            obs: Visual observation
            eval_mode: If True, use greedy policy

        Returns:
            action: Selected action
        """
        if eval_mode or np.random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(obs)
                action = q_values.argmax(dim=-1).item()
        else:
            action = np.random.randint(0, self.num_actions)

        return action

    def update(self, replay_buffer: ReplayBuffer, batch_size: int = 256) -> Dict:
        """
        Perform DrQ update with data augmentation.

        Args:
            replay_buffer: Replay buffer to sample from
            batch_size: Mini-batch size

        Returns:
            stats: Training statistics
        """
        # Sample from replay buffer
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)

        total_loss = 0.0
        q_values_list = []

        # Multiple augmented views (key DrQ technique)
        for k in range(self.num_augmentations):
            # Apply augmentation
            if self.use_augmentation:
                aug_obs = self.augmentation.apply(obs, self.aug_type)
                aug_next_obs = self.augmentation.apply(next_obs, self.aug_type)
            else:
                aug_obs = obs
                aug_next_obs = next_obs

            # Compute Q-values
            q_values = self.q_network(aug_obs)
            q_values_actions = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)

            # Compute target Q-values
            with torch.no_grad():
                next_q_values = self.target_q_network(aug_next_obs)
                max_next_q_values = next_q_values.max(dim=-1)[0]
                q_target = rewards + (1 - dones) * self.gamma * max_next_q_values

            # Loss for this augmentation
            loss_k = F.mse_loss(q_values_actions, q_target)
            total_loss += loss_k

            q_values_list.append(q_values_actions.mean().item())

        # Average loss over augmentations
        loss = total_loss / self.num_augmentations

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self._soft_update()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        stats = {
            "loss": loss.item(),
            "q_values": np.mean(q_values_list),
            "epsilon": self.epsilon,
        }

        return stats

    def _soft_update(self):
        """Soft update of target network."""
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class DrQAgent:
    """
    DrQ agent for training and evaluation.
    """

    def __init__(
        self,
        env,
        q_network: DrQNetwork,
        trainer: DrQTrainer,
        buffer_capacity: int = 100000,
        warmup_steps: int = 1000,
    ):
        self.env = env
        self.q_network = q_network
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
        num_steps: int = 50000,
        batch_size: int = 256,
        update_freq: int = 1,
        eval_freq: int = 5000,
    ) -> Dict:
        """
        Train the DrQ agent.

        Args:
            num_steps: Total training steps
            batch_size: Mini-batch size
            update_freq: Update frequency (steps between updates)
            eval_freq: Evaluation frequency

        Returns:
            history: Training history
        """
        print(f"\n{'='*80}")
        print(f"Starting DrQ Training")
        print(f"{'='*80}")
        print(f"Total steps: {num_steps:,}")
        print(
            f"Augmentation: {self.trainer.aug_type if self.trainer.use_augmentation else 'None'}"
        )
        print(f"Num augmentations: {self.trainer.num_augmentations}")
        print(f"{'='*80}\n")

        # Warmup
        if len(self.replay_buffer) < self.warmup_steps:
            self.collect_random_data()

        history = {
            "step": [],
            "mean_return": [],
            "loss": [],
            "q_values": [],
            "epsilon": [],
        }

        obs, _ = self.env.reset()
        obs_array = np.array(obs)
        episode_return = 0
        episode_length = 0

        for step in range(num_steps):
            # Select action
            obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
            action = self.trainer.select_action(obs_tensor, eval_mode=False)

            # Step environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs_array = np.array(next_obs)
            done = terminated or truncated

            # Store transition
            self.replay_buffer.add(obs_array, action, reward, next_obs_array, done)

            episode_return += reward
            episode_length += 1

            # Update network
            if step % update_freq == 0 and len(self.replay_buffer) >= batch_size:
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
                history["loss"].append(stats["loss"])
                history["q_values"].append(stats["q_values"])
                history["epsilon"].append(stats["epsilon"])

                print(
                    f"Step {step + 1:6d}/{num_steps} | "
                    f"Return: {mean_return:7.2f} | "
                    f"Loss: {stats['loss']:.4f} | "
                    f"Q-values: {stats['q_values']:.2f} | "
                    f"Epsilon: {stats['epsilon']:.4f}"
                )

        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}\n")

        return history

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate the trained policy.

        Args:
            num_episodes: Number of evaluation episodes

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
                action = self.trainer.select_action(obs_tensor, eval_mode=True)

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
    Visualization utilities for DrQ analysis.
    """

    @staticmethod
    def plot_training_curves(histories: Dict[str, Dict], save_path: str = None):
        """Plot training curves for different configurations."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

        # Episode return
        for idx, (name, history) in enumerate(histories.items()):
            if "mean_return" in history and len(history["mean_return"]) > 0:
                axes[0].plot(
                    history["step"],
                    history["mean_return"],
                    label=name,
                    color=colors[idx % len(colors)],
                    linewidth=2,
                )
        axes[0].set_xlabel("Training Steps", fontsize=11)
        axes[0].set_ylabel("Episode Return", fontsize=11)
        axes[0].set_title(
            "Episode Return vs Training Steps", fontsize=12, fontweight="bold"
        )
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Loss
        for idx, (name, history) in enumerate(histories.items()):
            if "loss" in history and len(history["loss"]) > 0:
                axes[1].plot(
                    history["step"],
                    history["loss"],
                    label=name,
                    color=colors[idx % len(colors)],
                    linewidth=2,
                )
        axes[1].set_xlabel("Training Steps", fontsize=11)
        axes[1].set_ylabel("Loss", fontsize=11)
        axes[1].set_title("Training Loss", fontsize=12, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Q-values
        for idx, (name, history) in enumerate(histories.items()):
            if "q_values" in history and len(history["q_values"]) > 0:
                axes[2].plot(
                    history["step"],
                    history["q_values"],
                    label=name,
                    color=colors[idx % len(colors)],
                    linewidth=2,
                )
        axes[2].set_xlabel("Training Steps", fontsize=11)
        axes[2].set_ylabel("Average Q-values", fontsize=11)
        axes[2].set_title("Q-value Estimates", fontsize=12, fontweight="bold")
        axes[2].legend(fontsize=10)
        axes[2].grid(True, alpha=0.3)

        # Epsilon
        for idx, (name, history) in enumerate(histories.items()):
            if "epsilon" in history and len(history["epsilon"]) > 0:
                axes[3].plot(
                    history["step"],
                    history["epsilon"],
                    label=name,
                    color=colors[idx % len(colors)],
                    linewidth=2,
                )
        axes[3].set_xlabel("Training Steps", fontsize=11)
        axes[3].set_ylabel("Epsilon", fontsize=11)
        axes[3].set_title("Exploration Rate (ε)", fontsize=12, fontweight="bold")
        axes[3].legend(fontsize=10)
        axes[3].grid(True, alpha=0.3)

        plt.suptitle("DrQ Training Analysis", fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_performance_comparison(
        results_dict: Dict[str, Dict], save_path: str = None
    ):
        """Compare performance of different augmentation strategies."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        names = list(results_dict.keys())
        means = [results_dict[name]["mean_return"] for name in names]
        stds = [results_dict[name]["std_return"] for name in names]
        colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]

        # Bar plot
        x_pos = np.arange(len(names))
        axes[0].bar(
            x_pos,
            means,
            yerr=stds,
            color=colors[: len(names)],
            alpha=0.7,
            capsize=10,
            edgecolor="black",
            linewidth=1.5,
        )
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(names, fontsize=10, rotation=15, ha="right")
        axes[0].set_ylabel("Average Return", fontsize=12)
        axes[0].set_title("Performance Comparison", fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Distribution plot
        for idx, name in enumerate(names):
            returns = results_dict[name]["returns"]
            axes[1].hist(
                returns,
                bins=15,
                alpha=0.5,
                label=name,
                color=colors[idx],
                edgecolor="black",
            )
        axes[1].set_xlabel("Episode Return", fontsize=12)
        axes[1].set_ylabel("Frequency", fontsize=12)
        axes[1].set_title("Return Distribution", fontsize=14, fontweight="bold")
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def visualize_augmentations(
        env, augmentation: ImageAugmentation, save_path: str = None
    ):
        """Visualize different augmentation strategies."""
        obs, _ = env.reset()
        obs_array = np.array(obs)
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)

        # Generate augmented versions
        aug_types = ["none", "crop", "shift", "intensity"]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        for idx, aug_type in enumerate(aug_types):
            # First row: single augmentation
            aug_obs = augmentation.apply(obs_tensor, aug_type)

            # Convert to displayable format
            img = aug_obs[0].cpu().numpy()
            if img.shape[0] == 1:  # Grayscale
                img = img[0]
            else:  # RGB
                img = np.transpose(img, (1, 2, 0))

            axes[0, idx].imshow(img, cmap="gray" if img.ndim == 2 else None)
            axes[0, idx].set_title(
                f"{aug_type.capitalize()}", fontsize=11, fontweight="bold"
            )
            axes[0, idx].axis("off")

            # Second row: another random sample
            aug_obs2 = augmentation.apply(obs_tensor, aug_type)
            img2 = aug_obs2[0].cpu().numpy()
            if img2.shape[0] == 1:
                img2 = img2[0]
            else:
                img2 = np.transpose(img2, (1, 2, 0))

            axes[1, idx].imshow(img2, cmap="gray" if img2.ndim == 2 else None)
            axes[1, idx].set_title(
                f"{aug_type.capitalize()} (2)", fontsize=11, fontweight="bold"
            )
            axes[1, idx].axis("off")

        plt.suptitle("Image Augmentation Examples", fontsize=15, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def main():
    """
    Main DrQ demonstration with ablation studies.
    """
    print("\n" + "=" * 80)
    print(" DrQ: DATA-REGULARIZED Q-LEARNING WITH AUGMENTATION ".center(80, "="))
    print("=" * 80 + "\n")

    # ==================== Configuration ====================
    ENV_NAME = "CartPole-v1"
    IMG_SIZE = 84
    STACK_FRAMES = 4
    GRAYSCALE = True

    BUFFER_CAPACITY = 50000
    WARMUP_STEPS = 1000
    NUM_STEPS = 30000
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4

    GAMMA = 0.99
    TAU = 0.005
    NUM_AUGMENTATIONS = 2

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

    # Create augmentation module
    augmentation = ImageAugmentation(img_size=IMG_SIZE, pad=4)

    # ==================== Visualize Augmentations ====================
    print("\n" + "=" * 80)
    print(" VISUALIZING AUGMENTATION STRATEGIES ".center(80))
    print("=" * 80 + "\n")

    visualizer = Visualizer()
    visualizer.visualize_augmentations(
        env, augmentation, save_path="drq_augmentations.png"
    )

    # ==================== Train with Different Strategies ====================
    print("\n" + "=" * 80)
    print(" TRAINING WITH DIFFERENT AUGMENTATION STRATEGIES ".center(80))
    print("=" * 80)

    configs = [
        ("DrQ (Crop)", True, "crop"),
        ("No Augmentation", False, "none"),
    ]

    histories = {}
    results_dict = {}

    for name, use_aug, aug_type in configs:
        print(f"\n{'='*80}")
        print(f" Training: {name} ".center(80))
        print(f"{'='*80}\n")

        # Create fresh network and trainer
        q_network = DrQNetwork(input_channels, num_actions).to(device)
        trainer = DrQTrainer(
            q_network,
            augmentation,
            num_actions,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            tau=TAU,
            num_augmentations=NUM_AUGMENTATIONS,
            aug_type=aug_type,
            use_augmentation=use_aug,
        )

        # Create agent
        agent = DrQAgent(env, q_network, trainer, BUFFER_CAPACITY, WARMUP_STEPS)

        # Train
        history = agent.train(
            num_steps=NUM_STEPS, batch_size=BATCH_SIZE, update_freq=1, eval_freq=5000
        )
        histories[name] = history

        # Evaluate
        print(f"\nEvaluating {name}...")
        results = agent.evaluate(num_episodes=NUM_EVAL_EPISODES)
        results_dict[name] = results
        print(
            f"{name} - Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}"
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

    results_dict["Random"] = {
        "mean_return": np.mean(random_returns),
        "std_return": np.std(random_returns),
        "returns": random_returns,
    }
    print(
        f"Random - Mean Return: {results_dict['Random']['mean_return']:.2f} ± {results_dict['Random']['std_return']:.2f}"
    )

    # ==================== Visualizations ====================
    print("\n" + "=" * 80)
    print(" GENERATING VISUALIZATIONS ".center(80))
    print("=" * 80 + "\n")

    print("Plotting training curves...")
    visualizer.plot_training_curves(histories, save_path="drq_training_curves.png")

    print("Plotting performance comparison...")
    visualizer.plot_performance_comparison(
        results_dict, save_path="drq_performance_comparison.png"
    )

    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print(" FINAL SUMMARY ".center(80, "="))
    print("=" * 80 + "\n")

    print("Performance Results:")
    for name in ["DrQ (Crop)", "No Augmentation", "Random"]:
        if name in results_dict:
            print(
                f"  - {name:20s}: {results_dict[name]['mean_return']:6.2f} ± {results_dict[name]['std_return']:5.2f}"
            )
    print()

    # Calculate improvement
    if "DrQ (Crop)" in results_dict and "No Augmentation" in results_dict:
        improvement = (
            (
                results_dict["DrQ (Crop)"]["mean_return"]
                - results_dict["No Augmentation"]["mean_return"]
            )
            / results_dict["No Augmentation"]["mean_return"]
            * 100
        )
        print(f"DrQ improvement over no augmentation: {improvement:.1f}%")
        print()

    print("Key DrQ Contributions:")
    print("  1. Data augmentation as implicit regularization")
    print("  2. Random crop for spatial invariance")
    print("  3. Multiple augmented views per sample")
    print("  4. Significant sample efficiency gains")
    print("  5. Simple yet highly effective approach")
    print()

    print("DrQ Advantages:")
    print("  - Minimal code changes to standard Q-learning")
    print("  - No additional learnable parameters")
    print("  - Works with any Q-learning variant")
    print("  - Improves sample efficiency dramatically")
    print("  - Easy to implement and tune")
    print()

    print("Practical Insights:")
    print("  - Random crop is the most effective augmentation")
    print("  - 2-4 augmented views per sample works well")
    print("  - Augmentation acts as strong regularizer")
    print("  - Essential for sample-efficient visual RL")

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE ".center(80, "="))
    print("=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    main()
