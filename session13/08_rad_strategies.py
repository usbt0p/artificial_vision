"""
RAD: Reinforcement Learning with Augmented Data - Strategy Comparison
======================================================================

This demo illustrates RAD with different augmentation strategies:
1. Comprehensive augmentation library (crop, shift, cutout, color jitter, etc.)
2. Ablation study comparing augmentation strategies
3. Performance analysis across different augmentations
4. Sample efficiency comparison
5. Best practices for visual RL

Lecture 13: Visual Policy Learning - From Imitation to Foundation Models
Prof. David Olivieri - UVigo - VIAR25/26

Key Concepts Demonstrated:
--------------------------
- Data augmentation for visual RL
- Comparison of augmentation strategies
- Ablation studies and best practices
- Sample efficiency improvements
- Task-specific augmentation selection

Reference: Laskin et al., "Reinforcement Learning with Augmented Data" (NeurIPS 2020)
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
    """Wrapper to convert any Gym environment to visual observations."""

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
    """Create a visual control environment with preprocessing."""
    env = gym.make(env_name, render_mode="rgb_array")
    env = VisualWrapper(env, img_size=(img_size, img_size))

    if grayscale:
        env = GrayScaleObservation(env, keep_dim=False)

    env = FrameStack(env, num_stack=stack_frames)

    return env


class RADAugmentations:
    """
    Comprehensive augmentation library for RAD.
    Implements various augmentation strategies for visual RL.
    """

    def __init__(self, img_size: int = 84):
        self.img_size = img_size

    def random_crop(self, images: torch.Tensor, pad: int = 4) -> torch.Tensor:
        """
        Random crop augmentation (most effective for visual RL).

        Args:
            images: Batch of images [batch_size, channels, height, width]
            pad: Padding amount

        Returns:
            cropped: Randomly cropped images
        """
        batch_size, channels, h, w = images.shape

        # Pad images
        padded = F.pad(images, (pad, pad, pad, pad), mode="replicate")

        # Random crop
        crop_max = h + 2 * pad - self.img_size
        w1 = torch.randint(0, crop_max + 1, (batch_size,))
        h1 = torch.randint(0, crop_max + 1, (batch_size,))

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
            images: Batch of images
            max_shift: Maximum shift in pixels

        Returns:
            shifted: Randomly shifted images
        """
        batch_size, channels, h, w = images.shape

        # Pad images
        padded = F.pad(
            images, (max_shift, max_shift, max_shift, max_shift), mode="replicate"
        )

        # Random shifts
        shift_h = torch.randint(-max_shift, max_shift + 1, (batch_size,))
        shift_w = torch.randint(-max_shift, max_shift + 1, (batch_size,))

        shifted = torch.zeros_like(images)
        for i in range(batch_size):
            h_start = max_shift + shift_h[i]
            w_start = max_shift + shift_w[i]
            shifted[i] = padded[i, :, h_start : h_start + h, w_start : w_start + w]

        return shifted

    def cutout(self, images: torch.Tensor, mask_size: int = 20) -> torch.Tensor:
        """
        Cutout augmentation (random rectangular occlusion).

        Args:
            images: Batch of images
            mask_size: Size of cutout square

        Returns:
            occluded: Images with random cutout
        """
        batch_size, channels, h, w = images.shape

        occluded = images.clone()

        for i in range(batch_size):
            # Random position for cutout
            h1 = torch.randint(0, h - mask_size + 1, (1,)).item()
            w1 = torch.randint(0, w - mask_size + 1, (1,)).item()

            # Apply cutout (set to zero)
            occluded[i, :, h1 : h1 + mask_size, w1 : w1 + mask_size] = 0

        return occluded

    def cutout_color(self, images: torch.Tensor, mask_size: int = 20) -> torch.Tensor:
        """
        Cutout with random color fill instead of zeros.

        Args:
            images: Batch of images
            mask_size: Size of cutout square

        Returns:
            occluded: Images with colored cutout
        """
        batch_size, channels, h, w = images.shape

        occluded = images.clone()

        for i in range(batch_size):
            # Random position
            h1 = torch.randint(0, h - mask_size + 1, (1,)).item()
            w1 = torch.randint(0, w - mask_size + 1, (1,)).item()

            # Random color
            color = torch.rand(channels, 1, 1, device=images.device) * 255

            # Apply colored cutout
            occluded[i, :, h1 : h1 + mask_size, w1 : w1 + mask_size] = color

        return occluded

    def random_flip(self, images: torch.Tensor, prob: float = 0.5) -> torch.Tensor:
        """
        Random horizontal flip.

        Args:
            images: Batch of images
            prob: Probability of flipping

        Returns:
            flipped: Randomly flipped images
        """
        batch_size = images.shape[0]

        flipped = images.clone()

        for i in range(batch_size):
            if torch.rand(1).item() < prob:
                flipped[i] = torch.flip(images[i], dims=[-1])

        return flipped

    def random_rotation(self, images: torch.Tensor, max_angle: int = 5) -> torch.Tensor:
        """
        Random rotation (small angles).

        Args:
            images: Batch of images
            max_angle: Maximum rotation angle in degrees

        Returns:
            rotated: Randomly rotated images
        """
        batch_size = images.shape[0]

        rotated = images.clone()

        for i in range(batch_size):
            angle = (torch.rand(1).item() * 2 - 1) * max_angle
            # Note: Actual rotation would require interpolation
            # For simplicity, we skip implementation here
            # In practice, use torchvision.transforms.functional.rotate
            rotated[i] = images[i]  # Placeholder

        return rotated

    def color_jitter(
        self, images: torch.Tensor, brightness: float = 0.1, contrast: float = 0.1
    ) -> torch.Tensor:
        """
        Random color jitter (brightness and contrast).

        Args:
            images: Batch of images
            brightness: Brightness variation range
            contrast: Contrast variation range

        Returns:
            jittered: Color-jittered images
        """
        batch_size = images.shape[0]

        jittered = images.clone().float()

        for i in range(batch_size):
            # Random brightness
            brightness_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * brightness
            jittered[i] = jittered[i] * brightness_factor

            # Random contrast
            contrast_factor = 1.0 + (torch.rand(1).item() * 2 - 1) * contrast
            mean = jittered[i].mean()
            jittered[i] = (jittered[i] - mean) * contrast_factor + mean

            # Clamp to valid range
            jittered[i] = torch.clamp(jittered[i], 0, 255)

        return jittered

    def random_grayscale(self, images: torch.Tensor, prob: float = 0.3) -> torch.Tensor:
        """
        Randomly convert to grayscale.

        Args:
            images: Batch of images
            prob: Probability of grayscale conversion

        Returns:
            converted: Randomly grayscaled images
        """
        # Only works if images have 3 channels (RGB)
        if images.shape[1] != 3:
            return images

        batch_size = images.shape[0]
        converted = images.clone()

        for i in range(batch_size):
            if torch.rand(1).item() < prob:
                # Convert to grayscale (average of RGB)
                gray = images[i].mean(dim=0, keepdim=True)
                converted[i] = gray.repeat(3, 1, 1)

        return converted

    def no_augmentation(self, images: torch.Tensor) -> torch.Tensor:
        """Identity augmentation (no transformation)."""
        return images

    def apply(self, images: torch.Tensor, aug_type: str) -> torch.Tensor:
        """
        Apply specified augmentation.

        Args:
            images: Batch of images
            aug_type: Type of augmentation

        Returns:
            augmented: Augmented images
        """
        aug_map = {
            "crop": lambda x: self.random_crop(x, pad=4),
            "shift": lambda x: self.random_shift(x, max_shift=4),
            "cutout": lambda x: self.cutout(x, mask_size=20),
            "cutout_color": lambda x: self.cutout_color(x, mask_size=20),
            "flip": lambda x: self.random_flip(x, prob=0.5),
            "rotate": lambda x: self.random_rotation(x, max_angle=5),
            "color_jitter": lambda x: self.color_jitter(
                x, brightness=0.2, contrast=0.2
            ),
            "grayscale": lambda x: self.random_grayscale(x, prob=0.3),
            "none": lambda x: self.no_augmentation(x),
        }

        if aug_type not in aug_map:
            raise ValueError(f"Unknown augmentation: {aug_type}")

        return aug_map[aug_type](images)


class CNNEncoder(nn.Module):
    """CNN encoder for processing visual observations."""

    def __init__(self, input_channels: int = 4, output_dim: int = 256):
        super(CNNEncoder, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        def conv_output_size(size, kernel_size, stride, padding=0):
            return (size + 2 * padding - kernel_size) // stride + 1

        h = conv_output_size(84, 8, 4)
        h = conv_output_size(h, 4, 2)
        h = conv_output_size(h, 3, 1)

        conv_output_size_flat = h * h * 64

        self.fc = nn.Linear(conv_output_size_flat, output_dim)
        self.ln = nn.LayerNorm(output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.ln(x)
        x = F.relu(x)
        return x


class RADQNetwork(nn.Module):
    """Q-Network with RAD augmentation."""

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super(RADQNetwork, self).__init__()

        self.encoder = CNNEncoder(input_channels, feature_dim)

        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self._init_heads()

    def _init_heads(self):
        for layer in self.q_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        q_values = self.q_head(features)
        return q_values


class ReplayBuffer:
    """Replay buffer for off-policy learning."""

    def __init__(self, capacity: int, obs_shape: Tuple, device: torch.device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.observations = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.next_observations = torch.zeros((capacity, *obs_shape), dtype=torch.uint8)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.dones = torch.zeros(capacity, dtype=torch.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.observations[self.ptr] = torch.from_numpy(obs)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = torch.from_numpy(next_obs)
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
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


class RADAgent:
    """
    RL agent with RAD augmentation.
    """

    def __init__(
        self,
        env,
        input_channels: int,
        num_actions: int,
        augmentation: RADAugmentations,
        aug_type: str = "crop",
        use_augmentation: bool = True,
        learning_rate: float = 1e-4,
    ):
        self.env = env
        self.num_actions = num_actions
        self.augmentation = augmentation
        self.aug_type = aug_type
        self.use_augmentation = use_augmentation

        # Q-network and target
        self.q_network = RADQNetwork(input_channels, num_actions).to(device)
        self.target_q_network = RADQNetwork(input_channels, num_actions).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Target network update
        self.tau = 0.005

        # Get observation shape
        obs, _ = env.reset()
        obs_array = np.array(obs)
        self.obs_shape = obs_array.shape

        # Replay buffer
        self.buffer = ReplayBuffer(50000, self.obs_shape, device)

        # Statistics
        self.episode_returns = []

    def select_action(self, obs: torch.Tensor, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if eval_mode or np.random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.q_network(obs)
                action = q_values.argmax(dim=-1).item()
        else:
            action = np.random.randint(0, self.num_actions)

        return action

    def update(
        self, batch_size: int = 256, gamma: float = 0.99, num_augmentations: int = 2
    ) -> Dict:
        """
        Update Q-network with RAD augmentation.

        Args:
            batch_size: Mini-batch size
            gamma: Discount factor
            num_augmentations: Number of augmented views per sample

        Returns:
            stats: Training statistics
        """
        # Sample from buffer
        obs, actions, rewards, next_obs, dones = self.buffer.sample(batch_size)

        total_loss = 0.0
        q_values_list = []

        # Multiple augmented views (RAD technique)
        for k in range(num_augmentations):
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

            # Target Q-values
            with torch.no_grad():
                next_q_values = self.target_q_network(aug_next_obs)
                max_next_q_values = next_q_values.max(dim=-1)[0]
                q_target = rewards + (1 - dones) * gamma * max_next_q_values

            # Loss for this augmentation
            loss_k = F.mse_loss(q_values_actions, q_target)
            total_loss += loss_k

            q_values_list.append(q_values_actions.mean().item())

        # Average loss over augmentations
        loss = total_loss / num_augmentations

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self._soft_update()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {
            "loss": loss.item(),
            "q_values": np.mean(q_values_list),
            "epsilon": self.epsilon,
        }

    def _soft_update(self):
        """Soft update of target network."""
        for target_param, param in zip(
            self.target_q_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train(
        self, num_steps: int = 30000, batch_size: int = 256, warmup_steps: int = 1000
    ) -> Dict:
        """Train the RAD agent."""
        print(f"\n{'='*60}")
        print(
            f"Training with augmentation: {self.aug_type if self.use_augmentation else 'None'}"
        )
        print(f"{'='*60}\n")

        # Warmup
        if len(self.buffer) < warmup_steps:
            print(f"Warmup: collecting {warmup_steps} random transitions...")
            obs, _ = self.env.reset()
            obs_array = np.array(obs)

            for step in range(warmup_steps):
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_obs_array = np.array(next_obs)
                done = terminated or truncated

                self.buffer.add(obs_array, action, reward, next_obs_array, done)

                if done:
                    obs, _ = self.env.reset()
                    obs_array = np.array(obs)
                else:
                    obs_array = next_obs_array

            print(f"Warmup complete!\n")

        # Training
        history = {"step": [], "mean_return": [], "loss": [], "q_values": []}

        obs, _ = self.env.reset()
        obs_array = np.array(obs)
        episode_return = 0

        for step in range(num_steps):
            # Select action
            obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
            action = self.select_action(obs_tensor, eval_mode=False)

            # Step environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs_array = np.array(next_obs)
            done = terminated or truncated

            # Store transition
            self.buffer.add(obs_array, action, reward, next_obs_array, done)

            episode_return += reward

            # Update
            if len(self.buffer) >= batch_size:
                stats = self.update(batch_size)

            # Handle episode end
            if done:
                self.episode_returns.append(episode_return)
                obs, _ = self.env.reset()
                obs_array = np.array(obs)
                episode_return = 0
            else:
                obs_array = next_obs_array

            # Logging
            if (step + 1) % 5000 == 0 and len(self.episode_returns) > 0:
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

                print(
                    f"Step {step + 1:6d}/{num_steps} | Return: {mean_return:7.2f} | "
                    f"Loss: {stats['loss']:.4f} | Q: {stats['q_values']:.2f}"
                )

        print(f"\nTraining complete!\n")

        return history

    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate trained agent."""
        returns = []

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            obs_array = np.array(obs)
            episode_return = 0
            done = False

            while not done:
                obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
                action = self.select_action(obs_tensor, eval_mode=True)

                obs, reward, terminated, truncated, _ = self.env.step(action)
                obs_array = np.array(obs)
                done = terminated or truncated

                episode_return += reward

            returns.append(episode_return)

        return {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "returns": returns,
        }


class Visualizer:
    """Visualization utilities for RAD analysis."""

    @staticmethod
    def visualize_augmentations(
        env, augmentation: RADAugmentations, save_path: str = None
    ):
        """Visualize all augmentation strategies."""
        obs, _ = env.reset()
        obs_array = np.array(obs)
        obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)

        aug_types = [
            "none",
            "crop",
            "shift",
            "cutout",
            "cutout_color",
            "flip",
            "color_jitter",
            "grayscale",
        ]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()

        for idx, aug_type in enumerate(aug_types):
            aug_obs = augmentation.apply(obs_tensor, aug_type)

            # Convert to displayable format
            img = aug_obs[0].cpu().numpy()
            if img.shape[0] == 1:  # Grayscale
                img = img[0]
            else:  # RGB or stacked frames
                img = img[0]  # Take first frame

            axes[idx].imshow(img, cmap="gray")
            axes[idx].set_title(
                aug_type.replace("_", " ").title(), fontsize=11, fontweight="bold"
            )
            axes[idx].axis("off")

        plt.suptitle("RAD Augmentation Strategies", fontsize=15, fontweight="bold")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_comparison(results_dict: Dict[str, Dict], save_path: str = None):
        """Compare performance across augmentation strategies."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Sort by mean return
        sorted_items = sorted(
            results_dict.items(), key=lambda x: x[1]["mean_return"], reverse=True
        )

        names = [item[0] for item in sorted_items]
        means = [item[1]["mean_return"] for item in sorted_items]
        stds = [item[1]["std_return"] for item in sorted_items]

        # Color palette
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))

        # Bar plot
        x_pos = np.arange(len(names))
        axes[0].bar(
            x_pos,
            means,
            yerr=stds,
            color=colors,
            alpha=0.7,
            capsize=8,
            edgecolor="black",
            linewidth=1.5,
        )
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(names, fontsize=10, rotation=45, ha="right")
        axes[0].set_ylabel("Average Return", fontsize=12)
        axes[0].set_title("Performance Comparison", fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3, axis="y")

        # Box plot
        data = [results_dict[name]["returns"] for name in names]
        bp = axes[1].boxplot(data, labels=names, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_xticklabels(names, fontsize=10, rotation=45, ha="right")
        axes[1].set_ylabel("Episode Return", fontsize=12)
        axes[1].set_title("Return Distribution", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_learning_curves(histories: Dict[str, Dict], save_path: str = None):
        """Plot learning curves for different augmentation strategies."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

        for idx, (name, history) in enumerate(histories.items()):
            if "mean_return" in history and len(history["mean_return"]) > 0:
                ax.plot(
                    history["step"],
                    history["mean_return"],
                    label=name,
                    color=colors[idx],
                    linewidth=2,
                    marker="o",
                )

        ax.set_xlabel("Training Steps", fontsize=13)
        ax.set_ylabel("Mean Episode Return", fontsize=13)
        ax.set_title(
            "Learning Curves: Augmentation Strategy Comparison",
            fontsize=15,
            fontweight="bold",
        )
        ax.legend(fontsize=11, loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def main():
    """Main RAD demonstration with ablation study."""
    print("\n" + "=" * 80)
    print(" RAD: AUGMENTATION STRATEGY COMPARISON ".center(80, "="))
    print("=" * 80 + "\n")

    # ==================== Configuration ====================
    ENV_NAME = "CartPole-v1"
    IMG_SIZE = 84
    STACK_FRAMES = 4
    GRAYSCALE = True

    NUM_STEPS = 20000  # Reduced for quick comparison
    BATCH_SIZE = 256
    WARMUP_STEPS = 1000
    NUM_EVAL_EPISODES = 20

    # Augmentation strategies to compare
    AUG_STRATEGIES = ["none", "crop", "shift", "cutout", "color_jitter"]

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
    print(f"Number of actions: {num_actions}\n")

    # Create augmentation module
    augmentation = RADAugmentations(img_size=IMG_SIZE)

    # ==================== Visualize Augmentations ====================
    print("\n" + "=" * 80)
    print(" VISUALIZING AUGMENTATION STRATEGIES ".center(80))
    print("=" * 80 + "\n")

    visualizer = Visualizer()
    visualizer.visualize_augmentations(
        env, augmentation, save_path="rad_augmentations.png"
    )

    # ==================== Ablation Study ====================
    print("\n" + "=" * 80)
    print(" ABLATION STUDY: COMPARING AUGMENTATION STRATEGIES ".center(80))
    print("=" * 80)

    histories = {}
    results_dict = {}

    for aug_type in AUG_STRATEGIES:
        print(f"\n{'='*80}")
        print(f" Training with: {aug_type.upper()} ".center(80))
        print(f"{'='*80}\n")

        # Create agent
        use_aug = aug_type != "none"
        agent = RADAgent(
            env,
            input_channels,
            num_actions,
            augmentation,
            aug_type=aug_type,
            use_augmentation=use_aug,
        )

        # Train
        history = agent.train(
            num_steps=NUM_STEPS, batch_size=BATCH_SIZE, warmup_steps=WARMUP_STEPS
        )
        histories[aug_type] = history

        # Evaluate
        results = agent.evaluate(num_episodes=NUM_EVAL_EPISODES)
        results_dict[aug_type] = results

        print(
            f"\n{aug_type.upper()} - Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}\n"
        )

    # ==================== Visualizations ====================
    print("\n" + "=" * 80)
    print(" GENERATING VISUALIZATIONS ".center(80))
    print("=" * 80 + "\n")

    print("Plotting performance comparison...")
    visualizer.plot_comparison(results_dict, save_path="rad_performance_comparison.png")

    print("Plotting learning curves...")
    visualizer.plot_learning_curves(histories, save_path="rad_learning_curves.png")

    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print(" FINAL SUMMARY ".center(80, "="))
    print("=" * 80 + "\n")

    print("Performance Results (sorted by mean return):")
    sorted_results = sorted(
        results_dict.items(), key=lambda x: x[1]["mean_return"], reverse=True
    )

    for rank, (name, results) in enumerate(sorted_results, 1):
        print(
            f"  {rank}. {name:15s}: {results['mean_return']:6.2f} ± {results['std_return']:5.2f}"
        )

    print()

    # Calculate improvements
    baseline = results_dict["none"]["mean_return"]
    best_aug = sorted_results[0]
    improvement = (
        ((best_aug[1]["mean_return"] - baseline) / baseline * 100)
        if baseline > 0
        else 0
    )

    print(f"Best augmentation: {best_aug[0]}")
    print(f"Improvement over no augmentation: {improvement:.1f}%")
    print()

    print("Key Findings:")
    print("  1. Random crop is typically most effective")
    print("  2. Different tasks may benefit from different augmentations")
    print("  3. Simple augmentations often outperform complex ones")
    print("  4. Combining multiple augmentations can help")
    print("  5. Task-specific augmentation selection is important")
    print()

    print("RAD Best Practices:")
    print("  - Start with random crop (most reliable)")
    print("  - Avoid augmentations that break task semantics")
    print("  - Use 2-4 augmented views per sample")
    print("  - Combine with other techniques (CURL, DrQ)")
    print("  - Ablation studies guide augmentation selection")

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE ".center(80, "="))
    print("=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    main()
