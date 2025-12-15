"""
Domain Randomization for Sim-to-Real Transfer
==============================================

This demo illustrates domain randomization for robust visual policies:
1. Randomization of visual appearance (colors, lighting, noise, etc.)
2. Training on randomized simulations
3. Transfer to canonical/"real" environment
4. Comparison: with vs without randomization
5. Analysis of generalization and robustness

Lecture 13: Visual Policy Learning - From Imitation to Foundation Models
Prof. David Olivieri - UVigo - VIAR25/26

Key Concepts Demonstrated:
--------------------------
- Domain randomization for sim-to-real transfer
- Visual appearance randomization
- Training on distribution of environments
- Zero-shot transfer to real world
- Robustness through diversity

Reference: Tobin et al., "Domain Randomization for Transferring Deep Neural 
           Networks from Simulation to the Real World" (IROS 2017)
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
warnings.filterwarnings('ignore')

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
        frame = self.env.render()
        if frame is None:
            frame = np.zeros((*self.img_size, 3), dtype=np.uint8)
        
        if frame.shape[:2] != self.img_size:
            import cv2
            frame = cv2.resize(frame, self.img_size)
        
        return frame.astype(np.uint8)


class DomainRandomization:
    """
    Domain randomization module for visual robustness.
    Randomizes various visual aspects of observations.
    """
    def __init__(self, randomize: bool = True):
        self.randomize = randomize
    
    def randomize_colors(self, obs: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """
        Randomize color channels.
        
        Args:
            obs: Observation [H, W, C]
            intensity: Randomization intensity
        
        Returns:
            randomized: Color-randomized observation
        """
        if not self.randomize or len(obs.shape) < 3 or obs.shape[-1] != 3:
            return obs
        
        randomized = obs.astype(np.float32)
        
        # Random color multipliers for each channel
        color_mult = np.random.uniform(1.0 - intensity, 1.0 + intensity, size=3)
        
        for c in range(3):
            randomized[:, :, c] *= color_mult[c]
        
        randomized = np.clip(randomized, 0, 255).astype(np.uint8)
        
        return randomized
    
    def randomize_brightness(self, obs: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """
        Randomize brightness.
        
        Args:
            obs: Observation
            intensity: Randomization intensity
        
        Returns:
            randomized: Brightness-randomized observation
        """
        if not self.randomize:
            return obs
        
        brightness_mult = np.random.uniform(1.0 - intensity, 1.0 + intensity)
        
        randomized = obs.astype(np.float32) * brightness_mult
        randomized = np.clip(randomized, 0, 255).astype(np.uint8)
        
        return randomized
    
    def randomize_contrast(self, obs: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """
        Randomize contrast.
        
        Args:
            obs: Observation
            intensity: Randomization intensity
        
        Returns:
            randomized: Contrast-randomized observation
        """
        if not self.randomize:
            return obs
        
        contrast_mult = np.random.uniform(1.0 - intensity, 1.0 + intensity)
        
        mean = obs.mean()
        randomized = (obs.astype(np.float32) - mean) * contrast_mult + mean
        randomized = np.clip(randomized, 0, 255).astype(np.uint8)
        
        return randomized
    
    def add_noise(self, obs: np.ndarray, noise_level: float = 10.0) -> np.ndarray:
        """
        Add Gaussian noise.
        
        Args:
            obs: Observation
            noise_level: Standard deviation of noise
        
        Returns:
            noisy: Noisy observation
        """
        if not self.randomize:
            return obs
        
        noise = np.random.normal(0, noise_level, obs.shape)
        noisy = obs.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
    
    def randomize_hue(self, obs: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
        """
        Randomize hue (color shift).
        
        Args:
            obs: Observation [H, W, C]
            shift_range: Hue shift range
        
        Returns:
            randomized: Hue-shifted observation
        """
        if not self.randomize or len(obs.shape) < 3 or obs.shape[-1] != 3:
            return obs
        
        # Simple hue shift by rotating RGB values
        shift = np.random.uniform(-shift_range, shift_range)
        
        randomized = obs.astype(np.float32)
        
        # Circular shift of channels (simplified hue change)
        if np.random.random() < abs(shift):
            randomized = np.roll(randomized, shift=1, axis=-1)
        
        randomized = randomized.astype(np.uint8)
        
        return randomized
    
    def randomize_saturation(self, obs: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """
        Randomize saturation.
        
        Args:
            obs: Observation [H, W, C]
            intensity: Randomization intensity
        
        Returns:
            randomized: Saturation-randomized observation
        """
        if not self.randomize or len(obs.shape) < 3 or obs.shape[-1] != 3:
            return obs
        
        # Convert to grayscale and blend
        gray = obs.mean(axis=-1, keepdims=True).astype(np.uint8)
        gray = np.repeat(gray, 3, axis=-1)
        
        # Random saturation (blend between color and grayscale)
        saturation = np.random.uniform(1.0 - intensity, 1.0 + intensity)
        saturation = np.clip(saturation, 0.0, 2.0)
        
        if saturation < 1.0:
            # Desaturate
            randomized = (obs * saturation + gray * (1.0 - saturation)).astype(np.uint8)
        else:
            # Oversaturate (simplified)
            randomized = obs
        
        return randomized
    
    def apply_all(self, obs: np.ndarray, 
                  color_intensity: float = 0.2,
                  brightness_intensity: float = 0.2,
                  contrast_intensity: float = 0.2,
                  noise_level: float = 5.0,
                  hue_shift: float = 0.1,
                  saturation_intensity: float = 0.2) -> np.ndarray:
        """
        Apply all randomization techniques.
        
        Args:
            obs: Observation
            Various intensity parameters
        
        Returns:
            randomized: Fully randomized observation
        """
        if not self.randomize:
            return obs
        
        randomized = obs.copy()
        
        # Apply randomizations in sequence
        randomized = self.randomize_colors(randomized, color_intensity)
        randomized = self.randomize_brightness(randomized, brightness_intensity)
        randomized = self.randomize_contrast(randomized, contrast_intensity)
        randomized = self.randomize_hue(randomized, hue_shift)
        randomized = self.randomize_saturation(randomized, saturation_intensity)
        randomized = self.add_noise(randomized, noise_level)
        
        return randomized
    
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply domain randomization to observation.
        
        Args:
            obs: Observation
        
        Returns:
            randomized: Domain-randomized observation
        """
        return self.apply_all(obs)


class DomainRandomizedEnv(gym.Wrapper):
    """
    Environment wrapper that applies domain randomization.
    """
    def __init__(self, env, randomize: bool = True, 
                 color_intensity: float = 0.2,
                 brightness_intensity: float = 0.2,
                 contrast_intensity: float = 0.2,
                 noise_level: float = 5.0):
        super().__init__(env)
        
        self.domain_randomizer = DomainRandomization(randomize=randomize)
        self.color_intensity = color_intensity
        self.brightness_intensity = brightness_intensity
        self.contrast_intensity = contrast_intensity
        self.noise_level = noise_level
    
    def _randomize_obs(self, obs):
        """Apply domain randomization to observation."""
        # Handle stacked frames
        if isinstance(obs, tuple):
            # LazyFrames
            obs_array = np.array(obs)
        else:
            obs_array = obs
        
        # If grayscale (single channel), skip color randomizations
        if len(obs_array.shape) == 2:
            # Add channel dimension
            obs_array = obs_array[:, :, np.newaxis]
            randomized = self.domain_randomizer.add_noise(obs_array, self.noise_level)
            randomized = self.domain_randomizer.randomize_brightness(randomized, self.brightness_intensity)
            randomized = randomized.squeeze(-1)
        elif obs_array.shape[-1] == 1:
            # Grayscale with channel
            randomized = self.domain_randomizer.add_noise(obs_array, self.noise_level)
            randomized = self.domain_randomizer.randomize_brightness(randomized, self.brightness_intensity)
        else:
            # RGB or multi-channel
            randomized = self.domain_randomizer.apply_all(
                obs_array,
                color_intensity=self.color_intensity,
                brightness_intensity=self.brightness_intensity,
                contrast_intensity=self.contrast_intensity,
                noise_level=self.noise_level
            )
        
        return randomized
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_randomized = self._randomize_obs(obs)
        return obs_randomized, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_randomized = self._randomize_obs(obs)
        return obs_randomized, reward, terminated, truncated, info


def make_visual_env(env_name: str = "CartPole-v1", img_size: int = 84, 
                    stack_frames: int = 4, grayscale: bool = True,
                    randomize: bool = False):
    """Create a visual control environment with optional domain randomization."""
    env = gym.make(env_name, render_mode="rgb_array")
    env = VisualWrapper(env, img_size=(img_size, img_size))
    
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=False)
    
    # Apply domain randomization before frame stacking
    if randomize:
        env = DomainRandomizedEnv(env, randomize=True)
    
    env = FrameStack(env, num_stack=stack_frames)
    
    return env


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


class PolicyNetwork(nn.Module):
    """Policy network for domain randomization experiments."""
    def __init__(self, input_channels: int, num_actions: int,
                 feature_dim: int = 256, hidden_dim: int = 256):
        super(PolicyNetwork, self).__init__()
        
        self.encoder = CNNEncoder(input_channels, feature_dim)
        
        self.policy_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
        
        self._init_heads()
    
    def _init_heads(self):
        for layer in self.policy_head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        logits = self.policy_head(features)
        return logits
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        """Select action from policy."""
        with torch.no_grad():
            logits = self.forward(obs)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                action = torch.multinomial(probs, 1).item()
        
        return action


class SimpleRLTrainer:
    """Simple RL trainer for domain randomization experiments."""
    def __init__(self, env, policy: PolicyNetwork, learning_rate: float = 1e-4):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        
        # Epsilon-greedy exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Statistics
        self.episode_returns = []
    
    def collect_episode(self) -> Tuple[List, List, List, float]:
        """Collect a single episode."""
        states = []
        actions = []
        rewards = []
        
        obs, _ = self.env.reset()
        obs_array = np.array(obs)
        episode_return = 0
        done = False
        
        while not done:
            # Epsilon-greedy action selection
            if np.random.random() > self.epsilon:
                obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
                action = self.policy.get_action(obs_tensor, deterministic=False)
            else:
                action = self.env.action_space.sample()
            
            states.append(obs_array)
            actions.append(action)
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            rewards.append(reward)
            episode_return += reward
            
            obs_array = np.array(next_obs)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return states, actions, rewards, episode_return
    
    def train_on_episode(self, states: List, actions: List, rewards: List):
        """Train policy on episode using REINFORCE."""
        # Convert to tensors
        states_tensor = torch.stack([torch.from_numpy(s) for s in states]).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        
        # Compute returns (simple Monte Carlo)
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        returns_tensor = torch.FloatTensor(returns).to(device)
        
        # Normalize returns
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        
        # Forward pass
        logits = self.policy(states_tensor)
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions_tensor.unsqueeze(-1)).squeeze(-1)
        
        # Policy gradient loss
        loss = -(action_log_probs * returns_tensor).mean()
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes: int = 200) -> Dict:
        """Train the policy."""
        print(f"\nTraining for {num_episodes} episodes...")
        
        history = {'episode': [], 'return': [], 'loss': []}
        
        for episode in range(num_episodes):
            # Collect episode
            states, actions, rewards, episode_return = self.collect_episode()
            self.episode_returns.append(episode_return)
            
            # Train on episode
            loss = self.train_on_episode(states, actions, rewards)
            
            # Record
            history['episode'].append(episode)
            history['return'].append(episode_return)
            history['loss'].append(loss)
            
            # Log
            if (episode + 1) % 20 == 0:
                recent_returns = self.episode_returns[-20:]
                mean_return = np.mean(recent_returns)
                print(f"Episode {episode + 1:3d}/{num_episodes} | "
                      f"Return: {episode_return:6.1f} | "
                      f"Avg (20): {mean_return:6.1f} | "
                      f"Epsilon: {self.epsilon:.3f}")
        
        print("Training complete!\n")
        
        return history
    
    def evaluate(self, num_episodes: int = 20, env=None) -> Dict:
        """Evaluate the policy."""
        eval_env = env if env is not None else self.env
        returns = []
        
        for episode in range(num_episodes):
            obs, _ = eval_env.reset()
            obs_array = np.array(obs)
            episode_return = 0
            done = False
            
            while not done:
                obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
                action = self.policy.get_action(obs_tensor, deterministic=True)
                
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                obs_array = np.array(obs)
                done = terminated or truncated
                
                episode_return += reward
            
            returns.append(episode_return)
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'returns': returns
        }


class Visualizer:
    """Visualization utilities for domain randomization analysis."""
    
    @staticmethod
    def visualize_randomization_effects(env_canonical, env_randomized, save_path: str = None):
        """Visualize domain randomization effects."""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Collect samples
        obs_canonical, _ = env_canonical.reset()
        
        # Canonical observation
        img_canonical = np.array(obs_canonical)[0]  # First frame
        axes[0, 0].imshow(img_canonical, cmap='gray')
        axes[0, 0].set_title('Canonical (Original)', fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Multiple randomized versions
        for i in range(3):
            obs_rand, _ = env_randomized.reset()
            img_rand = np.array(obs_rand)[0]  # First frame
            axes[0, i+1].imshow(img_rand, cmap='gray')
            axes[0, i+1].set_title(f'Randomized {i+1}', fontsize=11, fontweight='bold')
            axes[0, i+1].axis('off')
        
        # Second row: more examples
        for i in range(4):
            obs_rand, _ = env_randomized.reset()
            img_rand = np.array(obs_rand)[0]
            axes[1, i].imshow(img_rand, cmap='gray')
            axes[1, i].set_title(f'Randomized {i+4}', fontsize=11, fontweight='bold')
            axes[1, i].axis('off')
        
        plt.suptitle('Domain Randomization Effects', fontsize=15, fontweight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_training_comparison(history_rand: Dict, history_no_rand: Dict,
                                save_path: str = None):
        """Plot training curves for randomized vs non-randomized."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Smooth curves
        def smooth(data, window=10):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # Returns
        axes[0].plot(history_no_rand['episode'], 
                    smooth(history_no_rand['return']),
                    color='#e74c3c', linewidth=2, label='No Randomization', alpha=0.8)
        axes[0].plot(history_rand['episode'], 
                    smooth(history_rand['return']),
                    color='#2ecc71', linewidth=2, label='With Randomization', alpha=0.8)
        axes[0].set_xlabel('Episode', fontsize=12)
        axes[0].set_ylabel('Episode Return', fontsize=12)
        axes[0].set_title('Training Performance', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(history_no_rand['episode'], 
                    smooth(history_no_rand['loss']),
                    color='#e74c3c', linewidth=2, label='No Randomization', alpha=0.8)
        axes[1].plot(history_rand['episode'], 
                    smooth(history_rand['loss']),
                    color='#2ecc71', linewidth=2, label='With Randomization', alpha=0.8)
        axes[1].set_xlabel('Episode', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_transfer_comparison(results_dict: Dict[str, Dict], save_path: str = None):
        """Plot sim-to-real transfer comparison."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        labels = list(results_dict.keys())
        means = [results_dict[k]['mean_return'] for k in labels]
        stds = [results_dict[k]['std_return'] for k in labels]
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        
        x_pos = np.arange(len(labels))
        axes[0].bar(x_pos, means, yerr=stds, color=colors[:len(labels)], alpha=0.7,
                   capsize=10, edgecolor='black', linewidth=1.5)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
        axes[0].set_ylabel('Average Return', fontsize=12)
        axes[0].set_title('Transfer Performance Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Box plot
        data = [results_dict[k]['returns'] for k in labels]
        bp = axes[1].boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(labels)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_xticklabels(labels, fontsize=10, rotation=15, ha='right')
        axes[1].set_ylabel('Episode Return', fontsize=12)
        axes[1].set_title('Return Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main domain randomization demonstration."""
    print("\n" + "="*80)
    print(" DOMAIN RANDOMIZATION FOR SIM-TO-REAL TRANSFER ".center(80, "="))
    print("="*80 + "\n")
    
    # ==================== Configuration ====================
    ENV_NAME = "CartPole-v1"
    IMG_SIZE = 84
    STACK_FRAMES = 4
    GRAYSCALE = True
    
    NUM_TRAIN_EPISODES = 200
    NUM_EVAL_EPISODES = 30
    LEARNING_RATE = 1e-3
    
    # ==================== Create Environments ====================
    print("Creating environments...")
    
    # Canonical environment (represents "real world")
    env_canonical = make_visual_env(ENV_NAME, IMG_SIZE, STACK_FRAMES, GRAYSCALE, randomize=False)
    
    # Randomized environment (training environment)
    env_randomized = make_visual_env(ENV_NAME, IMG_SIZE, STACK_FRAMES, GRAYSCALE, randomize=True)
    
    obs, _ = env_canonical.reset()
    obs_array = np.array(obs)
    input_channels = obs_array.shape[0] if len(obs_array.shape) == 3 else 1
    num_actions = env_canonical.action_space.n
    
    print(f"Environment: {ENV_NAME}")
    print(f"Observation shape: {obs_array.shape}")
    print(f"Input channels: {input_channels}")
    print(f"Number of actions: {num_actions}\n")
    
    # ==================== Visualize Randomization ====================
    print("\n" + "="*80)
    print(" VISUALIZING DOMAIN RANDOMIZATION EFFECTS ".center(80))
    print("="*80 + "\n")
    
    visualizer = Visualizer()
    visualizer.visualize_randomization_effects(env_canonical, env_randomized,
                                              save_path='domain_randomization_effects.png')
    
    # ==================== Train with Randomization ====================
    print("\n" + "="*80)
    print(" TRAINING WITH DOMAIN RANDOMIZATION ".center(80))
    print("="*80)
    
    policy_rand = PolicyNetwork(input_channels, num_actions).to(device)
    trainer_rand = SimpleRLTrainer(env_randomized, policy_rand, LEARNING_RATE)
    history_rand = trainer_rand.train(num_episodes=NUM_TRAIN_EPISODES)
    
    # ==================== Train without Randomization ====================
    print("\n" + "="*80)
    print(" TRAINING WITHOUT DOMAIN RANDOMIZATION (BASELINE) ".center(80))
    print("="*80)
    
    policy_no_rand = PolicyNetwork(input_channels, num_actions).to(device)
    trainer_no_rand = SimpleRLTrainer(env_canonical, policy_no_rand, LEARNING_RATE)
    history_no_rand = trainer_no_rand.train(num_episodes=NUM_TRAIN_EPISODES)
    
    # ==================== Evaluate Transfer ====================
    print("\n" + "="*80)
    print(" EVALUATING SIM-TO-REAL TRANSFER ".center(80))
    print("="*80 + "\n")
    
    print("Evaluating policies on canonical (real) environment...")
    
    # Policy trained with randomization, tested on canonical
    results_rand_to_canonical = trainer_rand.evaluate(NUM_EVAL_EPISODES, env_canonical)
    print(f"Randomized → Canonical: {results_rand_to_canonical['mean_return']:.2f} ± "
          f"{results_rand_to_canonical['std_return']:.2f}")
    
    # Policy trained without randomization, tested on canonical (sanity check)
    results_no_rand_to_canonical = trainer_no_rand.evaluate(NUM_EVAL_EPISODES, env_canonical)
    print(f"No Rand → Canonical:    {results_no_rand_to_canonical['mean_return']:.2f} ± "
          f"{results_no_rand_to_canonical['std_return']:.2f}")
    
    # Policy trained with randomization, tested on randomized
    results_rand_to_rand = trainer_rand.evaluate(NUM_EVAL_EPISODES, env_randomized)
    print(f"Randomized → Randomized: {results_rand_to_rand['mean_return']:.2f} ± "
          f"{results_rand_to_rand['std_return']:.2f}")
    
    # Policy trained without randomization, tested on randomized (worst case)
    results_no_rand_to_rand = trainer_no_rand.evaluate(NUM_EVAL_EPISODES, env_randomized)
    print(f"No Rand → Randomized:   {results_no_rand_to_rand['mean_return']:.2f} ± "
          f"{results_no_rand_to_rand['std_return']:.2f}")
    
    # ==================== Visualizations ====================
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATIONS ".center(80))
    print("="*80 + "\n")
    
    print("Plotting training comparison...")
    visualizer.plot_training_comparison(history_rand, history_no_rand,
                                       save_path='domain_rand_training.png')
    
    print("Plotting transfer comparison...")
    transfer_results = {
        'Rand→Real': results_rand_to_canonical,
        'NoRand→Real': results_no_rand_to_canonical,
        'Rand→Rand': results_rand_to_rand,
        'NoRand→Rand': results_no_rand_to_rand
    }
    visualizer.plot_transfer_comparison(transfer_results,
                                       save_path='domain_rand_transfer.png')
    
    # ==================== Summary ====================
    print("\n" + "="*80)
    print(" FINAL SUMMARY ".center(80, "="))
    print("="*80 + "\n")
    
    print("Transfer Performance:")
    print(f"  Randomized → Real:    {results_rand_to_canonical['mean_return']:6.2f} ± {results_rand_to_canonical['std_return']:5.2f}")
    print(f"  No Rand → Real:       {results_no_rand_to_canonical['mean_return']:6.2f} ± {results_no_rand_to_canonical['std_return']:5.2f}")
    print(f"  Randomized → Rand:    {results_rand_to_rand['mean_return']:6.2f} ± {results_rand_to_rand['std_return']:5.2f}")
    print(f"  No Rand → Rand:       {results_no_rand_to_rand['mean_return']:6.2f} ± {results_no_rand_to_rand['std_return']:5.2f}")
    print()
    
    # Compute robustness metrics
    robustness_rand = min(results_rand_to_canonical['mean_return'], 
                         results_rand_to_rand['mean_return'])
    robustness_no_rand = min(results_no_rand_to_canonical['mean_return'],
                            results_no_rand_to_rand['mean_return'])
    
    print(f"Robustness (min performance):")
    print(f"  With Randomization:    {robustness_rand:.2f}")
    print(f"  Without Randomization: {robustness_no_rand:.2f}")
    print(f"  Improvement:           {((robustness_rand - robustness_no_rand) / robustness_no_rand * 100):.1f}%")
    print()
    
    print("Key Domain Randomization Insights:")
    print("  1. Training on diverse visuals improves robustness")
    print("  2. Policies generalize better to unseen appearances")
    print("  3. Sim-to-real gap reduced through randomization")
    print("  4. No free lunch: may slow initial learning")
    print("  5. Essential for real-world deployment")
    print()
    
    print("Domain Randomization Best Practices:")
    print("  - Randomize all visual aspects that vary in real world")
    print("  - Don't randomize task-relevant information")
    print("  - Start with moderate randomization intensity")
    print("  - Increase diversity gradually if needed")
    print("  - Validate transfer on real data when possible")
    
    print("\n" + "="*80)
    print(" DEMO COMPLETE ".center(80, "="))
    print("="*80 + "\n")
    
    env_canonical.close()
    env_randomized.close()


if __name__ == "__main__":
    main()