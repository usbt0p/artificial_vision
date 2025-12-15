"""
CURL: Contrastive Unsupervised Representations for Reinforcement Learning
=========================================================================

This demo illustrates CURL algorithm for learning visual representations:
1. Momentum encoder for stable contrastive learning
2. Contrastive loss (InfoNCE) with image augmentations
3. Integration with SAC for visual control
4. Auxiliary contrastive learning task
5. Improved sample efficiency through better representations

Lecture 13: Visual Policy Learning - From Imitation to Foundation Models
Prof. David Olivieri - UVigo - VIAR25/26

Key Concepts Demonstrated:
--------------------------
- Self-supervised representation learning for RL
- Contrastive learning with momentum encoder
- InfoNCE loss and positive/negative pairs
- Data augmentation for contrastive learning
- Decoupling representation learning from RL
- Sample efficiency improvements

Reference: Srinivas et al., "CURL: Contrastive Unsupervised Representations 
           for Reinforcement Learning" (ICML 2020)
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
from sklearn.manifold import TSNE
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
        frame = self.env.render()
        if frame is None:
            frame = np.zeros((*self.img_size, 3), dtype=np.uint8)
        
        if frame.shape[:2] != self.img_size:
            import cv2
            frame = cv2.resize(frame, self.img_size)
        
        return frame.astype(np.uint8)


def make_visual_env(env_name: str = "CartPole-v1", img_size: int = 84, 
                    stack_frames: int = 4, grayscale: bool = True):
    """Create a visual control environment with preprocessing."""
    env = gym.make(env_name, render_mode="rgb_array")
    env = VisualWrapper(env, img_size=(img_size, img_size))
    
    if grayscale:
        env = GrayScaleObservation(env, keep_dim=False)
    
    env = FrameStack(env, num_stack=stack_frames)
    
    return env


class RandomCropAugmentation:
    """
    Random crop augmentation for CURL.
    Creates positive pairs by cropping the same observation twice.
    """
    def __init__(self, img_size: int = 84, crop_size: int = 84):
        self.img_size = img_size
        self.crop_size = crop_size
        self.pad = 4  # Padding for random crop
        
    def __call__(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create two random crops of the observation (positive pair).
        
        Args:
            obs: Observation [batch_size, channels, height, width]
        
        Returns:
            obs_anchor: First crop
            obs_positive: Second crop (different random crop of same obs)
        """
        batch_size, channels, h, w = obs.shape
        
        # Pad observation
        padded = F.pad(obs, (self.pad, self.pad, self.pad, self.pad), mode='replicate')
        
        # Random crop 1 (anchor)
        obs_anchor = self._random_crop(padded, batch_size, channels)
        
        # Random crop 2 (positive - different crop of same observation)
        obs_positive = self._random_crop(padded, batch_size, channels)
        
        return obs_anchor, obs_positive
    
    def _random_crop(self, padded_obs: torch.Tensor, batch_size: int, 
                     channels: int) -> torch.Tensor:
        """Perform random crop on padded observation."""
        crop_max = self.img_size + 2 * self.pad - self.crop_size
        w1 = torch.randint(0, crop_max + 1, (batch_size,))
        h1 = torch.randint(0, crop_max + 1, (batch_size,))
        
        cropped = torch.zeros(batch_size, channels, self.crop_size, self.crop_size,
                             device=padded_obs.device, dtype=padded_obs.dtype)
        
        for i in range(batch_size):
            cropped[i] = padded_obs[i, :, h1[i]:h1[i]+self.crop_size, w1[i]:w1[i]+self.crop_size]
        
        return cropped


class CURLEncoder(nn.Module):
    """
    CNN encoder for CURL.
    Used for both query and key encoders.
    """
    def __init__(self, input_channels: int = 4, output_dim: int = 128):
        super(CURLEncoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        
        # Calculate output size for 84x84 input
        def conv_output_size(size, kernel_size, stride, padding):
            return (size + 2 * padding - kernel_size) // stride + 1
        
        h = conv_output_size(84, 3, 2, 1)  # After conv1
        # Remaining convs maintain size
        
        conv_output_size_flat = h * h * 32
        
        # Projection head for contrastive learning
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
    
    def forward(self, x: torch.Tensor, detach: bool = False) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Visual observations [batch_size, channels, height, width]
            detach: If True, detach gradients (for momentum encoder)
        
        Returns:
            features: Encoded features [batch_size, output_dim]
        """
        # Normalize pixel values
        x = x.float() / 255.0
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten
        x = x.reshape(x.size(0), -1)
        
        # Projection head
        x = self.fc(x)
        x = self.ln(x)
        
        if detach:
            x = x.detach()
        
        return x


class CURLModule(nn.Module):
    """
    CURL module implementing contrastive learning.
    Consists of query encoder and momentum key encoder.
    """
    def __init__(self, input_channels: int, output_dim: int = 128, 
                 momentum: float = 0.99):
        super(CURLModule, self).__init__()
        
        # Query encoder (trainable)
        self.query_encoder = CURLEncoder(input_channels, output_dim)
        
        # Key encoder (momentum-updated)
        self.key_encoder = CURLEncoder(input_channels, output_dim)
        
        # Initialize key encoder with same weights as query encoder
        self.key_encoder.load_state_dict(self.query_encoder.state_dict())
        
        # Key encoder parameters are not trained directly
        for param in self.key_encoder.parameters():
            param.requires_grad = False
        
        self.momentum = momentum
        
        # W matrix for bilinear product (query^T W key)
        self.W = nn.Parameter(torch.randn(output_dim, output_dim))
        nn.init.xavier_uniform_(self.W)
    
    def encode_query(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation with query encoder."""
        return self.query_encoder(obs, detach=False)
    
    def encode_key(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation with key encoder."""
        with torch.no_grad():
            return self.key_encoder(obs, detach=True)
    
    def compute_logits(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity logits between query and key.
        Uses bilinear product: query^T W key
        
        Args:
            query: Query features [batch_size, dim]
            key: Key features [batch_size, dim]
        
        Returns:
            logits: Similarity matrix [batch_size, batch_size]
        """
        # Normalize features
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        
        # Bilinear product: query^T W key
        logits = torch.matmul(torch.matmul(query, self.W), key.T)
        
        return logits
    
    @torch.no_grad()
    def update_key_encoder(self):
        """
        Momentum update of key encoder.
        key_params = m * key_params + (1 - m) * query_params
        """
        for query_param, key_param in zip(self.query_encoder.parameters(),
                                          self.key_encoder.parameters()):
            key_param.data = self.momentum * key_param.data + \
                           (1 - self.momentum) * query_param.data


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for CURL.
    Contrastive loss that pulls positive pairs together and pushes negatives apart.
    """
    def __init__(self, temperature: float = 0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            logits: Similarity matrix [batch_size, batch_size]
                   logits[i,i] is the positive pair (same observation, different crop)
                   logits[i,j] (i≠j) are negative pairs (different observations)
        
        Returns:
            loss: InfoNCE contrastive loss
        """
        batch_size = logits.shape[0]
        
        # Scale by temperature
        logits = logits / self.temperature
        
        # Labels: positive pair is on the diagonal
        labels = torch.arange(batch_size, device=logits.device)
        
        # Cross-entropy loss (softmax over all pairs, target is diagonal)
        loss = F.cross_entropy(logits, labels)
        
        return loss


class CURLSACAgent:
    """
    SAC agent augmented with CURL for representation learning.
    """
    def __init__(self, env, input_channels: int, num_actions: int,
                 curl_latent_dim: int = 128, actor_latent_dim: int = 256,
                 learning_rate: float = 1e-4, curl_weight: float = 1.0):
        self.env = env
        self.num_actions = num_actions
        self.curl_weight = curl_weight
        
        # CURL module
        self.curl = CURLModule(input_channels, curl_latent_dim).to(device)
        
        # Actor network (uses CURL encoder features)
        self.actor = nn.Sequential(
            nn.Linear(curl_latent_dim, actor_latent_dim),
            nn.ReLU(),
            nn.Linear(actor_latent_dim, num_actions)
        ).to(device)
        
        # Q-network (simple for discrete actions)
        self.q_network = nn.Sequential(
            nn.Linear(curl_latent_dim, actor_latent_dim),
            nn.ReLU(),
            nn.Linear(actor_latent_dim, num_actions)
        ).to(device)
        
        # Augmentation
        self.augmentation = RandomCropAugmentation()
        
        # Loss functions
        self.curl_loss_fn = InfoNCELoss(temperature=0.1)
        
        # Optimizers
        self.curl_optimizer = optim.Adam(self.curl.parameters(), lr=learning_rate)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
    def select_action(self, obs: torch.Tensor, eval_mode: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if eval_mode or np.random.random() > self.epsilon:
            with torch.no_grad():
                # Encode observation
                features = self.curl.encode_query(obs)
                
                # Get Q-values
                q_values = self.q_network(features)
                action = q_values.argmax(dim=-1).item()
        else:
            action = np.random.randint(0, self.num_actions)
        
        return action
    
    def update_curl(self, obs: torch.Tensor) -> Dict:
        """
        Update CURL encoder using contrastive learning.
        
        Args:
            obs: Batch of observations
        
        Returns:
            stats: Training statistics
        """
        # Create positive pairs via random cropping
        obs_anchor, obs_positive = self.augmentation(obs)
        
        # Encode anchor with query encoder
        query = self.curl.encode_query(obs_anchor)
        
        # Encode positive with key encoder (no gradients)
        key = self.curl.encode_key(obs_positive)
        
        # Compute similarity logits
        logits = self.curl.compute_logits(query, key)
        
        # InfoNCE loss
        curl_loss = self.curl_loss_fn(logits)
        
        # Update query encoder
        self.curl_optimizer.zero_grad()
        curl_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.curl.parameters(), 10.0)
        self.curl_optimizer.step()
        
        # Momentum update of key encoder
        self.curl.update_key_encoder()
        
        # Compute accuracy (positive pair should have highest similarity)
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            labels = torch.arange(obs.shape[0], device=device)
            accuracy = (predictions == labels).float().mean()
        
        stats = {
            'curl_loss': curl_loss.item(),
            'curl_accuracy': accuracy.item()
        }
        
        return stats
    
    def update_rl(self, obs: torch.Tensor, actions: torch.Tensor, 
                  rewards: torch.Tensor, next_obs: torch.Tensor,
                  dones: torch.Tensor, gamma: float = 0.99) -> Dict:
        """
        Update Q-network using standard Q-learning.
        
        Args:
            obs: Current observations
            actions: Actions taken
            rewards: Rewards received
            next_obs: Next observations
            dones: Done flags
            gamma: Discount factor
        
        Returns:
            stats: Training statistics
        """
        # Encode observations with CURL encoder
        with torch.no_grad():
            features = self.curl.encode_query(obs).detach()
            next_features = self.curl.encode_query(next_obs).detach()
        
        # Current Q-values
        q_values = self.q_network(features)
        q_values_actions = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.q_network(next_features)
            max_next_q_values = next_q_values.max(dim=-1)[0]
            q_target = rewards + (1 - dones) * gamma * max_next_q_values
        
        # Q-network loss
        q_loss = F.mse_loss(q_values_actions, q_target)
        
        # Update Q-network
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.q_optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        stats = {
            'q_loss': q_loss.item(),
            'q_values': q_values_actions.mean().item(),
            'epsilon': self.epsilon
        }
        
        return stats


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
        """Add transition to buffer."""
        self.observations[self.ptr] = torch.from_numpy(obs)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = torch.from_numpy(next_obs)
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        """Sample batch of transitions."""
        indices = torch.randint(0, self.size, (batch_size,))
        
        return (
            self.observations[indices].to(self.device),
            self.actions[indices].to(self.device),
            self.rewards[indices].to(self.device),
            self.next_observations[indices].to(self.device),
            self.dones[indices].to(self.device)
        )
    
    def __len__(self):
        return self.size


class CURLTrainer:
    """Trainer for CURL-augmented RL."""
    def __init__(self, env, agent: CURLSACAgent, buffer_capacity: int = 50000):
        self.env = env
        self.agent = agent
        
        # Get observation shape
        obs, _ = env.reset()
        obs_array = np.array(obs)
        self.obs_shape = obs_array.shape
        
        # Create replay buffer
        self.buffer = ReplayBuffer(buffer_capacity, self.obs_shape, device)
        
        # Statistics
        self.episode_returns = []
    
    def train(self, num_steps: int = 30000, batch_size: int = 128,
              warmup_steps: int = 1000, update_freq: int = 1,
              curl_update_freq: int = 1) -> Dict:
        """
        Train CURL agent.
        
        Args:
            num_steps: Total training steps
            batch_size: Mini-batch size
            warmup_steps: Random exploration steps
            update_freq: RL update frequency
            curl_update_freq: CURL update frequency
        
        Returns:
            history: Training history
        """
        print(f"\n{'='*80}")
        print(f"Starting CURL Training")
        print(f"{'='*80}")
        print(f"Total steps: {num_steps:,}")
        print(f"Batch size: {batch_size}")
        print(f"Warmup steps: {warmup_steps:,}")
        print(f"{'='*80}\n")
        
        # Warmup with random actions
        print(f"Warmup phase: collecting {warmup_steps} random transitions...")
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
        
        print(f"Warmup complete! Buffer size: {len(self.buffer)}\n")
        
        # Training
        history = {
            'step': [],
            'mean_return': [],
            'curl_loss': [],
            'curl_accuracy': [],
            'q_loss': [],
            'q_values': []
        }
        
        obs, _ = self.env.reset()
        obs_array = np.array(obs)
        episode_return = 0
        
        for step in range(num_steps):
            # Select action
            obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
            action = self.agent.select_action(obs_tensor, eval_mode=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            next_obs_array = np.array(next_obs)
            done = terminated or truncated
            
            # Store transition
            self.buffer.add(obs_array, action, reward, next_obs_array, done)
            
            episode_return += reward
            
            # Update networks
            if len(self.buffer) >= batch_size:
                # Sample batch
                batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = \
                    self.buffer.sample(batch_size)
                
                # Update CURL
                if step % curl_update_freq == 0:
                    curl_stats = self.agent.update_curl(batch_obs)
                
                # Update RL
                if step % update_freq == 0:
                    rl_stats = self.agent.update_rl(
                        batch_obs, batch_actions, batch_rewards,
                        batch_next_obs, batch_dones
                    )
            
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
                recent_returns = self.episode_returns[-10:] if len(self.episode_returns) >= 10 else self.episode_returns
                mean_return = np.mean(recent_returns)
                
                history['step'].append(step + 1)
                history['mean_return'].append(mean_return)
                history['curl_loss'].append(curl_stats['curl_loss'])
                history['curl_accuracy'].append(curl_stats['curl_accuracy'])
                history['q_loss'].append(rl_stats['q_loss'])
                history['q_values'].append(rl_stats['q_values'])
                
                print(f"Step {step + 1:6d}/{num_steps} | "
                      f"Return: {mean_return:7.2f} | "
                      f"CURL Loss: {curl_stats['curl_loss']:.4f} | "
                      f"CURL Acc: {curl_stats['curl_accuracy']:.3f} | "
                      f"Q Loss: {rl_stats['q_loss']:.4f}")
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"{'='*80}\n")
        
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
                action = self.agent.select_action(obs_tensor, eval_mode=True)
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                obs_array = np.array(obs)
                done = terminated or truncated
                
                episode_return += reward
            
            returns.append(episode_return)
        
        return {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'returns': returns
        }
    
    def visualize_representations(self, num_samples: int = 500) -> np.ndarray:
        """
        Extract and visualize learned representations using t-SNE.
        
        Args:
            num_samples: Number of samples to visualize
        
        Returns:
            embeddings: t-SNE embeddings
        """
        print(f"\nExtracting representations from {num_samples} samples...")
        
        features_list = []
        states_list = []
        
        # Collect samples
        num_collected = 0
        obs, _ = self.env.reset()
        obs_array = np.array(obs)
        
        while num_collected < num_samples:
            obs_tensor = torch.from_numpy(obs_array).unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                features = self.agent.curl.encode_query(obs_tensor)
                features_list.append(features.cpu().numpy())
            
            # Store state for coloring
            states_list.append(obs_array[0])  # First frame
            
            # Step environment
            action = self.agent.select_action(obs_tensor, eval_mode=True)
            obs, _, terminated, truncated, _ = self.env.step(action)
            obs_array = np.array(obs)
            done = terminated or truncated
            
            if done:
                obs, _ = self.env.reset()
                obs_array = np.array(obs)
            
            num_collected += 1
        
        features_array = np.concatenate(features_list, axis=0)
        
        print(f"Running t-SNE on {features_array.shape[0]} samples...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = tsne.fit_transform(features_array)
        
        return embeddings, np.array(states_list)


class Visualizer:
    """Visualization utilities for CURL analysis."""
    
    @staticmethod
    def plot_training_curves(curl_history: Dict, baseline_history: Dict = None,
                            save_path: str = None):
        """Plot CURL training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        # Episode return
        axes[0].plot(curl_history['step'], curl_history['mean_return'],
                    color='#2ecc71', linewidth=2, label='CURL', marker='o')
        if baseline_history:
            axes[0].plot(baseline_history['step'], baseline_history['mean_return'],
                        color='#e74c3c', linewidth=2, label='Baseline', marker='s')
        axes[0].set_xlabel('Training Steps', fontsize=11)
        axes[0].set_ylabel('Mean Return', fontsize=11)
        axes[0].set_title('Policy Performance', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # CURL loss
        axes[1].plot(curl_history['step'], curl_history['curl_loss'],
                    color='#9b59b6', linewidth=2)
        axes[1].set_xlabel('Training Steps', fontsize=11)
        axes[1].set_ylabel('CURL Loss', fontsize=11)
        axes[1].set_title('Contrastive Loss (InfoNCE)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # CURL accuracy
        axes[2].plot(curl_history['step'], curl_history['curl_accuracy'],
                    color='#3498db', linewidth=2)
        axes[2].set_xlabel('Training Steps', fontsize=11)
        axes[2].set_ylabel('CURL Accuracy', fontsize=11)
        axes[2].set_title('Contrastive Accuracy', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Q loss
        axes[3].plot(curl_history['step'], curl_history['q_loss'],
                    color='#f39c12', linewidth=2, label='CURL')
        if baseline_history:
            axes[3].plot(baseline_history['step'], baseline_history['q_loss'],
                        color='#e74c3c', linewidth=2, label='Baseline')
        axes[3].set_xlabel('Training Steps', fontsize=11)
        axes[3].set_ylabel('Q Loss', fontsize=11)
        axes[3].set_title('Q-Learning Loss', fontsize=12, fontweight='bold')
        axes[3].legend(fontsize=10)
        axes[3].grid(True, alpha=0.3)
        
        # Q values
        axes[4].plot(curl_history['step'], curl_history['q_values'],
                    color='#1abc9c', linewidth=2, label='CURL')
        if baseline_history:
            axes[4].plot(baseline_history['step'], baseline_history['q_values'],
                        color='#e74c3c', linewidth=2, label='Baseline')
        axes[4].set_xlabel('Training Steps', fontsize=11)
        axes[4].set_ylabel('Average Q-values', fontsize=11)
        axes[4].set_title('Q-Value Estimates', fontsize=12, fontweight='bold')
        axes[4].legend(fontsize=10)
        axes[4].grid(True, alpha=0.3)
        
        # Sample efficiency comparison
        if baseline_history:
            curl_final = curl_history['mean_return'][-1] if len(curl_history['mean_return']) > 0 else 0
            baseline_final = baseline_history['mean_return'][-1] if len(baseline_history['mean_return']) > 0 else 0
            
            improvement = ((curl_final - baseline_final) / baseline_final * 100) if baseline_final > 0 else 0
            
            axes[5].bar(['Baseline', 'CURL'], [baseline_final, curl_final],
                       color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
            axes[5].set_ylabel('Final Return', fontsize=11)
            axes[5].set_title(f'Sample Efficiency (+{improvement:.1f}%)', 
                            fontsize=12, fontweight='bold')
            axes[5].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('CURL Training Analysis', fontsize=15, fontweight='bold', y=0.995)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_representations(embeddings: np.ndarray, states: np.ndarray = None,
                           save_path: str = None):
        """Plot t-SNE visualization of learned representations."""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        if states is not None:
            # Color by average pixel intensity
            colors = states.mean(axis=(1, 2))
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1],
                               c=colors, cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(scatter, ax=ax, label='Avg Pixel Intensity')
        else:
            ax.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=20)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title('CURL Learned Representations (t-SNE)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main CURL demonstration."""
    print("\n" + "="*80)
    print(" CURL: CONTRASTIVE UNSUPERVISED REPRESENTATIONS FOR RL ".center(80, "="))
    print("="*80 + "\n")
    
    # ==================== Configuration ====================
    ENV_NAME = "CartPole-v1"
    IMG_SIZE = 84
    STACK_FRAMES = 4
    GRAYSCALE = True
    
    NUM_STEPS = 30000
    BATCH_SIZE = 128
    WARMUP_STEPS = 1000
    CURL_LATENT_DIM = 128
    
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
    
    # ==================== Train CURL Agent ====================
    print("\n" + "="*80)
    print(" TRAINING CURL AGENT ".center(80))
    print("="*80)
    
    curl_agent = CURLSACAgent(env, input_channels, num_actions,
                             curl_latent_dim=CURL_LATENT_DIM)
    curl_trainer = CURLTrainer(env, curl_agent)
    
    curl_history = curl_trainer.train(
        num_steps=NUM_STEPS,
        batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS
    )
    
    # ==================== Evaluate ====================
    print("\n" + "="*80)
    print(" EVALUATING CURL AGENT ".center(80))
    print("="*80 + "\n")
    
    curl_results = curl_trainer.evaluate(num_episodes=NUM_EVAL_EPISODES)
    print(f"CURL - Mean Return: {curl_results['mean_return']:.2f} ± {curl_results['std_return']:.2f}")
    
    # ==================== Visualizations ====================
    print("\n" + "="*80)
    print(" GENERATING VISUALIZATIONS ".center(80))
    print("="*80 + "\n")
    
    visualizer = Visualizer()
    
    print("Plotting training curves...")
    visualizer.plot_training_curves(curl_history, save_path='curl_training_curves.png')
    
    print("Visualizing learned representations...")
    embeddings, states = curl_trainer.visualize_representations(num_samples=500)
    visualizer.plot_representations(embeddings, states, save_path='curl_representations.png')
    
    # ==================== Summary ====================
    print("\n" + "="*80)
    print(" FINAL SUMMARY ".center(80, "="))
    print("="*80 + "\n")
    
    print("Performance Results:")
    print(f"  - CURL: {curl_results['mean_return']:6.2f} ± {curl_results['std_return']:5.2f}")
    print()
    
    print("Key CURL Mechanisms Demonstrated:")
    print("  1. Contrastive learning with momentum encoder")
    print("  2. InfoNCE loss for representation learning")
    print("  3. Data augmentation creates positive pairs")
    print("  4. Decoupled representation and RL learning")
    print("  5. Improved sample efficiency")
    print()
    
    print("CURL Advantages:")
    print("  - Better visual representations")
    print("  - Improved sample efficiency")
    print("  - Auxiliary self-supervised task")
    print("  - Works with any RL algorithm")
    print("  - Momentum encoder for stability")
    
    print("\n" + "="*80)
    print(" DEMO COMPLETE ".center(80, "="))
    print("="*80 + "\n")
    
    env.close()


if __name__ == "__main__":
    main()