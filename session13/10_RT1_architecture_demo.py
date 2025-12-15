"""
RT-1: Robotics Transformer - Foundation Model Architecture Demo
================================================================

This demo illustrates key concepts from vision-language-action foundation models:
1. Vision tokenization (image patches as tokens)
2. Language instruction encoding
3. Transformer architecture for robotics
4. Action prediction via token generation
5. Attention mechanisms over vision and language
6. Multi-task learning principles
7. Foundation model scaling

Lecture 13: Visual Policy Learning - From Imitation to Foundation Models
Prof. David Olivieri - UVigo - VIAR25/26

Key Concepts Demonstrated:
--------------------------
- Transformer-based policies for robotics
- Vision-language-action integration
- Tokenization of visual observations
- Instruction conditioning
- Attention over multimodal inputs
- Foundation model principles

Reference: Brohan et al., "RT-1: Robotics Transformer for Real-World Control
           at Scale" (arXiv 2022)

Note: This is an educational implementation demonstrating architectural concepts.
      A full RT-1 would require large-scale multi-task datasets.
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
import math
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


def make_visual_env(env_name: str = "CartPole-v1", img_size: int = 84):
    """Create a visual control environment."""
    env = gym.make(env_name, render_mode="rgb_array")
    env = VisualWrapper(env, img_size=(img_size, img_size))
    # Note: Not using grayscale for RT-1 (uses RGB)
    return env


class ImageTokenizer(nn.Module):
    """
    Vision Transformer-style image tokenizer.
    Converts image into sequence of patch tokens.
    """

    def __init__(
        self,
        img_size: int = 84,
        patch_size: int = 14,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super(ImageTokenizer, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding via convolution
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embeddings (learned)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Layer norm
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images to sequence of patch tokens.

        Args:
            images: [batch_size, channels, height, width]

        Returns:
            tokens: [batch_size, num_patches, embed_dim]
        """
        batch_size = images.shape[0]

        # Normalize images
        images = images.float() / 255.0

        # Extract patches and embed
        # Conv2d output: [batch_size, embed_dim, num_patches_h, num_patches_w]
        x = self.patch_embed(images)

        # Reshape to sequence
        # [batch_size, embed_dim, num_patches_h, num_patches_w]
        # -> [batch_size, embed_dim, num_patches]
        x = x.flatten(2)

        # Transpose to [batch_size, num_patches, embed_dim]
        x = x.transpose(1, 2)

        # Add positional embeddings
        x = x + self.pos_embed

        # Layer norm
        x = self.ln(x)

        return x


class LanguageEncoder(nn.Module):
    """
    Simple language encoder for instruction conditioning.
    In real RT-1, this would be a pretrained language model (e.g., T5).
    """

    def __init__(
        self, vocab_size: int = 1000, embed_dim: int = 256, max_length: int = 20
    ):
        super(LanguageEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.max_length = max_length

        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, embed_dim))

        # Simple transformer encoder for language
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Layer norm
        self.ln = nn.LayerNorm(embed_dim)

    def forward(
        self,
        instruction_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode language instruction.

        Args:
            instruction_tokens: [batch_size, seq_length] token IDs
            attention_mask: [batch_size, seq_length] mask (1 = valid, 0 = padding)

        Returns:
            encoded: [batch_size, seq_length, embed_dim]
        """
        batch_size, seq_length = instruction_tokens.shape

        # Token embeddings
        x = self.token_embed(instruction_tokens)

        # Add positional embeddings
        x = x + self.pos_embed[:, :seq_length, :]

        # Transformer encoding
        if attention_mask is not None:
            # Convert mask: 1 -> False (attend), 0 -> True (ignore)
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Layer norm
        x = self.ln(x)

        return x


class RT1Transformer(nn.Module):
    """
    RT-1 Transformer architecture.
    Processes vision and language tokens, outputs action tokens.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        num_actions: int = 2,
    ):
        super(RT1Transformer, self).__init__()

        self.embed_dim = embed_dim

        # Transformer encoder (processes vision + language)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_actions),
        )

        # Special tokens
        self.action_query_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(
        self,
        vision_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through RT-1 transformer.

        Args:
            vision_tokens: [batch_size, num_patches, embed_dim]
            language_tokens: [batch_size, lang_seq_len, embed_dim]
            return_attention: If True, return attention weights

        Returns:
            action_logits: [batch_size, num_actions]
        """
        batch_size = vision_tokens.shape[0]

        # Concatenate vision and language tokens
        # [batch_size, num_patches + lang_seq_len, embed_dim]
        multimodal_tokens = torch.cat([vision_tokens, language_tokens], dim=1)

        # Add action query token
        action_query = self.action_query_token.expand(batch_size, -1, -1)
        tokens_with_query = torch.cat([multimodal_tokens, action_query], dim=1)

        # Process through transformer
        encoded = self.encoder(tokens_with_query)

        # Extract action token (last token)
        action_token = encoded[:, -1, :]

        # Predict action
        action_logits = self.action_head(action_token)

        if return_attention:
            # Note: Getting attention weights from nn.TransformerEncoder is tricky
            # This is simplified for demonstration
            return action_logits, None

        return action_logits


class RT1Policy(nn.Module):
    """
    Complete RT-1 policy combining all components.
    """

    def __init__(
        self,
        img_size: int = 84,
        patch_size: int = 14,
        vocab_size: int = 100,
        embed_dim: int = 256,
        num_actions: int = 2,
        max_instruction_length: int = 20,
    ):
        super(RT1Policy, self).__init__()

        # Vision tokenizer
        self.vision_tokenizer = ImageTokenizer(
            img_size=img_size, patch_size=patch_size, in_channels=3, embed_dim=embed_dim
        )

        # Language encoder
        self.language_encoder = LanguageEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_length=max_instruction_length,
        )

        # RT-1 transformer
        self.transformer = RT1Transformer(
            embed_dim=embed_dim, num_heads=8, num_layers=4, num_actions=num_actions
        )

    def forward(
        self,
        images: torch.Tensor,
        instructions: torch.Tensor,
        instruction_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ):
        """
        Forward pass through RT-1 policy.

        Args:
            images: [batch_size, channels, height, width]
            instructions: [batch_size, seq_length] token IDs
            instruction_mask: [batch_size, seq_length] attention mask
            return_attention: Whether to return attention weights

        Returns:
            action_logits: [batch_size, num_actions]
        """
        # Tokenize vision
        vision_tokens = self.vision_tokenizer(images)

        # Encode language
        language_tokens = self.language_encoder(instructions, instruction_mask)

        # Process through transformer
        action_logits = self.transformer(
            vision_tokens, language_tokens, return_attention=return_attention
        )

        return action_logits

    def get_action(
        self,
        image: np.ndarray,
        instruction: str,
        instruction_encoder,
        deterministic: bool = False,
    ) -> int:
        """
        Get action from policy given image and instruction.

        Args:
            image: Image observation
            instruction: Text instruction
            instruction_encoder: Function to encode instruction to tokens
            deterministic: If True, select argmax action

        Returns:
            action: Selected action
        """
        with torch.no_grad():
            # Prepare inputs
            image_tensor = (
                torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            instruction_tokens = (
                instruction_encoder(instruction).unsqueeze(0).to(device)
            )

            # Forward pass
            action_logits = self.forward(image_tensor, instruction_tokens)

            # Select action
            if deterministic:
                action = action_logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(action_logits, dim=-1)
                action = torch.multinomial(probs, 1).item()

        return action


class SimpleInstructionEncoder:
    """
    Simple instruction encoder for demonstration.
    Maps text instructions to token sequences.
    """

    def __init__(self, vocab_size: int = 100, max_length: int = 20):
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Simple vocabulary (for demo purposes)
        self.vocab = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "balance": 3,
            "pole": 4,
            "keep": 5,
            "upright": 6,
            "move": 7,
            "left": 8,
            "right": 9,
            "cart": 10,
            "stabilize": 11,
        }

    def encode(self, instruction: str) -> torch.Tensor:
        """
        Encode instruction to token IDs.

        Args:
            instruction: Text instruction

        Returns:
            tokens: Token IDs tensor
        """
        # Tokenize (simple word splitting)
        words = instruction.lower().split()

        # Convert to IDs
        token_ids = [self.vocab.get("<START>", 1)]
        for word in words:
            token_id = self.vocab.get(word, hash(word) % (self.vocab_size - 20) + 20)
            token_ids.append(token_id)
        token_ids.append(self.vocab.get("<END>", 2))

        # Pad or truncate
        if len(token_ids) < self.max_length:
            token_ids += [self.vocab.get("<PAD>", 0)] * (
                self.max_length - len(token_ids)
            )
        else:
            token_ids = token_ids[: self.max_length]

        return torch.LongTensor(token_ids)

    def create_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Create attention mask (1 for valid tokens, 0 for padding)."""
        return (tokens != self.vocab.get("<PAD>", 0)).long()


class RT1Trainer:
    """
    Trainer for RT-1 policy (simplified for demonstration).
    """

    def __init__(
        self, env, policy: RT1Policy, instruction_encoder, learning_rate: float = 1e-4
    ):
        self.env = env
        self.policy = policy
        self.instruction_encoder = instruction_encoder
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

        # Epsilon for exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # Statistics
        self.episode_returns = []

    def collect_episode(self, instruction: str) -> Tuple[List, List, List, float]:
        """Collect a single episode with instruction conditioning."""
        states = []
        actions = []
        rewards = []

        obs, _ = self.env.reset()
        episode_return = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if np.random.random() > self.epsilon:
                action = self.policy.get_action(
                    obs,
                    instruction,
                    self.instruction_encoder.encode,
                    deterministic=False,
                )
            else:
                action = self.env.action_space.sample()

            states.append(obs)
            actions.append(action)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            rewards.append(reward)
            episode_return += reward

            obs = next_obs

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return states, actions, rewards, episode_return

    def train_on_episode(
        self, states: List, actions: List, rewards: List, instruction: str
    ):
        """Train policy on episode."""
        # Prepare data
        images = torch.stack([torch.from_numpy(s).permute(2, 0, 1) for s in states]).to(
            device
        )

        actions_tensor = torch.LongTensor(actions).to(device)

        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        returns_tensor = torch.FloatTensor(returns).to(device)
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (
            returns_tensor.std() + 1e-8
        )

        # Encode instruction (same for all timesteps)
        instruction_tokens = self.instruction_encoder.encode(instruction)
        instruction_batch = (
            instruction_tokens.unsqueeze(0).repeat(len(states), 1).to(device)
        )

        # Forward pass
        action_logits = self.policy(images, instruction_batch)
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_probs = log_probs.gather(-1, actions_tensor.unsqueeze(-1)).squeeze(
            -1
        )

        # Policy gradient loss
        loss = -(action_log_probs * returns_tensor).mean()

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes: int = 100, instructions: List[str] = None) -> Dict:
        """Train RT-1 policy."""
        if instructions is None:
            instructions = ["balance pole upright"]

        print(f"\nTraining RT-1 for {num_episodes} episodes...")
        print(f"Instructions: {instructions}\n")

        history = {"episode": [], "return": [], "loss": [], "instruction": []}

        for episode in range(num_episodes):
            # Sample instruction (multi-task learning)
            instruction = np.random.choice(instructions)

            # Collect episode
            states, actions, rewards, episode_return = self.collect_episode(instruction)
            self.episode_returns.append(episode_return)

            # Train
            loss = self.train_on_episode(states, actions, rewards, instruction)

            # Record
            history["episode"].append(episode)
            history["return"].append(episode_return)
            history["loss"].append(loss)
            history["instruction"].append(instruction)

            # Log
            if (episode + 1) % 10 == 0:
                recent_returns = self.episode_returns[-10:]
                mean_return = np.mean(recent_returns)
                print(
                    f"Episode {episode + 1:3d}/{num_episodes} | "
                    f"Return: {episode_return:6.1f} | "
                    f"Avg: {mean_return:6.1f} | "
                    f"Instruction: '{instruction}'"
                )

        print("\nTraining complete!\n")

        return history

    def evaluate(
        self, num_episodes: int = 10, instruction: str = "balance pole upright"
    ) -> Dict:
        """Evaluate RT-1 policy."""
        returns = []

        print(f"\nEvaluating with instruction: '{instruction}'")

        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_return = 0
            done = False

            while not done:
                action = self.policy.get_action(
                    obs,
                    instruction,
                    self.instruction_encoder.encode,
                    deterministic=True,
                )
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_return += reward

            returns.append(episode_return)

        return {
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "returns": returns,
        }


class Visualizer:
    """Visualization utilities for RT-1 analysis."""

    @staticmethod
    def visualize_architecture(policy: RT1Policy, save_path: str = None):
        """Visualize RT-1 architecture components."""
        fig = plt.figure(figsize=(16, 10))

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle(
            "RT-1 Architecture: Vision-Language-Action Transformer",
            fontsize=16,
            fontweight="bold",
        )

        # Component 1: Image tokenization
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(
            0.5,
            0.5,
            "Image\n(84×84×3)",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightblue"),
        )
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis("off")
        ax1.set_title("1. Visual Input", fontweight="bold")

        # Component 2: Patch embedding
        ax2 = fig.add_subplot(gs[0, 1])
        num_patches = policy.vision_tokenizer.num_patches
        ax2.text(
            0.5,
            0.5,
            f"Vision Tokens\n({num_patches} patches)",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgreen"),
        )
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis("off")
        ax2.set_title("2. Image Tokenization", fontweight="bold")

        # Component 3: Language encoding
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.text(
            0.5,
            0.5,
            "Instruction\nTokens",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightyellow"),
        )
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis("off")
        ax3.set_title("3. Language Encoding", fontweight="bold")

        # Component 4: Multimodal tokens
        ax4 = fig.add_subplot(gs[1, :])
        ax4.text(
            0.5,
            0.5,
            "Multimodal Token Sequence\n[Vision Tokens | Language Tokens | Action Query]",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightcoral"),
        )
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis("off")
        ax4.set_title("4. Concatenated Tokens", fontweight="bold")

        # Component 5: Transformer
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.text(
            0.5,
            0.5,
            "Transformer\nEncoder\n(4 layers)",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="plum"),
        )
        ax5.set_xlim(0, 1)
        ax5.set_ylim(0, 1)
        ax5.axis("off")
        ax5.set_title("5. Self-Attention", fontweight="bold")

        # Component 6: Action token
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.text(
            0.5,
            0.5,
            "Action\nToken\nEmbedding",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgray"),
        )
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis("off")
        ax6.set_title("6. Extract Action", fontweight="bold")

        # Component 7: Action prediction
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.text(
            0.5,
            0.5,
            "Action\nLogits",
            ha="center",
            va="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="orange"),
        )
        ax7.set_xlim(0, 1)
        ax7.set_ylim(0, 1)
        ax7.axis("off")
        ax7.set_title("7. Action Prediction", fontweight="bold")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_training_curves(history: Dict, save_path: str = None):
        """Plot RT-1 training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Smooth curves
        def smooth(data, window=10):
            if len(data) < window:
                return data
            return np.convolve(data, np.ones(window) / window, mode="valid")

        # Returns
        axes[0].plot(
            history["episode"],
            smooth(history["return"]),
            color="#2ecc71",
            linewidth=2,
            alpha=0.8,
        )
        axes[0].set_xlabel("Episode", fontsize=12)
        axes[0].set_ylabel("Episode Return", fontsize=12)
        axes[0].set_title("RT-1 Training Performance", fontsize=14, fontweight="bold")
        axes[0].grid(True, alpha=0.3)

        # Loss
        axes[1].plot(
            history["episode"],
            smooth(history["loss"]),
            color="#e74c3c",
            linewidth=2,
            alpha=0.8,
        )
        axes[1].set_xlabel("Episode", fontsize=12)
        axes[1].set_ylabel("Loss", fontsize=12)
        axes[1].set_title("RT-1 Training Loss", fontsize=14, fontweight="bold")
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def visualize_tokenization(policy: RT1Policy, env, save_path: str = None):
        """Visualize image tokenization process."""
        obs, _ = env.reset()

        # Get visual tokens
        image_tensor = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            vision_tokens = policy.vision_tokenizer(image_tensor)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(obs)
        axes[0].set_title("Original Image (84×84×3)", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Patch grid
        patch_size = policy.vision_tokenizer.patch_size
        img_with_grid = obs.copy()
        for i in range(0, obs.shape[0], patch_size):
            img_with_grid[i, :] = 255
        for j in range(0, obs.shape[1], patch_size):
            img_with_grid[:, j] = 255

        axes[1].imshow(img_with_grid)
        num_patches = policy.vision_tokenizer.num_patches
        axes[1].set_title(
            f"Patch Grid ({num_patches} patches, {patch_size}×{patch_size})",
            fontsize=12,
            fontweight="bold",
        )
        axes[1].axis("off")

        # Token embeddings (show first 16 dimensions)
        tokens_np = vision_tokens[0].cpu().numpy()
        axes[2].imshow(tokens_np[:, :16].T, cmap="viridis", aspect="auto")
        axes[2].set_xlabel("Patch Index", fontsize=11)
        axes[2].set_ylabel("Embedding Dimension", fontsize=11)
        axes[2].set_title(
            f"Vision Token Embeddings ({tokens_np.shape[0]}×{tokens_np.shape[1]})",
            fontsize=12,
            fontweight="bold",
        )
        axes[2].colorbar()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def main():
    """Main RT-1 demonstration."""
    print("\n" + "=" * 80)
    print(" RT-1: ROBOTICS TRANSFORMER FOUNDATION MODEL ".center(80, "="))
    print("=" * 80 + "\n")

    # ==================== Configuration ====================
    ENV_NAME = "CartPole-v1"
    IMG_SIZE = 84
    PATCH_SIZE = 14
    EMBED_DIM = 256
    VOCAB_SIZE = 100
    NUM_ACTIONS = 2

    NUM_TRAIN_EPISODES = 100
    NUM_EVAL_EPISODES = 20
    LEARNING_RATE = 1e-3

    # Instructions for multi-task learning
    INSTRUCTIONS = ["balance pole upright", "keep cart stable", "stabilize system"]

    # ==================== Create Environment ====================
    print("Creating environment...")
    env = make_visual_env(ENV_NAME, IMG_SIZE)

    obs, _ = env.reset()
    print(f"Environment: {ENV_NAME}")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space.n} actions\n")

    # ==================== Create RT-1 Policy ====================
    print("Initializing RT-1 policy...")
    rt1_policy = RT1Policy(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_actions=NUM_ACTIONS,
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in rt1_policy.parameters())
    print(f"RT-1 policy parameters: {num_params:,}")
    print(f"Vision patches: {rt1_policy.vision_tokenizer.num_patches}")
    print(f"Embedding dimension: {EMBED_DIM}\n")

    # ==================== Create Instruction Encoder ====================
    instruction_encoder = SimpleInstructionEncoder(VOCAB_SIZE)

    # ==================== Visualize Architecture ====================
    print("\n" + "=" * 80)
    print(" VISUALIZING RT-1 ARCHITECTURE ".center(80))
    print("=" * 80 + "\n")

    visualizer = Visualizer()
    visualizer.visualize_architecture(rt1_policy, save_path="rt1_architecture.png")
    visualizer.visualize_tokenization(rt1_policy, env, save_path="rt1_tokenization.png")

    # ==================== Train RT-1 ====================
    print("\n" + "=" * 80)
    print(" TRAINING RT-1 POLICY ".center(80))
    print("=" * 80)

    trainer = RT1Trainer(env, rt1_policy, instruction_encoder, LEARNING_RATE)
    history = trainer.train(num_episodes=NUM_TRAIN_EPISODES, instructions=INSTRUCTIONS)

    # ==================== Evaluate RT-1 ====================
    print("\n" + "=" * 80)
    print(" EVALUATING RT-1 POLICY ".center(80))
    print("=" * 80 + "\n")

    # Evaluate on each instruction
    for instruction in INSTRUCTIONS:
        results = trainer.evaluate(
            num_episodes=NUM_EVAL_EPISODES, instruction=instruction
        )
        print(
            f"'{instruction}': {results['mean_return']:.2f} ± {results['std_return']:.2f}"
        )

    # ==================== Visualizations ====================
    print("\n" + "=" * 80)
    print(" GENERATING VISUALIZATIONS ".center(80))
    print("=" * 80 + "\n")

    print("Plotting training curves...")
    visualizer.plot_training_curves(history, save_path="rt1_training.png")

    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print(" FINAL SUMMARY ".center(80, "="))
    print("=" * 80 + "\n")

    print("RT-1 Foundation Model Concepts Demonstrated:")
    print("  1. Vision tokenization (image patches as tokens)")
    print("  2. Language conditioning (instruction encoding)")
    print("  3. Transformer architecture for multimodal input")
    print("  4. Action prediction via token generation")
    print("  5. Multi-task learning (multiple instructions)")
    print("  6. Scalable architecture design")
    print()

    print("Key RT-1 Architectural Components:")
    print("  • Image Tokenizer: Patches → Embeddings")
    print("  • Language Encoder: Text → Token Sequence")
    print("  • Transformer: Multimodal Self-Attention")
    print("  • Action Head: Token → Action Logits")
    print()

    print("Foundation Model Principles:")
    print("  - Large-scale pretraining on diverse data")
    print("  - Transformer architecture for flexibility")
    print("  - Multimodal understanding (vision + language)")
    print("  - Zero-shot / few-shot generalization")
    print("  - Transfer across tasks and embodiments")
    print()

    print("Scaling RT-1 to Real Foundation Model:")
    print("  • Data: Millions of trajectories across robots")
    print("  • Tasks: Hundreds of diverse manipulation tasks")
    print("  • Model: Larger transformers (100M+ parameters)")
    print("  • Language: Pretrained models (T5, BERT)")
    print("  • Vision: Pretrained encoders (ResNet, ViT)")
    print()

    print("RT-1 → RT-2 Evolution:")
    print("  RT-1: Robotics-specific training")
    print("  RT-2: Vision-language model (PaLI) + robotics")
    print("  Benefit: Internet-scale pretraining for robotics")
    print("  Result: Better generalization, reasoning, planning")
    print()

    print("The Future: Foundation Models for Robotics")
    print("  • Scaling: Bigger models, more data, more compute")
    print("  • Generalization: Cross-task, cross-embodiment")
    print("  • Reasoning: LLM integration for planning")
    print("  • Sim-to-real: Robust transfer via scale")
    print("  • Open research: Open X-Embodiment datasets")

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE - LECTURE 13 SERIES COMPLETE! ".center(80, "="))
    print("=" * 80 + "\n")

    print("🎉 Congratulations! You've completed all demos:")
    print("  ✓ Behavioral Cloning")
    print("  ✓ PPO (Visual Control)")
    print("  ✓ SAC (Pixels)")
    print("  ✓ DrQ (Augmentation)")
    print("  ✓ GAIL")
    print("  ✓ DAgger")
    print("  ✓ CURL (Contrastive Learning)")
    print("  ✓ RAD (Augmentation Strategies)")
    print("  ✓ Domain Randomization")
    print("  ✓ RT-1 (Foundation Models)")

    print("\n" + "=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    main()
