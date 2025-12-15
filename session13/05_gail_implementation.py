"""
Generative Adversarial Imitation Learning (GAIL)
================================================

This demo illustrates GAIL algorithm for learning from demonstrations:
1. Discriminator network to distinguish expert from learner
2. Policy network trained to fool the discriminator
3. Adversarial training framework (GAN-style)
4. Comparison with behavioral cloning
5. Trajectory distribution matching

Lecture 13: Visual Policy Learning - From Imitation to Foundation Models
Prof. David Olivieri - UVigo - VIAR25/26

Key Concepts Demonstrated:
--------------------------
- Adversarial imitation learning
- Implicit reward learning via discriminator
- Distribution matching vs supervised learning
- GAN-style training for imitation
- Overcoming limitations of behavioral cloning

Reference: Ho & Ermon, "Generative Adversarial Imitation Learning" (NeurIPS 2016)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
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


class ExpertPolicy:
    """
    Expert policy for generating demonstrations.
    Uses a heuristic for CartPole.
    """

    def __init__(self, env):
        self.env = env

    def get_action(self, state: np.ndarray) -> int:
        """Expert heuristic policy."""
        _, cart_velocity, pole_angle, pole_velocity = state

        angle_threshold = 0.05
        velocity_weight = 0.3

        weighted_angle = pole_angle + velocity_weight * pole_velocity

        if weighted_angle > angle_threshold:
            return 1  # Push right
        else:
            return 0  # Push left


class ExpertDataCollector:
    """
    Collects expert demonstrations for imitation learning.
    """

    def __init__(self, env, expert_policy):
        self.env = env
        self.expert = expert_policy

    def collect_trajectories(
        self, num_episodes: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Collect expert demonstrations.

        Args:
            num_episodes: Number of episodes to collect

        Returns:
            states: Expert states
            actions: Expert actions
            trajectories: List of (states, actions) per episode
        """
        states_list = []
        actions_list = []
        trajectories = []
        episode_returns = []

        print(f"\n{'='*60}")
        print(f"Collecting {num_episodes} expert demonstrations...")
        print(f"{'='*60}")

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_return = 0
            episode_states = []
            episode_actions = []
            done = False

            while not done:
                action = self.expert.get_action(state)
                episode_states.append(state)
                episode_actions.append(action)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_return += reward
                state = next_state

            states_list.extend(episode_states)
            actions_list.extend(episode_actions)
            trajectories.append((np.array(episode_states), np.array(episode_actions)))
            episode_returns.append(episode_return)

            if (episode + 1) % 20 == 0:
                avg_return = np.mean(episode_returns[-20:])
                print(
                    f"Episode {episode + 1}/{num_episodes} | Avg Return: {avg_return:.2f}"
                )

        states = np.array(states_list)
        actions = np.array(actions_list)

        print(f"\nCollected {len(states)} state-action pairs")
        print(
            f"Expert average return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}"
        )
        print(f"{'='*60}\n")

        return states, actions, trajectories


class Discriminator(nn.Module):
    """
    Discriminator network for GAIL.
    Distinguishes between expert and learner trajectories.

    Output interpretation:
    - D(s,a) ≈ 1: Expert-like behavior
    - D(s,a) ≈ 0: Learner behavior
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]
    ):
        super(Discriminator, self).__init__()

        # For discrete actions, we'll use one-hot encoding
        self.action_dim = action_dim
        input_dim = state_dim + action_dim

        # Build MLP layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim

        # Output layer: probability of being expert
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through discriminator.

        Args:
            state: State tensor [batch_size, state_dim]
            action: Action tensor [batch_size] (discrete actions)

        Returns:
            prob: Probability of being expert [batch_size, 1]
        """
        # One-hot encode actions
        action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()

        # Concatenate state and action
        sa_pair = torch.cat([state, action_one_hot], dim=-1)

        # Forward pass
        prob = self.network(sa_pair)

        return prob


class GAILPolicy(nn.Module):
    """
    Policy network for GAIL (generator).
    Learns to generate expert-like behavior.
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]
    ):
        super(GAILPolicy, self).__init__()

        # Build MLP layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.1))
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through policy network.

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            logits: Action logits [batch_size, action_dim]
        """
        return self.network(state)

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float]:
        """
        Sample action from policy.

        Args:
            state: Current state
            deterministic: If True, select argmax action

        Returns:
            action: Selected action
            log_prob: Log probability of action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits = self.forward(state_tensor)
            probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = torch.argmax(probs, dim=-1).item()
            else:
                action = torch.multinomial(probs, 1).item()

            log_probs = F.log_softmax(logits, dim=-1)
            action_log_prob = log_probs[0, action].item()

        return action, action_log_prob

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        """
        Evaluate actions for policy update.

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            log_probs: Log probabilities of actions
            entropy: Policy entropy
        """
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        action_log_probs = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(-1)

        return action_log_probs, entropy


class GAILBuffer:
    """
    Replay buffer for GAIL training.
    Stores learner trajectories for policy updates.
    """

    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, batch_size)

        return (
            torch.FloatTensor(self.states[indices]).to(device),
            torch.LongTensor(self.actions[indices]).to(device),
            torch.FloatTensor(self.rewards[indices]).to(device),
            torch.FloatTensor(self.next_states[indices]).to(device),
            torch.FloatTensor(self.dones[indices]).to(device),
        )

    def get_all(self):
        """Get all data from buffer."""
        return (
            torch.FloatTensor(self.states[: self.size]).to(device),
            torch.LongTensor(self.actions[: self.size]).to(device),
            torch.FloatTensor(self.rewards[: self.size]).to(device),
            torch.FloatTensor(self.next_states[: self.size]).to(device),
            torch.FloatTensor(self.dones[: self.size]).to(device),
        )

    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class GAILTrainer:
    """
    GAIL trainer implementing adversarial imitation learning.
    """

    def __init__(
        self,
        policy: GAILPolicy,
        discriminator: Discriminator,
        expert_states: np.ndarray,
        expert_actions: np.ndarray,
        policy_lr: float = 3e-4,
        discriminator_lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
    ):
        self.policy = policy
        self.discriminator = discriminator

        # Expert data
        self.expert_states = torch.FloatTensor(expert_states).to(device)
        self.expert_actions = torch.LongTensor(expert_actions).to(device)

        # Optimizers
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=policy_lr)
        self.discriminator_optimizer = optim.Adam(
            discriminator.parameters(), lr=discriminator_lr
        )

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef

    def compute_gae(self, rewards: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: Rewards from discriminator
            dones: Done flags

        Returns:
            advantages: GAE advantages
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = rewards[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - rewards[t]
            advantages[t] = last_gae = (
                delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            )

        return advantages

    def update_discriminator(
        self,
        learner_states: torch.Tensor,
        learner_actions: torch.Tensor,
        num_epochs: int = 3,
        batch_size: int = 64,
    ) -> Dict:
        """
        Update discriminator to distinguish expert from learner.

        Args:
            learner_states: States from learner
            learner_actions: Actions from learner
            num_epochs: Number of training epochs
            batch_size: Mini-batch size

        Returns:
            stats: Training statistics
        """
        stats = {"discriminator_loss": [], "expert_acc": [], "learner_acc": []}

        num_expert = len(self.expert_states)
        num_learner = len(learner_states)

        for epoch in range(num_epochs):
            # Shuffle data
            expert_indices = torch.randperm(num_expert)
            learner_indices = torch.randperm(num_learner)

            # Number of batches
            num_batches = min(num_expert, num_learner) // batch_size

            for i in range(num_batches):
                # Sample expert batch
                expert_idx = expert_indices[i * batch_size : (i + 1) * batch_size]
                expert_s = self.expert_states[expert_idx]
                expert_a = self.expert_actions[expert_idx]

                # Sample learner batch
                learner_idx = learner_indices[i * batch_size : (i + 1) * batch_size]
                learner_s = learner_states[learner_idx]
                learner_a = learner_actions[learner_idx]

                # Discriminator predictions
                expert_pred = self.discriminator(expert_s, expert_a)
                learner_pred = self.discriminator(learner_s, learner_a)

                # Binary cross-entropy loss
                # Expert should be classified as 1, learner as 0
                expert_loss = F.binary_cross_entropy(
                    expert_pred, torch.ones_like(expert_pred)
                )
                learner_loss = F.binary_cross_entropy(
                    learner_pred, torch.zeros_like(learner_pred)
                )

                discriminator_loss = expert_loss + learner_loss

                # Update discriminator
                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                self.discriminator_optimizer.step()

                # Compute accuracies
                expert_acc = (expert_pred > 0.5).float().mean().item()
                learner_acc = (learner_pred < 0.5).float().mean().item()

                stats["discriminator_loss"].append(discriminator_loss.item())
                stats["expert_acc"].append(expert_acc)
                stats["learner_acc"].append(learner_acc)

        # Average statistics
        for key in stats:
            stats[key] = np.mean(stats[key]) if len(stats[key]) > 0 else 0.0

        return stats

    def update_policy(
        self, learner_buffer: GAILBuffer, num_epochs: int = 4, batch_size: int = 64
    ) -> Dict:
        """
        Update policy using discriminator rewards.

        Args:
            learner_buffer: Buffer with learner trajectories
            num_epochs: Number of training epochs
            batch_size: Mini-batch size

        Returns:
            stats: Training statistics
        """
        # Get all data from buffer
        states, actions, _, next_states, dones = learner_buffer.get_all()

        # Compute rewards from discriminator
        with torch.no_grad():
            discriminator_output = self.discriminator(states, actions)
            # GAIL reward: log(D(s,a))
            rewards = torch.log(discriminator_output + 1e-8).squeeze(-1)

        # Compute advantages
        advantages = self.compute_gae(rewards, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = {"policy_loss": [], "entropy": []}

        # Multiple epochs of policy optimization
        for epoch in range(num_epochs):
            indices = torch.randperm(len(states))

            for start_idx in range(0, len(states), batch_size):
                end_idx = start_idx + batch_size
                batch_indices = indices[start_idx:end_idx]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Evaluate actions
                log_probs, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                # Policy gradient loss
                policy_loss = -(log_probs * batch_advantages).mean()

                # Entropy bonus for exploration
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.entropy_coef * entropy_loss

                # Update policy
                self.policy_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
                self.policy_optimizer.step()

                stats["policy_loss"].append(policy_loss.item())
                stats["entropy"].append(entropy.mean().item())

        # Average statistics
        for key in stats:
            stats[key] = np.mean(stats[key]) if len(stats[key]) > 0 else 0.0

        return stats


class GAILAgent:
    """
    GAIL agent for training and evaluation.
    """

    def __init__(
        self,
        env,
        policy: GAILPolicy,
        trainer: GAILTrainer,
        buffer_capacity: int = 10000,
    ):
        self.env = env
        self.policy = policy
        self.trainer = trainer

        # Create buffer
        state_dim = env.observation_space.shape[0]
        self.buffer = GAILBuffer(buffer_capacity, state_dim)

        # Statistics
        self.episode_returns = []

    def collect_trajectories(self, num_steps: int = 2048) -> int:
        """
        Collect trajectories using current policy.

        Args:
            num_steps: Number of steps to collect

        Returns:
            num_episodes: Number of complete episodes
        """
        self.buffer.clear()

        state, _ = self.env.reset()
        episode_return = 0
        num_episodes = 0

        for step in range(num_steps):
            # Get action from policy
            action, _ = self.policy.get_action(state, deterministic=False)

            # Step environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Store transition (reward will be overwritten by discriminator)
            self.buffer.add(state, action, reward, next_state, done)

            episode_return += reward

            if done:
                self.episode_returns.append(episode_return)
                num_episodes += 1
                state, _ = self.env.reset()
                episode_return = 0
            else:
                state = next_state

        return num_episodes

    def train(
        self,
        num_iterations: int = 100,
        steps_per_iter: int = 2048,
        discriminator_epochs: int = 3,
        policy_epochs: int = 4,
    ) -> Dict:
        """
        Train GAIL agent.

        Args:
            num_iterations: Number of training iterations
            steps_per_iter: Steps to collect per iteration
            discriminator_epochs: Epochs to train discriminator
            policy_epochs: Epochs to train policy

        Returns:
            history: Training history
        """
        print(f"\n{'='*80}")
        print(f"Starting GAIL Training")
        print(f"{'='*80}")
        print(f"Iterations: {num_iterations}")
        print(f"Steps per iteration: {steps_per_iter}")
        print(f"{'='*80}\n")

        history = {
            "iteration": [],
            "mean_return": [],
            "discriminator_loss": [],
            "policy_loss": [],
            "expert_acc": [],
            "learner_acc": [],
        }

        for iteration in range(num_iterations):
            # Collect trajectories with current policy
            num_episodes = self.collect_trajectories(steps_per_iter)

            # Get learner data
            learner_states, learner_actions, _, _, _ = self.buffer.get_all()

            # Update discriminator
            disc_stats = self.trainer.update_discriminator(
                learner_states, learner_actions, num_epochs=discriminator_epochs
            )

            # Update policy
            policy_stats = self.trainer.update_policy(
                self.buffer, num_epochs=policy_epochs
            )

            # Record statistics
            if len(self.episode_returns) > 0:
                recent_returns = (
                    self.episode_returns[-10:]
                    if len(self.episode_returns) >= 10
                    else self.episode_returns
                )
                mean_return = np.mean(recent_returns)

                history["iteration"].append(iteration)
                history["mean_return"].append(mean_return)
                history["discriminator_loss"].append(disc_stats["discriminator_loss"])
                history["policy_loss"].append(policy_stats["policy_loss"])
                history["expert_acc"].append(disc_stats["expert_acc"])
                history["learner_acc"].append(disc_stats["learner_acc"])

                if (iteration + 1) % 10 == 0:
                    print(
                        f"Iter {iteration + 1:3d}/{num_iterations} | "
                        f"Return: {mean_return:7.2f} | "
                        f"Disc Loss: {disc_stats['discriminator_loss']:.4f} | "
                        f"Policy Loss: {policy_stats['policy_loss']:.4f} | "
                        f"Expert Acc: {disc_stats['expert_acc']:.3f} | "
                        f"Learner Acc: {disc_stats['learner_acc']:.3f}"
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
            state, _ = self.env.reset()
            episode_return = 0
            episode_length = 0
            done = False

            while not done:
                action, _ = self.policy.get_action(state, deterministic)
                state, reward, terminated, truncated, _ = self.env.step(action)
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
    Visualization utilities for GAIL analysis.
    """

    @staticmethod
    def plot_training_curves(history: Dict, save_path: str = None):
        """Plot GAIL training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()

        metrics = [
            ("mean_return", "Episode Return", "#2ecc71"),
            ("discriminator_loss", "Discriminator Loss", "#e74c3c"),
            ("policy_loss", "Policy Loss", "#3498db"),
            ("expert_acc", "Expert Accuracy", "#9b59b6"),
            ("learner_acc", "Learner Accuracy", "#f39c12"),
        ]

        for idx, (key, title, color) in enumerate(metrics):
            if key in history and len(history[key]) > 0:
                axes[idx].plot(
                    history["iteration"],
                    history[key],
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                )
                axes[idx].set_xlabel("Iteration", fontsize=11)
                axes[idx].set_ylabel(title, fontsize=11)
                axes[idx].set_title(title, fontsize=12, fontweight="bold")
                axes[idx].grid(True, alpha=0.3)

        # Combined accuracy plot
        if "expert_acc" in history and "learner_acc" in history:
            axes[5].plot(
                history["iteration"],
                history["expert_acc"],
                label="Expert Accuracy",
                color="#9b59b6",
                linewidth=2,
            )
            axes[5].plot(
                history["iteration"],
                history["learner_acc"],
                label="Learner Accuracy",
                color="#f39c12",
                linewidth=2,
            )
            axes[5].axhline(
                y=0.5, color="red", linestyle="--", alpha=0.5, label="Random"
            )
            axes[5].set_xlabel("Iteration", fontsize=11)
            axes[5].set_ylabel("Accuracy", fontsize=11)
            axes[5].set_title("Discriminator Accuracy", fontsize=12, fontweight="bold")
            axes[5].legend(fontsize=10)
            axes[5].grid(True, alpha=0.3)

        plt.suptitle("GAIL Training Curves", fontsize=15, fontweight="bold", y=0.995)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_comparison(
        gail_results: Dict,
        bc_results: Dict,
        expert_results: Dict,
        random_results: Dict,
        save_path: str = None,
    ):
        """Compare GAIL vs BC vs Expert vs Random."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar plot
        policies = ["Expert", "GAIL", "BC", "Random"]
        means = [
            expert_results["mean_return"],
            gail_results["mean_return"],
            bc_results["mean_return"],
            random_results["mean_return"],
        ]
        stds = [
            expert_results["std_return"],
            gail_results["std_return"],
            bc_results["std_return"],
            random_results["std_return"],
        ]
        colors = ["#2ecc71", "#9b59b6", "#3498db", "#e74c3c"]

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
            expert_results["returns"],
            bins=15,
            alpha=0.5,
            label="Expert",
            color="#2ecc71",
            edgecolor="black",
        )
        axes[1].hist(
            gail_results["returns"],
            bins=15,
            alpha=0.5,
            label="GAIL",
            color="#9b59b6",
            edgecolor="black",
        )
        axes[1].hist(
            bc_results["returns"],
            bins=15,
            alpha=0.5,
            label="BC",
            color="#3498db",
            edgecolor="black",
        )
        axes[1].hist(
            random_results["returns"],
            bins=15,
            alpha=0.5,
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
    Main GAIL demonstration.
    """
    print("\n" + "=" * 80)
    print(" GENERATIVE ADVERSARIAL IMITATION LEARNING (GAIL) ".center(80, "="))
    print("=" * 80 + "\n")

    # ==================== Configuration ====================
    ENV_NAME = "CartPole-v1"
    NUM_EXPERT_EPISODES = 100
    NUM_ITERATIONS = 100
    STEPS_PER_ITER = 2048
    DISCRIMINATOR_EPOCHS = 3
    POLICY_EPOCHS = 4
    BATCH_SIZE = 64
    NUM_EVAL_EPISODES = 50

    # ==================== Setup Environment ====================
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: {ENV_NAME}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}\n")

    # ==================== Collect Expert Demonstrations ====================
    print("\n" + "=" * 80)
    print(" STEP 1: COLLECTING EXPERT DEMONSTRATIONS ".center(80))
    print("=" * 80)

    expert = ExpertPolicy(env)
    collector = ExpertDataCollector(env, expert)
    expert_states, expert_actions, expert_trajectories = collector.collect_trajectories(
        num_episodes=NUM_EXPERT_EPISODES
    )

    # ==================== Train GAIL ====================
    print("\n" + "=" * 80)
    print(" STEP 2: TRAINING GAIL ".center(80))
    print("=" * 80)

    # Create networks
    gail_policy = GAILPolicy(state_dim, action_dim).to(device)
    discriminator = Discriminator(state_dim, action_dim).to(device)

    # Create trainer
    gail_trainer = GAILTrainer(
        gail_policy,
        discriminator,
        expert_states,
        expert_actions,
        policy_lr=3e-4,
        discriminator_lr=3e-4,
    )

    # Create agent and train
    gail_agent = GAILAgent(env, gail_policy, gail_trainer)
    gail_history = gail_agent.train(
        num_iterations=NUM_ITERATIONS,
        steps_per_iter=STEPS_PER_ITER,
        discriminator_epochs=DISCRIMINATOR_EPOCHS,
        policy_epochs=POLICY_EPOCHS,
    )

    # ==================== Train BC for Comparison ====================
    print("\n" + "=" * 80)
    print(" STEP 3: TRAINING BC FOR COMPARISON ".center(80))
    print("=" * 80 + "\n")

    # Simple BC policy
    from behavioral_cloning import BCPolicy, BCTrainer

    bc_policy = BCPolicy(state_dim, action_dim).to(device)
    bc_trainer = BCTrainer(bc_policy, learning_rate=1e-3)
    bc_history = bc_trainer.train(
        expert_states, expert_actions, num_epochs=100, batch_size=64
    )

    # ==================== Evaluate Policies ====================
    print("\n" + "=" * 80)
    print(" STEP 4: EVALUATING POLICIES ".center(80))
    print("=" * 80 + "\n")

    # Evaluate GAIL
    gail_results = gail_agent.evaluate(num_episodes=NUM_EVAL_EPISODES)
    print(
        f"GAIL   - Mean Return: {gail_results['mean_return']:.2f} ± {gail_results['std_return']:.2f}"
    )

    # Evaluate BC
    class BCEvaluator:
        def __init__(self, env, policy):
            self.env = env
            self.policy = policy

        def evaluate(self, num_episodes):
            returns = []
            for _ in range(num_episodes):
                state, _ = self.env.reset()
                episode_return = 0
                done = False
                while not done:
                    action = self.policy.get_action(state, deterministic=True)
                    state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    episode_return += reward
                returns.append(episode_return)
            return {
                "mean_return": np.mean(returns),
                "std_return": np.std(returns),
                "returns": returns,
            }

    bc_evaluator = BCEvaluator(env, bc_policy)
    bc_results = bc_evaluator.evaluate(NUM_EVAL_EPISODES)
    print(
        f"BC     - Mean Return: {bc_results['mean_return']:.2f} ± {bc_results['std_return']:.2f}"
    )

    # Evaluate Expert
    expert_evaluator = BCEvaluator(env, expert)
    expert_results = expert_evaluator.evaluate(NUM_EVAL_EPISODES)
    print(
        f"Expert - Mean Return: {expert_results['mean_return']:.2f} ± {expert_results['std_return']:.2f}"
    )

    # Evaluate Random
    random_returns = []
    for _ in range(NUM_EVAL_EPISODES):
        state, _ = env.reset()
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
        f"Random - Mean Return: {random_results['mean_return']:.2f} ± {random_results['std_return']:.2f}"
    )

    # ==================== Visualizations ====================
    print("\n" + "=" * 80)
    print(" STEP 5: GENERATING VISUALIZATIONS ".center(80))
    print("=" * 80 + "\n")

    visualizer = Visualizer()

    print("Plotting GAIL training curves...")
    visualizer.plot_training_curves(gail_history, save_path="gail_training_curves.png")

    print("Plotting performance comparison...")
    visualizer.plot_comparison(
        gail_results,
        bc_results,
        expert_results,
        random_results,
        save_path="gail_performance_comparison.png",
    )

    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print(" FINAL SUMMARY ".center(80, "="))
    print("=" * 80 + "\n")

    print("Performance Results:")
    print(
        f"  - Expert: {expert_results['mean_return']:6.2f} ± {expert_results['std_return']:5.2f}"
    )
    print(
        f"  - GAIL:   {gail_results['mean_return']:6.2f} ± {gail_results['std_return']:5.2f}"
    )
    print(
        f"  - BC:     {bc_results['mean_return']:6.2f} ± {bc_results['std_return']:5.2f}"
    )
    print(
        f"  - Random: {random_results['mean_return']:6.2f} ± {random_results['std_return']:5.2f}"
    )
    print()

    gail_vs_bc = (
        (gail_results["mean_return"] - bc_results["mean_return"])
        / bc_results["mean_return"]
        * 100
    )
    print(f"GAIL improvement over BC: {gail_vs_bc:.1f}%")
    print()

    print("Key GAIL Mechanisms Demonstrated:")
    print("  1. Adversarial training: discriminator vs policy")
    print("  2. Implicit reward learning from demonstrations")
    print("  3. Distribution matching instead of supervised learning")
    print("  4. Overcomes BC's distribution shift problem")
    print("  5. More robust to compounding errors")
    print()

    print("GAIL vs BC Comparison:")
    print("  - GAIL: Learns from state-action distributions")
    print("  - BC: Learns from individual state-action pairs")
    print("  - GAIL: More sample efficient with limited demos")
    print("  - BC: Simpler but suffers from distribution shift")
    print("  - GAIL: Requires RL optimization (slower)")
    print("  - BC: Fast supervised learning (but less robust)")

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE ".center(80, "="))
    print("=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    main()
