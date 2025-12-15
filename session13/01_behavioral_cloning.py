"""
Behavioral Cloning (BC) Demo
============================

This demo illustrates the core concepts of imitation learning through behavioral cloning:
1. Expert trajectory collection
2. Supervised learning of policy from demonstrations
3. Performance comparison: BC vs Expert vs Random
4. Analysis of distribution shift (covariate shift problem)

Lecture 13: Visual Policy Learning - From Imitation to Foundation Models
Prof. David Olivieri - UVigo - VIAR25/26

Key Concepts Demonstrated:
--------------------------
- Behavioral Cloning as supervised learning
- Expert demonstrations as training data
- Distribution mismatch between training and deployment
- Compounding errors in sequential decision making
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
from typing import List, Tuple, Dict
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
    Simple expert policy for CartPole environment.
    Uses a basic heuristic: if pole is falling right, push right; if falling left, push left.
    This is not optimal but provides reasonable demonstrations for BC.
    """

    def __init__(self, env):
        self.env = env

    def get_action(self, state: np.ndarray) -> int:
        """
        Heuristic expert policy for CartPole.

        Args:
            state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        Returns:
            action: 0 (left) or 1 (right)
        """
        # Extract relevant features
        _, cart_velocity, pole_angle, pole_velocity = state

        # Weighted combination of angle and angular velocity
        # This heuristic tries to keep the pole balanced
        angle_threshold = 0.05
        velocity_weight = 0.3

        weighted_angle = pole_angle + velocity_weight * pole_velocity

        if weighted_angle > angle_threshold:
            return 1  # Push right
        else:
            return 0  # Push left


class BCPolicy(nn.Module):
    """
    Behavioral Cloning Policy Network.
    Simple MLP that learns to map states to actions via supervised learning.
    """

    def __init__(
        self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]
    ):
        super(BCPolicy, self).__init__()

        # Build MLP layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Small dropout for regularization
            input_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Xavier initialization for better training stability."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Batch of states [batch_size, state_dim]

        Returns:
            logits: Action logits [batch_size, action_dim]
        """
        return self.network(state)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Select action given a state.

        Args:
            state: Current state
            deterministic: If True, select argmax; if False, sample from distribution

        Returns:
            action: Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            logits = self.forward(state_tensor)

            if deterministic:
                action = logits.argmax(dim=1).item()
            else:
                probs = F.softmax(logits, dim=1)
                action = torch.multinomial(probs, 1).item()

        return action


class ExpertDataCollector:
    """
    Collects expert demonstrations for imitation learning.
    """

    def __init__(self, env, expert_policy):
        self.env = env
        self.expert = expert_policy

    def collect_trajectories(
        self, num_episodes: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect expert demonstrations.

        Args:
            num_episodes: Number of episodes to collect

        Returns:
            states: Array of states [N, state_dim]
            actions: Array of actions [N]
        """
        states_list = []
        actions_list = []
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
            episode_returns.append(episode_return)

            if (episode + 1) % 20 == 0:
                avg_return = np.mean(episode_returns[-20:])
                print(
                    f"Episode {episode + 1}/{num_episodes} | Avg Return (last 20): {avg_return:.2f}"
                )

        states = np.array(states_list)
        actions = np.array(actions_list)

        print(f"\nCollected {len(states)} state-action pairs")
        print(
            f"Expert average return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}"
        )
        print(f"{'='*60}\n")

        return states, actions, episode_returns


class BCTrainer:
    """
    Trains a behavioral cloning policy using supervised learning.
    """

    def __init__(self, policy: BCPolicy, learning_rate: float = 1e-3):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        num_epochs: int = 100,
        batch_size: int = 64,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Train the BC policy using supervised learning.

        Args:
            states: Expert states
            actions: Expert actions
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation

        Returns:
            history: Dictionary with training metrics
        """
        # Split into train/validation
        num_samples = len(states)
        indices = np.random.permutation(num_samples)
        split_idx = int(num_samples * (1 - validation_split))

        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        train_states = torch.FloatTensor(states[train_indices]).to(device)
        train_actions = torch.LongTensor(actions[train_indices]).to(device)
        val_states = torch.FloatTensor(states[val_indices]).to(device)
        val_actions = torch.LongTensor(actions[val_indices]).to(device)

        print(f"\n{'='*60}")
        print(f"Training Behavioral Cloning Policy")
        print(f"{'='*60}")
        print(f"Training samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"{'='*60}\n")

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(num_epochs):
            # Training phase
            self.policy.train()
            train_losses = []
            train_accs = []

            # Mini-batch training
            num_batches = len(train_states) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size

                batch_states = train_states[start_idx:end_idx]
                batch_actions = train_actions[start_idx:end_idx]

                # Forward pass
                logits = self.policy(batch_states)
                loss = self.loss_fn(logits, batch_actions)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    predictions = logits.argmax(dim=1)
                    accuracy = (predictions == batch_actions).float().mean()

                train_losses.append(loss.item())
                train_accs.append(accuracy.item())

            # Validation phase
            self.policy.eval()
            with torch.no_grad():
                val_logits = self.policy(val_states)
                val_loss = self.loss_fn(val_logits, val_actions)
                val_predictions = val_logits.argmax(dim=1)
                val_accuracy = (val_predictions == val_actions).float().mean()

            # Record history
            history["train_loss"].append(np.mean(train_losses))
            history["train_acc"].append(np.mean(train_accs))
            history["val_loss"].append(val_loss.item())
            history["val_acc"].append(val_accuracy.item())

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:3d}/{num_epochs} | "
                    f"Train Loss: {history['train_loss'][-1]:.4f} | "
                    f"Train Acc: {history['train_acc'][-1]:.4f} | "
                    f"Val Loss: {history['val_loss'][-1]:.4f} | "
                    f"Val Acc: {history['val_acc'][-1]:.4f}"
                )

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Final validation accuracy: {history['val_acc'][-1]:.4f}")
        print(f"{'='*60}\n")

        return history


class PolicyEvaluator:
    """
    Evaluates different policies and compares their performance.
    """

    def __init__(self, env):
        self.env = env

    def evaluate_policy(
        self,
        policy,
        num_episodes: int = 50,
        policy_name: str = "Policy",
        render: bool = False,
    ) -> Dict:
        """
        Evaluate a policy over multiple episodes.

        Args:
            policy: Policy to evaluate (must have get_action method)
            num_episodes: Number of evaluation episodes
            policy_name: Name for logging
            render: Whether to render the environment

        Returns:
            results: Dictionary with evaluation metrics
        """
        returns = []
        episode_lengths = []

        print(f"\nEvaluating {policy_name} for {num_episodes} episodes...")

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_return = 0
            episode_length = 0
            done = False

            while not done:
                if render and episode == 0:  # Render first episode
                    self.env.render()

                action = policy.get_action(state, deterministic=True)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_return += reward
                episode_length += 1
                state = next_state

            returns.append(episode_return)
            episode_lengths.append(episode_length)

        results = {
            "returns": returns,
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "mean_length": np.mean(episode_lengths),
            "name": policy_name,
        }

        print(
            f"{policy_name} - Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}"
        )
        print(f"{policy_name} - Mean Episode Length: {results['mean_length']:.2f}")

        return results

    def evaluate_random_policy(self, num_episodes: int = 50) -> Dict:
        """Evaluate random policy baseline."""

        class RandomPolicy:
            def __init__(self, env):
                self.env = env

            def get_action(self, state, deterministic=False):
                return self.env.action_space.sample()

        random_policy = RandomPolicy(self.env)
        return self.evaluate_policy(random_policy, num_episodes, "Random Policy")


class DistributionShiftAnalyzer:
    """
    Analyzes the distribution shift problem in behavioral cloning.
    Demonstrates how BC policy can drift from expert demonstrations.
    """

    def __init__(self, env, expert_policy, bc_policy):
        self.env = env
        self.expert = expert_policy
        self.bc_policy = bc_policy

    def analyze_state_distribution(self, num_episodes: int = 10) -> Dict:
        """
        Compare state distributions visited by expert vs BC policy.
        This reveals the compounding error problem.

        Args:
            num_episodes: Number of episodes to collect for each policy

        Returns:
            analysis: Dictionary with state distributions
        """
        print(f"\n{'='*60}")
        print("Analyzing Distribution Shift (Covariate Shift)")
        print(f"{'='*60}\n")

        # Collect states from expert
        expert_states = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                expert_states.append(state)
                action = self.expert.get_action(state)
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

        # Collect states from BC policy
        bc_states = []
        for _ in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                bc_states.append(state)
                action = self.bc_policy.get_action(state, deterministic=True)
                state, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

        expert_states = np.array(expert_states)
        bc_states = np.array(bc_states)

        print(f"Expert visited {len(expert_states)} states")
        print(f"BC policy visited {len(bc_states)} states")

        # Compute statistics
        analysis = {
            "expert_states": expert_states,
            "bc_states": bc_states,
            "expert_mean": expert_states.mean(axis=0),
            "expert_std": expert_states.std(axis=0),
            "bc_mean": bc_states.mean(axis=0),
            "bc_std": bc_states.std(axis=0),
        }

        # Print comparison
        state_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Velocity"]
        print("\nState Distribution Comparison:")
        print("-" * 60)
        for i, name in enumerate(state_names):
            print(
                f"{name:20s} | Expert: {analysis['expert_mean'][i]:7.3f} ± {analysis['expert_std'][i]:6.3f} | "
                f"BC: {analysis['bc_mean'][i]:7.3f} ± {analysis['bc_std'][i]:6.3f}"
            )

        print(f"{'='*60}\n")

        return analysis


class Visualizer:
    """
    Visualization utilities for BC analysis.
    """

    @staticmethod
    def plot_training_curves(history: Dict, save_path: str = None):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        axes[0].plot(history["train_loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_xlabel("Epoch", fontsize=12)
        axes[0].set_ylabel("Cross-Entropy Loss", fontsize=12)
        axes[0].set_title(
            "Training and Validation Loss", fontsize=14, fontweight="bold"
        )
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Accuracy curves
        axes[1].plot(history["train_acc"], label="Train Accuracy", linewidth=2)
        axes[1].plot(history["val_acc"], label="Val Accuracy", linewidth=2)
        axes[1].set_xlabel("Epoch", fontsize=12)
        axes[1].set_ylabel("Accuracy", fontsize=12)
        axes[1].set_title(
            "Training and Validation Accuracy", fontsize=14, fontweight="bold"
        )
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    @staticmethod
    def plot_performance_comparison(
        results_dict: Dict[str, Dict], save_path: str = None
    ):
        """Compare performance of different policies."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        policy_names = list(results_dict.keys())
        means = [results_dict[name]["mean_return"] for name in policy_names]
        stds = [results_dict[name]["std_return"] for name in policy_names]

        # Bar plot with error bars
        colors = ["#2ecc71", "#3498db", "#e74c3c"]
        x_pos = np.arange(len(policy_names))

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
        axes[0].set_xticklabels(policy_names, fontsize=11)
        axes[0].set_ylabel("Average Return", fontsize=12)
        axes[0].set_title(
            "Policy Performance Comparison", fontsize=14, fontweight="bold"
        )
        axes[0].grid(True, alpha=0.3, axis="y")

        # Distribution plot
        for name, color in zip(policy_names, colors):
            returns = results_dict[name]["returns"]
            axes[1].hist(
                returns, bins=20, alpha=0.5, label=name, color=color, edgecolor="black"
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

    @staticmethod
    def plot_distribution_shift(analysis: Dict, save_path: str = None):
        """Visualize the distribution shift between expert and BC policy."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()

        state_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Velocity"]

        for i, name in enumerate(state_names):
            # Histogram comparison
            axes[i].hist(
                analysis["expert_states"][:, i],
                bins=50,
                alpha=0.6,
                label="Expert",
                color="#2ecc71",
                density=True,
                edgecolor="black",
            )
            axes[i].hist(
                analysis["bc_states"][:, i],
                bins=50,
                alpha=0.6,
                label="BC Policy",
                color="#3498db",
                density=True,
                edgecolor="black",
            )

            # Add mean lines
            axes[i].axvline(
                analysis["expert_mean"][i],
                color="#27ae60",
                linestyle="--",
                linewidth=2,
                label="Expert Mean",
            )
            axes[i].axvline(
                analysis["bc_mean"][i],
                color="#2980b9",
                linestyle="--",
                linewidth=2,
                label="BC Mean",
            )

            axes[i].set_xlabel(name, fontsize=11)
            axes[i].set_ylabel("Density", fontsize=11)
            axes[i].set_title(f"{name} Distribution", fontsize=12, fontweight="bold")
            axes[i].legend(fontsize=9)
            axes[i].grid(True, alpha=0.3, axis="y")

        plt.suptitle(
            "Distribution Shift Analysis: Expert vs BC Policy",
            fontsize=15,
            fontweight="bold",
            y=1.00,
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


def main():
    """
    Main demonstration of behavioral cloning.
    """
    print("\n" + "=" * 80)
    print(" BEHAVIORAL CLONING DEMONSTRATION ".center(80, "="))
    print("=" * 80 + "\n")

    # ==================== Configuration ====================
    ENV_NAME = "CartPole-v1"
    NUM_EXPERT_EPISODES = 100
    NUM_TRAINING_EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    NUM_EVAL_EPISODES = 50

    # ==================== Setup Environment ====================
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment: {ENV_NAME}")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Device: {device}\n")

    # ==================== Step 1: Collect Expert Demonstrations ====================
    print("\n" + "=" * 80)
    print(" STEP 1: COLLECTING EXPERT DEMONSTRATIONS ".center(80))
    print("=" * 80)

    expert = ExpertPolicy(env)
    collector = ExpertDataCollector(env, expert)
    expert_states, expert_actions, expert_returns = collector.collect_trajectories(
        num_episodes=NUM_EXPERT_EPISODES
    )

    # ==================== Step 2: Train BC Policy ====================
    print("\n" + "=" * 80)
    print(" STEP 2: TRAINING BEHAVIORAL CLONING POLICY ".center(80))
    print("=" * 80)

    bc_policy = BCPolicy(state_dim, action_dim).to(device)
    trainer = BCTrainer(bc_policy, learning_rate=LEARNING_RATE)

    training_history = trainer.train(
        expert_states,
        expert_actions,
        num_epochs=NUM_TRAINING_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # ==================== Step 3: Evaluate Policies ====================
    print("\n" + "=" * 80)
    print(" STEP 3: EVALUATING POLICIES ".center(80))
    print("=" * 80)

    evaluator = PolicyEvaluator(env)

    # Evaluate expert
    expert_results = evaluator.evaluate_policy(
        expert, num_episodes=NUM_EVAL_EPISODES, policy_name="Expert Policy"
    )

    # Evaluate BC policy
    bc_results = evaluator.evaluate_policy(
        bc_policy, num_episodes=NUM_EVAL_EPISODES, policy_name="BC Policy"
    )

    # Evaluate random baseline
    random_results = evaluator.evaluate_random_policy(num_episodes=NUM_EVAL_EPISODES)

    # ==================== Step 4: Analyze Distribution Shift ====================
    print("\n" + "=" * 80)
    print(" STEP 4: DISTRIBUTION SHIFT ANALYSIS ".center(80))
    print("=" * 80)

    analyzer = DistributionShiftAnalyzer(env, expert, bc_policy)
    shift_analysis = analyzer.analyze_state_distribution(num_episodes=20)

    # ==================== Step 5: Visualizations ====================
    print("\n" + "=" * 80)
    print(" STEP 5: GENERATING VISUALIZATIONS ".center(80))
    print("=" * 80 + "\n")

    visualizer = Visualizer()

    # Plot training curves
    print("Plotting training curves...")
    visualizer.plot_training_curves(
        training_history, save_path="bc_training_curves.png"
    )

    # Plot performance comparison
    print("Plotting performance comparison...")
    results_comparison = {
        "Expert": expert_results,
        "BC Policy": bc_results,
        "Random": random_results,
    }
    visualizer.plot_performance_comparison(
        results_comparison, save_path="bc_performance_comparison.png"
    )

    # Plot distribution shift
    print("Plotting distribution shift analysis...")
    visualizer.plot_distribution_shift(
        shift_analysis, save_path="bc_distribution_shift.png"
    )

    # ==================== Summary Report ====================
    print("\n" + "=" * 80)
    print(" FINAL SUMMARY ".center(80, "="))
    print("=" * 80 + "\n")

    print("Training Results:")
    print(f"  - Final validation accuracy: {training_history['val_acc'][-1]:.4f}")
    print(f"  - Final validation loss: {training_history['val_loss'][-1]:.4f}")
    print()

    print("Performance Comparison:")
    print(
        f"  - Expert Policy:  {expert_results['mean_return']:6.2f} ± {expert_results['std_return']:5.2f}"
    )
    print(
        f"  - BC Policy:      {bc_results['mean_return']:6.2f} ± {bc_results['std_return']:5.2f}"
    )
    print(
        f"  - Random Policy:  {random_results['mean_return']:6.2f} ± {random_results['std_return']:5.2f}"
    )
    print()

    performance_gap = expert_results["mean_return"] - bc_results["mean_return"]
    performance_ratio = (
        bc_results["mean_return"] / expert_results["mean_return"]
    ) * 100

    print("Analysis:")
    print(f"  - BC achieves {performance_ratio:.1f}% of expert performance")
    print(f"  - Performance gap: {performance_gap:.2f}")
    print(f"  - BC significantly outperforms random baseline")
    print()

    print("Key Observations:")
    print("  1. BC learns a reasonable policy from demonstrations alone")
    print("  2. Performance gap exists due to distribution shift")
    print("  3. BC policy visits different state distributions than expert")
    print("  4. Compounding errors lead to degraded performance over time")
    print()

    print("Limitations of Behavioral Cloning:")
    print("  - Covariate shift: BC sees different states at test time")
    print("  - No correction mechanism for errors")
    print("  - Requires large amounts of expert data")
    print("  - Cannot surpass expert performance")
    print()

    print("Solutions (covered in next demos):")
    print("  - DAgger: Interactive data collection")
    print("  - GAIL: Adversarial imitation learning")
    print("  - Hybrid approaches: BC + RL fine-tuning")

    print("\n" + "=" * 80)
    print(" DEMO COMPLETE ".center(80, "="))
    print("=" * 80 + "\n")

    env.close()


if __name__ == "__main__":
    main()
