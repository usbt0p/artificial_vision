================================================================================
  LECTURE 13: VISUAL POLICY LEARNING - COMPLETE DEMO SERIES
  From Imitation to Foundation Models
  
  Prof. David Olivieri - UVigo - VIAR25/26
  Artificial Vision Course (VIAR25/26)
================================================================================

OVERVIEW
========

This collection contains 10 comprehensive PyTorch demonstrations covering the
evolution of visual policy learning methods, from basic imitation learning to
modern foundation models.

Each demo is:
  • Self-contained and runnable independently
  • Extensively commented for educational clarity
  • Includes comprehensive visualizations
  • References original research papers
  • Follows consistent coding patterns

DEMO SERIES STRUCTURE
======================

The demos follow a progressive learning arc:

Part 1: Core Visual RL Methods (Demos 1-4)
  - Foundational approaches to learning from pixels
  - Direct policy learning and value-based methods

Part 2: Advanced Imitation Learning (Demos 5-6)
  - Beyond basic behavioral cloning
  - Adversarial and interactive approaches

Part 3: Representation Learning & Robustness (Demos 7-9)
  - Learning better visual features
  - Improving sample efficiency and transfer

Part 4: Foundation Models (Demo 10)
  - Scaling to internet-scale pretraining
  - Vision-language-action transformers

================================================================================

DEMO 1: BEHAVIORAL CLONING (behavioral_cloning.py)
===================================================

OVERVIEW
--------
Demonstrates supervised learning approach to imitation learning. The simplest
form of learning from demonstrations: collect expert data, train policy to
predict expert actions given states.

KEY FEATURES
------------
1. Expert Policy
   - Heuristic policy for CartPole demonstrations
   - Collects state-action pairs for training

2. BC Policy Network
   - MLP architecture (state → action)
   - Trained via supervised learning (cross-entropy loss)

3. Expert Data Collector
   - Gathers demonstrations (100 episodes default)
   - Creates training/validation splits

4. BC Trainer
   - Cross-entropy loss minimization
   - Adam optimizer with learning rate scheduling
   - Early stopping based on validation performance

5. Policy Evaluator
   - Compares Expert vs BC vs Random
   - Statistical performance analysis

6. Distribution Shift Analyzer
   - Demonstrates covariate shift problem
   - Shows how BC policies drift from expert distribution
   - Visualizes compounding errors over time

VISUALIZATIONS
--------------
• Training curves (loss, accuracy)
• Performance comparison (bar plots)
• State distribution analysis
• Distribution shift demonstration

KEY INSIGHT
-----------
BC learns reasonable policies but suffers from distribution shift:
  Training:  Expert visits states S_expert
  Testing:   Policy visits states S_policy
  Problem:   S_expert ≠ S_policy → compounding errors

EXPECTED RESULTS (CartPole)
---------------------------
  Expert:  200 ± 50
  BC:      150 ± 70  (75% of expert performance)
  Random:   20 ± 10

Distribution shift causes ~25% performance degradation

LIMITATIONS
-----------
• Sensitive to expert data quality
• No correction mechanism for errors
• Compounding errors over time
• Poor generalization outside training distribution

================================================================================

DEMO 2: PPO VISUAL CONTROL (PPO_visual_control.py)
===================================================

OVERVIEW
--------
Proximal Policy Optimization for visual control. On-policy policy gradient
method with clipped surrogate objective and visual observations.

KEY FEATURES
------------
1. Visual Wrapper
   - Converts environment to 84×84 pixel observations
   - RGB rendering from any Gym environment

2. CNN Encoder
   - Nature DQN architecture
   - 3 convolutional layers (32→64→64 filters)
   - Processes visual observations to features

3. Actor-Critic Network
   - Shared CNN encoder
   - Separate actor and critic heads
   - Outputs action probabilities and value estimates

4. Rollout Buffer
   - Stores on-policy trajectories (2048 steps default)
   - Computes GAE (Generalized Advantage Estimation)
   - Provides training batches

5. PPO Trainer
   - Clipped surrogate objective: min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)
   - Value function clipping for stability
   - Entropy bonus for exploration
   - Multiple epochs per rollout (4 default)

6. PPO Agent
   - Collects rollouts using current policy
   - Trains on collected data
   - Tracks training statistics

VISUALIZATIONS
--------------
6 training metrics plots:
  1. Episode returns
  2. Policy loss
  3. Value loss
  4. Entropy (exploration measure)
  5. KL divergence (policy change)
  6. Clip fraction (constraint activity)

KEY MECHANISMS
--------------
• Trust Region via Clipping: Prevents large policy updates
• GAE: Reduces variance in advantage estimates
• On-policy Learning: Updates from current policy's data
• Value Function Baseline: Reduces gradient variance

HYPERPARAMETERS
---------------
  CLIP_EPSILON = 0.2      # PPO clipping parameter
  GAE_LAMBDA = 0.95       # GAE smoothing
  NUM_EPOCHS = 4          # Optimization epochs per rollout
  BATCH_SIZE = 256        # Mini-batch size
  LEARNING_RATE = 3e-4    # Adam learning rate

EXPECTED RESULTS (CartPole)
---------------------------
  • Converges in ~50k steps
  • Final performance: 190-200 return
  • Stable learning (low variance)

PPO ADVANTAGES
--------------
• Robust and stable training
• Simple to implement
• Works across many domains
• Good sample efficiency for on-policy

================================================================================

DEMO 3: SAC PIXELS (SAC_pixels.py)
===================================

OVERVIEW
--------
Soft Actor-Critic for learning from pixels. Off-policy maximum entropy RL
algorithm with continuous action spaces (adapted for discrete actions).

KEY FEATURES
------------
1. CNN Encoder
   - Same architecture as PPO
   - Layer normalization for stability
   - Shared between Q-networks and policy

2. Twin Q-Networks
   - Two Q-networks (Q1, Q2) to reduce overestimation
   - Takes minimum for target computation
   - Separate optimizers

3. SAC Policy
   - Stochastic policy with temperature parameter
   - Outputs action probabilities (discrete adaptation)
   - Entropy-regularized objective

4. Replay Buffer
   - Off-policy storage (100k capacity)
   - Uniform sampling for training
   - Efficient memory management

5. SAC Trainer
   - Soft Q-targets: Q(s,a) - α·log π(a|s)
   - Policy loss: Maximize E[Q(s,a) - α·log π(a|s)]
   - Automatic entropy tuning (learnable temperature α)
   - Target networks with soft updates (τ=0.005)

6. SAC Agent
   - Warmup phase (random actions)
   - Off-policy updates every step
   - Epsilon-greedy exploration (decaying)

VISUALIZATIONS
--------------
5 training metrics:
  1. Episode returns
  2. Q-network losses (Q1, Q2)
  3. Policy loss
  4. Temperature (α) schedule
  5. Policy entropy

KEY DIFFERENCES FROM PPO
------------------------
  PPO:  On-policy, policy gradients, clipped objective
  SAC:  Off-policy, Q-learning + policy, maximum entropy

HYPERPARAMETERS
---------------
  GAMMA = 0.99                    # Discount factor
  TAU = 0.005                     # Soft update rate
  TARGET_ENTROPY = -log(|A|)×0.98 # Entropy target
  BUFFER_SIZE = 100000            # Replay capacity
  BATCH_SIZE = 256                # Training batch size

EXPECTED RESULTS (CartPole)
---------------------------
  • More sample efficient than PPO
  • Better exploration via entropy maximization
  • Final performance: 180-200 return
  • ~33% improvement over vanilla DQN

SAC ADVANTAGES
--------------
• Off-policy: Better sample efficiency
• Maximum entropy: Better exploration
• Twin Q-networks: Reduced overestimation
• Automatic temperature tuning: Adaptive exploration
• Robust across tasks

================================================================================

DEMO 4: DrQ AUGMENTATION (DrQ_augmentation.py)
===============================================

OVERVIEW
--------
Data-Regularized Q-learning with image augmentation. Shows that simple random
crops dramatically improve sample efficiency in visual RL.

KEY FEATURES
------------
1. Image Augmentation Module
   Four augmentation strategies:
   
   • random_crop: Pad by 4 pixels, random crop to 84×84 (KEY METHOD)
     - Most effective augmentation for visual RL
     - Provides translation invariance
     - Zero additional parameters
   
   • random_shift: Translation augmentation
     - Similar to crop but different implementation
   
   • random_intensity: Brightness variation
     - Color/lighting invariance
   
   • no_augmentation: Baseline (identity)

2. DrQ Network
   - Q-network with CNN encoder
   - Standard DQN architecture
   - No special modifications needed

3. DrQ Trainer
   - Multiple augmented views per sample (default: 2)
   - Average loss over all views
   - Epsilon-greedy with decay
   - Target network soft updates

4. Ablation Study
   - Compares DrQ (Crop) vs No Augmentation vs Random
   - Fair comparison (same hyperparameters)
   - Statistical evaluation

VISUALIZATIONS
--------------
• Augmentation examples (side-by-side comparison)
• Training curves (4 metrics)
• Performance comparison (bar plots)
• Sample efficiency analysis

KEY INSIGHT
-----------
Data augmentation = Implicit regularization

Training on augmented views:
  1. Creates effective data multiplication
  2. Enforces spatial invariance
  3. Prevents overfitting to specific pixels
  4. Improves generalization

EXPECTED RESULTS (CartPole)
---------------------------
  No Aug:      150 ± 70  (baseline)
  DrQ (Crop):  200 ± 50  (~33% improvement!)
  Random:       20 ± 10

Same training time, zero additional parameters, major gains!

WHY IT WORKS
------------
Random crops teach the policy to be invariant to:
  • Small translations
  • Exact pixel positions
  • Minor visual variations

This creates a more robust visual representation.

DrQ ADVANTAGES
--------------
• Simple to implement (just add augmentation!)
• Zero additional model parameters
• Major sample efficiency gains
• Works with any Q-learning algorithm
• Minimal computational overhead

BEST PRACTICE
--------------
Always use random crop augmentation for visual RL!
It's the easiest performance boost available.

================================================================================

DEMO 5: GAIL IMPLEMENTATION (GAIL_implementation.py)
====================================================

OVERVIEW
--------
Generative Adversarial Imitation Learning. Uses discriminator to distinguish
expert from learner, training policy to fool the discriminator.

KEY FEATURES
------------
1. Discriminator Network
   - Binary classifier (expert vs. learner)
   - Takes state-action pairs as input
   - Outputs probability D(s,a) ∈ [0,1]
     * D(s,a) ≈ 1: Expert-like
     * D(s,a) ≈ 0: Learner-like
   - MLP with dropout for stability

2. GAIL Policy Network
   - Generator/learner policy
   - Standard MLP architecture
   - Trained to maximize "expertness"

3. Adversarial Training Framework
   
   Discriminator Update:
   - Binary cross-entropy loss
   - Maximize accuracy on expert data (label = 1)
   - Maximize accuracy on learner data (label = 0)
   
   Policy Update:
   - Policy gradient with discriminator rewards
   - Reward: r(s,a) = log D(s,a)
   - Maximizes "fooling" the discriminator
   - GAE for advantage estimation

4. GAIL Buffer
   - Stores learner trajectories
   - On-policy collection (cleared each iteration)
   - Provides data for both discriminator and policy

5. Complete Training Loop
   ```
   For each iteration:
     1. Collect trajectories with current policy
     2. Update discriminator (expert vs. learner)
     3. Update policy using discriminator rewards
     4. Repeat
   ```

VISUALIZATIONS
--------------
6 training curves:
  1. Episode returns over iterations
  2. Discriminator loss
  3. Policy loss
  4. Expert accuracy (discriminator on expert data)
  5. Learner accuracy (discriminator on learner data)
  6. Combined accuracy plot

GAIL VS BC COMPARISON
---------------------
| Feature            | BC              | GAIL                  |
|--------------------|-----------------|------------------------|
| Learning           | Supervised      | Adversarial RL        |
| Training           | One-shot        | Iterative             |
| Objective          | Match actions   | Match distributions   |
| Distribution Shift | Suffers         | Robust                |
| Sample Efficiency  | Lower           | Higher                |
| Complexity         | Simple          | Moderate              |

EXPECTED RESULTS (CartPole)
---------------------------
  Expert:  200 ± 50
  GAIL:    180 ± 60  (90% of expert)
  BC:      150 ± 70  (75% of expert)
  Random:   20 ± 10

GAIL improvement over BC: ~20%

KEY ADVANTAGE
-------------
Unlike BC which learns:
  π(a|s) ≈ π_expert(a|s)  [individual pairs]

GAIL learns:
  p_π(s,a) ≈ p_expert(s,a)  [joint distribution]

This distribution matching is more robust to compounding errors!

TRAINING DYNAMICS
-----------------
Early Training:
  - Discriminator easily distinguishes expert from learner
  - Expert accuracy: ~90%, Learner accuracy: ~90%
  - Policy receives strong signal to improve

Mid Training:
  - Policy gets better, discriminator task harder
  - Accuracies converge toward 50%
  - Indicates distribution matching

Late Training:
  - Both converge to equilibrium
  - Policy closely mimics expert distribution
  - Discriminator struggles (both look similar)

RESEARCH IMPACT
---------------
GAIL showed that:
  • Imitation learning can be framed as adversarial training
  • Distribution matching > direct action matching
  • No explicit reward function needed
  • Scales to high-dimensional state spaces

================================================================================

DEMO 6: DAgger IMPLEMENTATION (DAgger_demo.py)
===============================================

OVERVIEW
--------
Dataset Aggregation for interactive imitation learning. Addresses BC's
distribution shift problem through expert queries on learner-visited states.

KEY FEATURES
------------
1. Interactive Expert Oracle
   - Can be queried for optimal actions
   - Tracks number of expert queries (important metric)
   - Demonstrates oracle-based learning

2. DAgger Policy Network
   - Same architecture as BC policy
   - Shows that algorithm matters, not architecture

3. Aggregated Dataset
   - Stores data from all iterations
   - Tracks contribution from each iteration
   - Grows over time as new data is added
   - Key difference from BC (single dataset)

4. DAgger Algorithm Implementation
   ```
   Initial: Collect D₀ from expert demonstrations
   
   For iteration i = 1 to N:
     1. Train policy πᵢ on aggregated dataset Dᵢ₋₁
     2. Run πᵢ to collect states Sᵢ
     3. Query expert for actions on Sᵢ → get labels Aᵢ
     4. Aggregate: Dᵢ = Dᵢ₋₁ ∪ {Sᵢ, Aᵢ}
   ```

5. Beta Scheduling
   Three strategies for mixing expert/learner during rollouts:
   
   • Constant (β=1.0): Always follow expert during collection
   • Linear decay: β = 1 - i/N (gradually reduce expert)
   • Exponential decay: β = 0.5^i (aggressive reduction)

6. On-Policy Data Collection (Core DAgger Innovation)
   - Runs current policy to visit states
   - Queries expert for labels on visited states
   - Creates dataset from learner's state distribution

VISUALIZATIONS
--------------
6 training curves:
  1. Policy Performance (return over iterations)
  2. Training Accuracy (how well policy fits dataset)
  3. Dataset Size (growth of aggregated data)
  4. Training Loss (convergence monitoring)
  5. Expert Queries (cost of oracle access)
  6. Beta Schedule (exploration strategy)

Plus:
  • Performance comparison (DAgger vs BC vs GAIL vs Expert vs Random)
  • Learning curve comparison
  • Distribution analysis

THE DISTRIBUTION SHIFT PROBLEM (BC)
-----------------------------------
Training:     Expert visits states S_expert
Testing:      Policy visits states S_policy
Problem:      S_expert ≠ S_policy → poor performance

DAGGER'S SOLUTION
-----------------
Iteration 1:  Policy visits S₁ → Expert labels S₁
Iteration 2:  Policy visits S₂ → Expert labels S₂
...
Result:       Training data covers S_policy!

EXPECTED RESULTS (CartPole)
---------------------------
  Expert:  200 ± 50
  DAgger:  195 ± 55  (97% of expert!)
  GAIL:    180 ± 60  (90% of expert)
  BC:      150 ± 70  (75% of expert)
  Random:   20 ± 10

DAgger improvement over BC: ~30%
Expert queries: ~2000-3000 total

DAgger VS BC VS GAIL
--------------------
| Feature         | BC      | DAgger     | GAIL            |
|-----------------|---------|------------|-----------------|
| Training        | One-shot| Iterative  | Adversarial     |
| Expert Access   | Demos   | Queries    | Demos           |
| Distribution    | Off     | On-policy  | Dist matching   |
| Shift Robust    | Poor    | Excellent  | Good            |
| Sample Eff      | High    | Medium     | Lower           |
| Implementation  | Simplest| Moderate   | Complex         |

THEORETICAL FOUNDATION
----------------------
DAgger reduces imitation learning to no-regret online learning:
  • Each iteration reduces distribution mismatch
  • Provably converges to expert performance
  • Mistake bound: O(T√N) where T=horizon, N=iterations

WHEN TO USE DAgger
------------------
✅ Use DAgger when:
  • You have access to an expert oracle
  • Distribution shift is a problem
  • You can afford iterative training
  • Expert queries are cheaper than full demonstrations

❌ Don't use DAgger when:
  • Expert oracle unavailable (use BC or GAIL)
  • Expert queries are very expensive
  • One-shot learning is required
  • Real-time constraints exist

================================================================================

DEMO 7: CURL CONTRASTIVE (CURL_contrastive.py)
===============================================

OVERVIEW
--------
Contrastive Unsupervised Representations for Reinforcement Learning.
Self-supervised representation learning improves RL sample efficiency.

KEY FEATURES
------------
1. Contrastive Learning Architecture
   
   • CURLEncoder: CNN encoder for visual features
   • Query Encoder: Trainable encoder (gets gradient updates)
   • Key Encoder: Momentum-updated encoder (slow-moving target)
   • Projection head for contrastive learning (128-dim latent)

2. Momentum Encoder
   - Key encoder updated via exponential moving average
   - Momentum parameter τ = 0.99 (slow updates)
   - Provides stable targets for contrastive learning
   - Inspired by MoCo (Momentum Contrast)

3. Data Augmentation for Contrastive Learning
   
   RandomCropAugmentation: Creates positive pairs
   - Same observation → Two different random crops
   - Positive pair: (crop1, crop2) of same observation
   - Negative pairs: Different observations in batch

4. InfoNCE Loss
   - Contrastive loss function
   - Pulls positive pairs together in embedding space
   - Pushes negative pairs apart
   - Temperature scaling (τ = 0.1)
   - Implemented as cross-entropy with diagonal targets

5. CURL Module
   - Complete contrastive learning system
   - Bilinear similarity: query^T W key
   - W matrix learns optimal feature comparison
   - L2 normalization of features before comparison

6. Integration with SAC
   - CURL encoder shared between contrastive and RL tasks
   - Q-network uses CURL features
   - Actor uses CURL features
   - Decoupled representation and RL learning

CONTRASTIVE LEARNING FRAMEWORK
-------------------------------
```
Observation → [Random Crop 1] → Query Encoder → q
           ↘ [Random Crop 2] → Key Encoder   → k

Similarity: sim(q, k) = q^T W k

Loss: InfoNCE pulls (q, k) together
             pushes (q, k') apart for k' ≠ k
```

WHY IT WORKS
------------
1. Self-supervision: Learn from augmentations (no labels needed)
2. Invariance: Features ignore irrelevant transformations
3. Better features: Contrastive task creates useful representations
4. Transfer: Features help both contrastive and RL tasks

DUAL OPTIMIZATION
-----------------
```python
# CURL update (representation learning)
curl_loss = InfoNCE(query, key)
curl_optimizer.step()
momentum_update(key_encoder)

# RL update (policy learning)
q_loss = MSE(Q(s,a), target)
q_optimizer.step()
```

InfoNCE LOSS EXPLAINED
----------------------
Given batch of N observations:
  - Positive pair: (obs_i_crop1, obs_i_crop2) → same obs
  - Negative pairs: (obs_i, obs_j) for i ≠ j → different obs

  Logits[i,j] = similarity(query_i, key_j)

  Loss = CrossEntropy(Logits, diagonal_labels)
       = -log(exp(sim(q_i, k_i)) / Σⱼ exp(sim(q_i, k_j)))

MOMENTUM ENCODER UPDATE
-----------------------
```python
# Slow-moving key encoder
θ_key ← τ × θ_key + (1-τ) × θ_query

# τ = 0.99: Very slow updates
# Provides stable targets for contrastive learning
```

HYPERPARAMETERS
---------------
  CURL_LATENT_DIM = 128    # Contrastive feature dimension
  MOMENTUM = 0.99          # Key encoder momentum
  TEMPERATURE = 0.1        # InfoNCE temperature
  CURL_WEIGHT = 1.0        # Contrastive loss weight
  LEARNING_RATE = 1e-4     # All networks

VISUALIZATIONS
--------------
• Training curves (6 metrics)
• t-SNE visualization of learned representations
  - Clustering: Similar states group together
  - Separation: Different states separate
  - Structure: Meaningful organization emerges

SAMPLE EFFICIENCY GAINS
------------------------
CURL achieves better performance with same data:
  Without CURL:  150 return @ 30k steps
  With CURL:     180 return @ 30k steps
  Improvement:   20% better sample efficiency

CURL VS STANDARD RL
-------------------
| Feature          | Standard RL   | CURL              |
|------------------|---------------|-------------------|
| Representation   | Task-specific | Self-supervised   |
| Learning         | Single task   | Dual task         |
| Sample Efficiency| Baseline      | Improved          |
| Features         | May overfit   | More general      |
| Augmentation     | Optional      | Essential         |

RESEARCH IMPACT
---------------
CURL showed that:
  • Self-supervised learning helps RL
  • Contrastive methods transfer to control
  • Simple augmentations (crops) are powerful
  • Momentum encoders stabilize learning

This bridges computer vision (contrastive learning) and RL!

================================================================================

DEMO 8: RAD STRATEGIES (RAD_strategies.py)
===========================================

OVERVIEW
--------
Reinforcement Learning with Augmented Data - comprehensive comparison of
augmentation strategies for visual RL.

KEY FEATURES
------------
1. Comprehensive Augmentation Library (RADAugmentations)
   
   9+ augmentation strategies:
   
   • random_crop: Pad + random crop (MOST EFFECTIVE)
     - Provides translation invariance
     - Key technique for visual RL
   
   • random_shift: Translation augmentation
     - Similar to crop, different implementation
   
   • cutout: Random rectangular occlusion (zeros)
     - Occlusion robustness
   
   • cutout_color: Random occlusion with random color
     - More realistic occlusion
   
   • random_flip: Horizontal flip
     - Symmetry augmentation
   
   • random_rotation: Small angle rotations
     - Rotational invariance
   
   • color_jitter: Brightness + contrast variation
     - Lighting robustness
   
   • random_grayscale: Probabilistic grayscale conversion
     - Color invariance
   
   • no_augmentation: Baseline (identity)

2. Ablation Study Framework
   - Compare multiple augmentation strategies
   - Fair comparison (same hyperparameters)
   - Statistical evaluation
   - Performance ranking

3. RAD Algorithm Implementation
   - Q-learning with augmentation
   - Multiple augmented views per sample (default: 2)
   - Averages loss over augmentations
   - Target network with soft updates

VISUALIZATIONS
--------------
A. Augmentation Gallery
   - Side-by-side visualization of all strategies
   - Shows effect of each augmentation

B. Performance Comparison
   - Bar plots with error bars
   - Box plots showing distributions
   - Sorted by performance

C. Learning Curves
   - All strategies on same plot
   - Sample efficiency comparison

AUGMENTATION STRATEGIES EXPLAINED
----------------------------------

Random Crop (Most Reliable):
  Original: 84×84
  Pad: +4 pixels → 92×92
  Crop: Random 84×84 region
  Effect: Translation invariance

Cutout (Occlusion Robustness):
  Random position: (x, y)
  Mask size: 20×20
  Fill: Zeros or random color
  Effect: Occlusion invariance

Color Jitter (Lighting Robustness):
  Brightness: ×(0.8 to 1.2)
  Contrast: ×(0.8 to 1.2)
  Effect: Lighting invariance

EXPECTED RESULTS (CartPole)
---------------------------
Performance Ranking (typical):
  1. Crop:         180 ± 50  (Best)
  2. Shift:        175 ± 55
  3. Color Jitter: 165 ± 60
  4. Cutout:       160 ± 65
  5. None:         150 ± 70  (Baseline)

Improvement: 20% with best augmentation

TASK-SPECIFIC INSIGHTS
----------------------
| Task Type      | Best Augmentations  | Avoid          |
|----------------|---------------------|----------------|
| Navigation     | Crop, Shift         | Flip, Rotate   |
| Manipulation   | Crop, Color Jitter  | Flip           |
| Atari Games    | Crop, Cutout        | Grayscale      |
| Robotics       | Crop, Color Jitter  | Rotate         |

RAD VS DrQ VS CURL
------------------
| Method | Focus              | Augmentations | Loss           |
|--------|--------------------|---------------|----------------|
| DrQ    | Single aug (crop)  | 1 type        | Q-learning     |
| RAD    | Multiple augs      | 8+ types      | Q-learning     |
| CURL   | Contrastive learn  | Crop          | InfoNCE + QL   |

BEST PRACTICES FROM RAD
-----------------------
1. Start with crop: Most reliable across tasks
2. Avoid semantic breaks: Don't flip if left/right matters
3. Task matters: Best aug depends on task
4. Combine carefully: Multiple augs can help or hurt
5. Ablate systematically: Empirical evaluation essential

WHEN TO USE EACH AUGMENTATION
------------------------------

✅ Always Safe:
  • Random Crop
  • Random Shift
  • Color Jitter

⚠️ Task-Dependent:
  • Cutout (if occlusion is realistic)
  • Flip (if left/right symmetry exists)
  • Grayscale (if color is irrelevant)

❌ Usually Harmful:
  • Rotation (breaks spatial relationships)
  • Extreme jitter (breaks visual coherence)

RESEARCH INSIGHTS
-----------------
RAD's key findings:
  1. Crop is king: Works for almost all tasks
  2. Simplicity wins: Simple augs often beat complex
  3. Task-specific: No universal best augmentation
  4. Diminishing returns: More augs ≠ always better
  5. Stability matters: Some augs hurt training

================================================================================

DEMO 9: DOMAIN RANDOMIZATION (domain_randomization.py)
=======================================================

OVERVIEW
--------
Domain randomization for sim-to-real transfer. Training on diverse visual
appearances improves robustness and real-world transfer.

KEY FEATURES
------------
1. Comprehensive Domain Randomization Module (DomainRandomization)
   
   Multiple randomization techniques:
   
   • Color Randomization: Per-channel color multipliers
   • Brightness Randomization: Overall illumination changes
   • Contrast Randomization: Contrast adjustment
   • Gaussian Noise: Sensor noise simulation
   • Hue Shift: Color space rotation
   • Saturation Randomization: Color intensity variation
   • Combined Randomization: All techniques together

2. Domain Randomized Environment Wrapper
   - Applies randomization to observations
   - Wraps any Gym environment
   - Configurable randomization intensity
   - Supports both RGB and grayscale

3. Sim-to-Real Transfer Experiment
   ```
   Training: Randomized simulation
   Testing:  Canonical/"real" environment
   Metric:   Zero-shot transfer performance
   ```

4. Transfer Evaluation - Four Scenarios:
   1. Randomized → Real: Main transfer test
   2. No Rand → Real: Baseline
   3. Randomized → Rand: Training performance
   4. No Rand → Rand: Overfitting test

VISUALIZATIONS
--------------
A. Randomization Effects
   - Side-by-side: Original vs. randomized
   - 8 different randomized versions
   - Shows diversity

B. Training Comparison
   - Learning curves with/without randomization
   - Loss curves
   - Smoothed for clarity

C. Transfer Analysis
   - Bar plots with error bars
   - Box plots showing distributions
   - All four transfer scenarios

CORE INSIGHT
------------
```
Train on diverse simulations → Robust to variations → Better real-world transfer

Diversity during training = Robustness at test time
```

RANDOMIZATION TECHNIQUES
------------------------
1. Colors:      RGB channels × random multipliers
2. Brightness:  Intensity × random factor
3. Contrast:    (pixel - mean) × random factor + mean
4. Noise:       Add Gaussian noise
5. Hue:         Rotate color channels
6. Saturation:  Blend with grayscale

RANDOMIZATION INTENSITY
-----------------------
  Low (0.1):    Subtle variations
  Medium (0.2): Moderate diversity (RECOMMENDED)
  High (0.3):   Strong randomization
  Extreme (0.5): May harm learning

SIM-TO-REAL GAP
---------------

Without Domain Randomization:
  Simulation: Perfect rendering, consistent visuals
  Real World: Varying lighting, textures, noise
  Result:     Policy fails due to visual mismatch

With Domain Randomization:
  Simulation: Random lighting, textures, noise
  Real World: Just another random variation!
  Result:     Policy generalizes successfully

BEST PRACTICES
--------------

✅ Do Randomize:
  • Lighting conditions
  • Colors and textures
  • Camera angles (if variable)
  • Sensor noise
  • Background elements

❌ Don't Randomize:
  • Task-critical features (e.g., object shape for grasping)
  • Physics parameters (handle separately)
  • Reward-relevant information
  • Spatial relationships (usually)

RANDOMIZATION STRATEGIES
------------------------

Conservative (Low variance):
  color_intensity = 0.1
  brightness_intensity = 0.1
  noise_level = 3.0

Moderate (Recommended):
  color_intensity = 0.2
  brightness_intensity = 0.2
  noise_level = 5.0

Aggressive (High variance):
  color_intensity = 0.3
  brightness_intensity = 0.3
  noise_level = 10.0

EVALUATION PROTOCOL
-------------------
1. Train on randomized simulation
2. Freeze policy
3. Test on canonical environment (no randomization)
4. Measure: Zero-shot transfer performance
5. Compare: vs. policy trained without randomization

KEY METRICS
-----------

Robustness:
  min(performance_on_canonical, performance_on_randomized)

Transfer Gap:
  performance_real - performance_sim

Generalization:
  performance_on_unseen_variations

PRACTICAL APPLICATIONS
----------------------

Robotics:
  • Train in Isaac Gym, deploy on real robot
  • Randomize: lighting, textures, camera noise
  • Result: Sim-to-real transfer without real data

Autonomous Driving:
  • Train in CARLA simulator
  • Randomize: weather, lighting, traffic
  • Result: Robust to real-world conditions

Manufacturing:
  • Train with varied object appearances
  • Randomize: colors, positions, orientations
  • Result: Generalize to new products

RESEARCH IMPACT
---------------
Domain randomization showed that:
  • Diversity during training → Robustness at test
  • Sim-to-real transfer is possible without real data
  • Simple randomization beats complex domain adaptation
  • Works across vision, robotics, control tasks

LIMITATIONS
-----------
⚠️ Reality Gap: Some aspects hard to randomize
⚠️ Sample Efficiency: More diversity = slower learning
⚠️ Hyperparameters: Intensity needs tuning
⚠️ Physical Plausibility: Random ≠ realistic

================================================================================

DEMO 10: RT-1 ARCHITECTURE (RT1_architecture_demo.py)
======================================================

🎓 CAPSTONE DEMO - FOUNDATION MODEL CONCEPTS

OVERVIEW
--------
Demonstrates vision-language-action transformer architecture inspired by RT-1
(Robotics Transformer). Shows how foundation models work for embodied AI.

KEY FEATURES
------------

1. VISION TOKENIZATION (ImageTokenizer)
   - Patch-based tokenization: Splits 84×84 image into patches
   - Patch size: 14×14 → 36 patches total
   - Convolutional patch embedding: Efficient extraction
   - Positional embeddings: Learnable position encoding
   - Output: Sequence of vision tokens [batch, 36, 256]

2. LANGUAGE INSTRUCTION ENCODING (LanguageEncoder)
   - Token embeddings: Maps words to vectors
   - Positional encoding: Sequence position information
   - Transformer encoder: 2-layer processing
   - Attention masking: Handles variable-length instructions
   - Output: Encoded instruction tokens [batch, seq_len, 256]

3. RT-1 TRANSFORMER (RT1Transformer)
   - Multimodal token fusion: Concatenates vision + language
   - Action query token: Special learnable token for action
   - Self-attention layers: 4-layer transformer encoder
   - Cross-modal attention: Vision ↔ Language interaction
   - Action prediction head: Maps action token → action logits

4. COMPLETE RT-1 POLICY
   ```
   Image (84×84×3) → Vision Tokens (36×256)
                      ↓
   Instruction     → Language Tokens (20×256)
                      ↓
                 [Vision | Language | Action Query]
                      ↓
                 Transformer Encoder (4 layers)
                      ↓
                 Action Token → Action Head
                      ↓
                 Action Logits
   ```

5. MULTI-TASK LEARNING
   - Multiple instructions: "balance pole", "keep stable", etc.
   - Shared policy: Same network for all tasks
   - Instruction conditioning: Task specified via language
   - Foundation model principle: One model, many tasks

RT-1 ARCHITECTURAL FLOW
------------------------
```
TOKENIZATION EVERYTHING:

Images  → Patches  → Tokens
Text    → Words    → Tokens
Actions → Discrete → Tokens

Everything is a token sequence!
```

TRANSFORMER ARCHITECTURE
------------------------
• Self-attention: All tokens attend to all tokens
• Multimodal: Vision and language in same space
• Scalable: Grows with data and compute
• Flexible: Easy to add new modalities

FOUNDATION MODEL PRINCIPLES
----------------------------
1. Large-scale pretraining: Millions of demonstrations
2. Multi-task learning: Hundreds of tasks
3. Transfer learning: Generalize to new tasks
4. Instruction following: Language-conditioned control

VISUALIZATIONS
--------------

A. Architecture Diagram (7 components):
   1. Visual Input
   2. Image Tokenization
   3. Language Encoding
   4. Token Concatenation
   5. Transformer Processing
   6. Action Token Extraction
   7. Action Prediction

B. Tokenization Process:
   • Original image display
   • Patch grid overlay
   • Token embedding heatmap
   • Shows how images become sequences

C. Training Curves:
   • Episode returns over time
   • Loss convergence
   • Smoothed for clarity

MODEL STATISTICS
----------------
Demo Implementation:
  - Vision Tokenizer:     ~100K parameters
  - Language Encoder:     ~500K parameters  
  - RT-1 Transformer:     ~2M parameters
  - Action Head:          ~100K parameters
  Total:                  ~2.7M parameters

Real World:
  - Real RT-1:            ~35M parameters
  - Real RT-2:            ~55B parameters (with PaLI)

RT-1 → RT-2 EVOLUTION
---------------------
| Feature        | RT-1              | RT-2                    |
|----------------|-------------------|-------------------------|
| Vision         | Custom CNN/ViT    | PaLI (pretrained)       |
| Language       | Task-specific     | Internet-scale LLM      |
| Training       | Robotics only     | Vision-language + robot |
| Parameters     | 35M               | 55B                     |
| Reasoning      | Limited           | Chain-of-thought        |
| Generalization | Good              | Exceptional             |

FOUNDATION MODEL SCALING
------------------------

Data Scaling:
  RT-1:  130K trajectories, 700 tasks
  RT-2:  Internet-scale vision-language data + robotics

Model Scaling:
  Small:  ~10M parameters
  Medium: ~100M parameters  
  Large:  ~1B parameters
  XLarge: ~10B+ parameters

Emergent Capabilities:
  • Zero-shot: New tasks without training
  • Few-shot: Learn from demonstrations
  • Chain-of-thought: Reasoning about actions
  • Multimodal understanding: Rich perception

WHY TRANSFORMERS FOR ROBOTICS?
-------------------------------
1. Sequence modeling: Natural for time-series control
2. Attention: Focus on relevant visual/language features
3. Scalability: Proven to scale to billions of parameters
4. Transfer: Pretrained models accelerate learning
5. Multimodal: Unified architecture for vision + language

WHY FOUNDATION MODELS?
----------------------
1. Data efficiency: Leverage internet-scale pretraining
2. Generalization: Transfer across tasks and embodiments
3. Reasoning: LLM capabilities for planning
4. Open-ended: Handle novel situations
5. Democratization: Shared models benefit all

THE PATH FORWARD
----------------
```
Specialized RL → Multi-task RL → Foundation Models
Small models  → Large models   → Internet-scale
Single robot  → Many robots    → Universal policies
```

THE FUTURE: FOUNDATION MODELS FOR ROBOTICS
-------------------------------------------
• Scaling: Bigger models, more data, more compute
• Generalization: Cross-task, cross-embodiment
• Reasoning: LLM integration for planning
• Sim-to-real: Robust transfer via scale
• Open research: Open X-Embodiment datasets

================================================================================

COMPLETE DEMO SERIES SUMMARY
=============================

🎉 CONGRATULATIONS! You have completed all 10 comprehensive demos!

PROGRESSIVE LEARNING ARC
------------------------

Part 1: Core Visual RL Methods (Demos 1-4)
  ✓ Behavioral Cloning - Imitation learning basics
  ✓ PPO - Policy gradients for visual control
  ✓ SAC - Off-policy continuous control
  ✓ DrQ - Data augmentation (crop focus)

Part 2: Advanced Imitation Learning (Demos 5-6)
  ✓ GAIL - Adversarial imitation
  ✓ DAgger - Interactive imitation

Part 3: Representation Learning & Robustness (Demos 7-9)
  ✓ CURL - Contrastive representation learning
  ✓ RAD - Augmentation strategy comparison
  ✓ Domain Randomization - Sim-to-real transfer

Part 4: Foundation Models (Demo 10)
  ✓ RT-1 - Foundation model architecture

DEMO CHARACTERISTICS
--------------------
Each demo is:
  🎓 Educational: Clear explanations and comments
  🔬 Self-contained: Runs independently
  📊 Visual: Comprehensive plots and analysis
  🏗️ Well-structured: Clean, modular code
  📚 Referenced: Cites original papers

COMMON PATTERNS ACROSS DEMOS
-----------------------------

1. Environment Wrappers
   - VisualWrapper: Converts any Gym env to pixel observations
   - Frame stacking: Temporal information (4 frames)
   - Grayscale conversion: Reduces input channels

2. CNN Architectures
   - Nature DQN architecture (standard)
   - 32→64→64 filters
   - 8×4, 4×2, 3×1 kernel sizes
   - Xavier/orthogonal initialization

3. Training Patterns
   - Replay buffers (off-policy methods)
   - Rollout buffers (on-policy methods)
   - Gradient clipping (stability)
   - Learning rate scheduling
   - Early stopping / convergence criteria

4. Evaluation Protocols
   - Multiple evaluation episodes (10-50)
   - Statistical analysis (mean ± std)
   - Comparison with baselines
   - Distribution analysis

5. Visualization Standards
   - Training curves (smoothed)
   - Performance comparisons (bar plots, box plots)
   - Method-specific visualizations
   - High-quality figures (matplotlib/seaborn)

KEY TAKEAWAYS BY TOPIC
----------------------

Imitation Learning:
  • BC: Simple but suffers from distribution shift
  • GAIL: Distribution matching via adversarial training
  • DAgger: Interactive queries fix distribution shift
  • Lesson: Distribution matching > action matching

Visual RL:
  • PPO: Stable on-policy learning
  • SAC: Sample-efficient off-policy learning
  • Lesson: Off-policy often better for pixels

Data Augmentation:
  • DrQ: Random crop is remarkably effective
  • RAD: Task-specific augmentation selection matters
  • CURL: Contrastive learning + augmentation
  • Lesson: Always use random crop for visual RL!

Robustness:
  • Domain Randomization: Sim-to-real via diversity
  • Lesson: Training diversity → test robustness

Foundation Models:
  • RT-1: Transformer architecture for robotics
  • Lesson: Scale + multimodal = generalization

PERFORMANCE HIERARCHY (CartPole Visual Control)
-----------------------------------------------
  Expert:           200 ± 50
  DAgger:           195 ± 55  (97% of expert)
  DrQ/CURL:         180 ± 60  (90% of expert)
  GAIL:             180 ± 60  (90% of expert)
  PPO/SAC:          170 ± 65  (85% of expert)
  BC:               150 ± 70  (75% of expert)
  Random:            20 ± 10

COMPUTATIONAL REQUIREMENTS
--------------------------
All demos designed to run on:
  • CPU: Works (slower)
  • Single GPU: Recommended
  • Time per demo: 10-30 minutes (depending on settings)

HYPERPARAMETER GUIDELINES
--------------------------

Learning Rates:
  • Policy networks: 1e-4 to 3e-4
  • Q-networks: 1e-4 to 1e-3
  • Discriminators (GAIL): 3e-4

Batch Sizes:
  • On-policy (PPO): 64-256
  • Off-policy (SAC, DrQ): 128-256

Exploration:
  • Epsilon decay: 0.995
  • Entropy coefficients: 0.01-0.1
  • Temperature (SAC): Auto-tuned

Network Architectures:
  • CNN features: 256-512 dimensions
  • MLP hidden: 128-256 dimensions
  • Transformer: 256-512 embedding dim

RECOMMENDED LEARNING PATH
--------------------------

For Students:
  1. Start with BC (simplest)
  2. Move to PPO (core RL)
  3. Compare SAC (off-policy)
  4. Add DrQ (augmentation)
  5. Try GAIL or DAgger (advanced imitation)
  6. Explore CURL/RAD (representation learning)
  7. Understand Domain Rand (robustness)
  8. Study RT-1 (foundation models)

For Researchers:
  • Focus on demos relevant to your work
  • Modify architectures for your domain
  • Combine techniques (e.g., CURL + DrQ + Domain Rand)
  • Scale up for real problems

EXTENDING THE DEMOS
--------------------

Easy Extensions:
  • Try different environments (Atari, MuJoCo)
  • Modify network architectures
  • Adjust hyperparameters
  • Combine techniques

Advanced Extensions:
  • Multi-task learning across environments
  • Continuous action spaces
  • Hierarchical policies
  • Meta-learning / few-shot adaptation
  • Real robot deployment

TROUBLESHOOTING
---------------

Common Issues:
  1. Slow learning: Reduce learning rate
  2. Instability: Add gradient clipping, reduce LR
  3. Poor performance: Check CNN architecture, augmentation
  4. Memory errors: Reduce batch size, buffer size
  5. GPU errors: Check CUDA compatibility

Debugging Tips:
  • Start with small number of episodes
  • Visualize observations (check preprocessing)
  • Monitor gradient norms
  • Compare with random policy
  • Check data distribution

CITATION & REFERENCES
---------------------

If using these demos in research or teaching, please reference:

  Lecture 13: Visual Policy Learning - From Imitation to Foundation Models
  Prof. David Olivieri
  Universidad de Vigo
  Artificial Vision Course (VIAR25/26)

Original Paper References:
  • BC: General supervised learning
  • PPO: Schulman et al., "Proximal Policy Optimization" (2017)
  • SAC: Haarnoja et al., "Soft Actor-Critic" (2018)
  • DrQ: Kostrikov et al., "Image Augmentation Is All You Need" (2020)
  • GAIL: Ho & Ermon, "Generative Adversarial Imitation Learning" (2016)
  • DAgger: Ross et al., "A Reduction of Imitation Learning..." (2011)
  • CURL: Srinivas et al., "CURL: Contrastive Unsupervised..." (2020)
  • RAD: Laskin et al., "Reinforcement Learning with Augmented Data" (2020)
  • Domain Rand: Tobin et al., "Domain Randomization..." (2017)
  • RT-1: Brohan et al., "RT-1: Robotics Transformer..." (2022)

ADDITIONAL RESOURCES
--------------------

Recommended Reading:
  • Sutton & Barto: "Reinforcement Learning: An Introduction"
  • Goodfellow et al.: "Deep Learning"
  • OpenAI Spinning Up: spinningup.openai.com
  • Lilian Weng's Blog: lilianweng.github.io

Frameworks & Libraries:
  • Stable-Baselines3: github.com/DLR-RM/stable-baselines3
  • CleanRL: github.com/vwxyzjn/cleanrl
  • RLlib: docs.ray.io/en/latest/rllib
  • Tianshou: github.com/thu-ml/tianshou

Datasets:
  • Open X-Embodiment: robotics-transformer-x.github.io
  • Atari 2600: github.com/openai/gym
  • DMControl: github.com/deepmind/dm_control

ACKNOWLEDGMENTS
---------------

These demonstrations were created for educational purposes to help students
understand the progression from classical imitation learning to modern
foundation models for embodied AI.

Special thanks to the research community for developing these methods and
making their work accessible through open publications.

================================================================================

GETTING STARTED
===============

To run any demo:

1. Install dependencies:
   pip install torch numpy gymnasium matplotlib seaborn scikit-learn

2. Optional (for some demos):
   pip install opencv-python

3. Run a demo:
   python behavioral_cloning.py
   python PPO_visual_control.py
   python SAC_pixels.py
   ... (etc)

4. Adjust settings:
   - Edit configuration section in each demo
   - Modify NUM_EPISODES, BATCH_SIZE, etc.
   - Change environment (ENV_NAME)

5. Experiment:
   - Try different hyperparameters
   - Compare different methods
   - Visualize results
   - Build on the code

================================================================================

FINAL NOTES
===========

These demos prioritize educational clarity over production performance.
For real research/applications:
  • Scale up model sizes
  • Use more training data
  • Tune hyperparameters carefully
  • Consider domain-specific modifications
  • Validate on diverse environments

The field of visual policy learning is rapidly evolving. These demos capture
the state of the art as of 2022-2023, with RT-1/RT-2 representing the frontier.
New methods continue to emerge!

We hope these demonstrations help you understand and implement modern visual
policy learning methods. Happy learning and researching!

================================================================================

Questions or Issues?
Contact: Prof. David Olivieri
Universidad de Vigo, Spain

================================================================================
END OF README
================================================================================