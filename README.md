# Drone Diffusion Brain (drone-diffusion-brain)

A modular, production-grade deep imitation learning repository engineered to train vision- and kinematic-conditioned Diffusion Policies for autonomous multirotor flight.
Instead of traditional Reinforcement Learning that outputs a single high-variance motor command per step, this framework treats trajectory generation as a generative problem: it predicts smooth, synchronized action chunks from random Gaussian noise conditioned on the drone's current physical state.

## Pipeline Architecture
```text
┌────────────────────────────────────────────────────────────────────────┐
│                        DATA GENERATION & INGESTION                     │
│                                                                        │
│   [ Analytic PID Expert ] ──► Continuous Flight Trajectories           │
│                                       │                                │
│                                       ▼                                │
│                           [ Compressed Zarr Store ]                    │
│                                       │                                │
│                                       ▼                                │
│                     [ Sliding-Window Dataset Generator ]               │
│                     • Samples: (State Obs, Action Chunk)               │
│                     • Chunk Horizon: Ta = 16 steps                     │
└───────────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        MODEL ARCHITECTURE (TRAINING)                   │
│                                                                        │
│   Ground Truth Actions [B, 16, 4] ──► Inject Noise (DDPM Schedule)     │
│                                                     │                  │
│                                                     ▼                  │
│   Sensory Obs ──► [ Condition Encoder ] ──► [ Noisy Actions Xt ]       │
│   (Kin / RGB)     (MLP or ConvNet)                  │                  │
│                           │                         ▼                  │
│                           └──── FiLM Layers ──► [ 1D Temporal U-Net ]  │
│                                                     │                  │
│                                                     ▼                  │
│                                              Predicted Noise ϵ_θ       │
│                                                     │                  │
│                                                     ▼                  │
│                                              Loss = MSE(ϵ, ϵ_θ)        │
└───────────────────────────────────────┬────────────────────────────────┘
                                        │
                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     DEPLOYMENT (RECEDING HORIZON)                      │
│                                                                        │
│   Current State Obs ──► Sample Gaussian Noise X_T ~ N(0, I)            │
│                                  │                                     │
│                                  ▼                                     │
│                       [ DDIM Fast Denoising ]                          │
│                       • 10 Sampling Steps (<50ms budget)               │
│                                  │                                     │
│                                  ▼                                     │
│                       Clean Action Chunk [16, 4]                       │
│                                  │                                     │
│                       • Execute first 4 steps (Te = 4)                 │
│                       • Discard tail & re-plan at 20 Hz                │
└────────────────────────────────────────────────────────────────────────┘
```

## Component Breakdown

### 1. Configuration Engine (configs/default.yaml)
A centralized YAML/Hydra configuration schema that eliminates hardcoded constants. It governs:
* **Horizons**: Observation horizon ($T_o$), action prediction horizon ($T_a = 16$), and execution horizon ($T_e = 4$).
* **Diffusion Parameters**: Training noise steps (100 DDPM steps), inference steps (10 DDIM steps), beta schedules ($\beta_{start}, \beta_{end}$).
* **Network Dimensions**: U-Net channel multipliers (128, 256, 512), kernel sizes, FiLM conditioning embedding dims (256-dim), and dropout rates.
* **Normalization Bounds**: Min/max clamping bounds for thrust and angular rates to ensure consistent tensor distribution in $[-1.0, 1.0]$.

### 2. Expert Trajectory Collector (dataset/collect_expert.py)
An automated synthetic data generation utility that bypasses manual human piloting:
* Spawns parameterized 3D flight paths (randomized waypoints, polynomial minimum-snap splines, obstacle slaloms).
* Drives the drone along trajectories using an analytic cascaded PID tracking controller.
* Logs high-frequency observations (linear velocities, angular velocities, orientation quaternions, positions, motor commands) directly to disk.

### 3. High-Throughput Chunk Storage (dataset/zarr_dataset.py)
A custom PyTorch Dataset backed by the Zarr hierarchical storage format:
* Stores massive multi-hour trajectory datasets in compressed chunked arrays rather than raw files or bloated CSVs.
* Generates sliding-window sequence slices on-the-fly: for any timestep $t$, it slices the observation at $t$ and pairs it with the future action chunk $[a_t, a_{t+1}, \dots, a_{t+15}]$.
* Keeps memory consumption minimal by streaming only requested batch slices directly into PyTorch tensors.

### 4. Observation Encoders (models/encoder.py)
Decoupled representation modules that condense multi-modal sensory inputs into a unified 1D latent conditioning vector $c$:
* **Kinematic Encoder**: Multi-layer perceptron (MLP) with Mish activations that projects a 20-dimensional kinematic state vector into a 256-dimensional latent representation.
* **Visual Encoder**: Lightweight ResNet-18 or custom 4-layer CNN backbone trained end-to-end to encode $224 \times 224$ synthetic RGB camera frames into feature vectors.

### 5. Conditional 1D Temporal U-Net (models/unet1d.py)
The primary generative engine that operates across the temporal dimension of the action sequence:
* **Downsampling & Upsampling Residual Blocks**: Uses 1D convolutions spanning the sequence length ($T_a = 16$) rather than spatial 2D convolutions.
* **Sinusoidal Timestep Embeddings**: Injects diffusion timestep $t \in [1, 100]$ via sinusoidal frequency projections.
* **FiLM (Feature-wise Linear Modulation)**: Dynamically scales and shifts intermediate convolution feature maps using the latent conditioning vector $c$ via affine transformations:
  $\text{FiLM}(h) = \gamma(c) \odot h + \beta(c)$

### 6. High-Level Policy Orchestrator (models/policy.py)
A unified `nn.Module` packaging the encoder, the 1D U-Net, and the Hugging Face diffusers scheduler into a single coherent interface:
* Exposes a standardized `.compute_loss(batch)` method for training.
* Exposes an optimized `.predict_action(obs)` method that accepts raw environment observations and outputs clean, denormalized continuous flight commands.

### 7. Training Loop (train.py)
The optimization script featuring standard generative diffusion objectives:
* Applies forward diffusion by sampling random Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$ and adding it to normalized ground-truth action chunks according to the DDPM variance schedule.
* Computes Mean Squared Error (MSE) between true noise $\epsilon$ and model-predicted noise $\epsilon_\theta$.
* Implements exponential moving average (EMA) model weight tracking to stabilize denoising dynamics.

### 8. Closed-Loop Inference Engine (deploy.py)
The real-time control loop executing Receding Horizon Control (RHC):
* Employs DDIM (Denoising Diffusion Implicit Models) to skip steps, condensing the denoising process to just 10 mathematical iterations.
* Generates a 16-step trajectory chunk in $<50\text{ms}$.
* Dispatches the initial 4 actions to the flight controller, discards the remaining 12 actions, and queries the policy again at 20 Hz to ensure low-latency responsiveness to physical disturbances.

## Implementation Phases

| Phase | Milestone | Core Deliverables | Technical Objectives |
|-------|-----------|-------------------|----------------------|
| Phase 1 | Configuration & Project Setup | `pyproject.toml`, `configs/default.yaml` | Define Hydra configuration hierarchies, schema validation, and hyperparameter structures. |
| Phase 2 | Data Generation & Dataset Engine | `collect_expert.py`, `zarr_dataset.py` | Implement autonomous PID spline navigation; construct low-latency Zarr chunk reader with temporal sequence slicing. |
| Phase 3 | Neural Architecture Construction | `encoder.py`, `unet1d.py`, `policy.py` | Build 1D temporal ResNet blocks, sinusoidal positional embeddings, and FiLM conditioning layers. |
| Phase 4 | Diffusion Training Pipeline | `train.py`, loss monitoring | Implement forward noise perturbation, MSE noise loss, EMA weight caching, and learning-rate warmup schedules. |
| Phase 5 | Receding Horizon Inference | `deploy.py`, latency benchmarks | Implement DDIM 10-step accelerated sampling, rolling execution window ($T_a=16 \to T_e=4$), and sub-50ms execution profiling. |

## Technical Specifications

| Parameter | Specification | Purpose |
|-----------|---------------|---------|
| Action Dimensions | 4 Continuous channels | Normalized Collective Thrust, Roll rate, Pitch rate, Yaw rate |
| Prediction Horizon ($T_a$) | 16 timesteps | Ensures temporal smoothness and trajectory consistency |
| Execution Horizon ($T_e$) | 4 timesteps | Enforces receding horizon replanning to counter aerodynamic drift |
| Training Schedule | DDPM (100 steps) | Stable convergence and noise profile optimization |
| Inference Schedule | DDIM (10 steps) | Sub-50ms deterministic sampling for real-time control |
| Conditioning Mode | Kinematic (20-dim) or RGB ($3 \times 224 \times 224$) | Adaptive multi-modal embedding passed via FiLM layers |
| Inference Frequency | 20 Hz | Action replanning cycle triggered every 50ms |
