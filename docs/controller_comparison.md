# Reinforcement Learning Controller — OT-2 Robot

**Author:** Filipp Lotsmanov  
**Algorithm:** PPO (Proximal Policy Optimization)  
**Best Reward Function:** `normalized_progress`  
**Final Success Rate:** ~99% at 1mm threshold  

---

## Table of Contents

1. [Overview](#overview)
2. [Implementation](#implementation)
3. [Reward Functions](#reward-functions)
4. [Training](#training)
5. [Individual vs Team Implementation Comparison](#individual-vs-team-implementation-comparison)
6. [Performance Metrics](#performance-metrics)
7. [Comparison with PID Controller](#comparison-with-pid-controller)
8. [Known Bugs and Limitations](#known-bugs-and-limitations)
9. [Libraries Used](#libraries-used)
10. [How to Run](#how-to-run)

---

## Overview

This task implements a Reinforcement Learning (RL) controller using Stable Baselines 3 to control the Opentrons OT-2 pipette robot. The agent is trained to move the pipette tip to any random target position within the working envelope with sub-millimetre precision.

The individual contribution focuses on a systematic comparison of five distinct reward functions, identifying `normalized_progress` as the best-performing design and achieving ~99% success rate at the 1mm precision threshold after 2 million training timesteps.

---

## Implementation

### Gymnasium Wrapper (`ot2_env.py`)

The wrapper subclasses `gym.Env` and exposes the OT-2 PyBullet simulation as a standard Gymnasium environment compatible with Stable Baselines 3.

**Observation Space:** 6D continuous box `[-1, 1]`
- `[current_x, current_y, current_z, goal_x, goal_y, goal_z]`
- All positions normalized to `[-1, 1]` relative to workspace bounds using:
  ```
  normalized = 2 * (pos - workspace_low) / (workspace_high - workspace_low) - 1
  ```
- `np.clip(..., -1.0, 1.0)` applied after normalization to prevent observation space violations from physics jitter

**Action Space:** 3D continuous box `[-1, 1]`
- Normalized velocity commands `[vx, vy, vz]`
- Scaled by `max_velocity=2.0` to `[-2, 2] m/s` before passing to the simulation
- Drop action always set to `0` (pipetting not used in this task)

**Workspace Bounds:**
```
X: [-0.1871,  0.2532] m
Y: [-0.1706,  0.2197] m
Z: [ 0.1700,  0.2897] m
```

**Termination Conditions:**
- `terminated = True` when Euclidean distance to goal < `target_threshold` (default 1mm)
- `truncated = True` when episode steps ≥ `max_steps` (default 250)

**State Tracking per Episode:**
- `initial_distance` — distance from spawn to goal at episode start; used to normalize progress rewards
- `previous_distance` — distance at previous step; used to compute per-step progress
- `previous_action` — action at previous step; used by `energy_efficient` reward for smoothness penalty
- `achieved_stages` — set of milestone indices already reached; used by `staged` reward to prevent double-counting
- `cumulative_reward` — running episode reward total; logged in `info` dict

---

## Reward Functions

Five reward functions were implemented and systematically compared. All are selectable via the `reward_type` parameter in the wrapper and training scripts.

### 1. Normalized Progress — Best Performing

```
reward = progress_scale * (prev_dist - curr_dist) / initial_dist
       - time_penalty
       + success_bonus + efficiency_bonus  (if goal reached)
```

**Parameters:** `progress_scale=300`, `time_penalty=0.05`, `success_bonus=200`, `efficiency_multiplier=1.0`

**Why it won:** Normalizing progress by `initial_distance` makes the reward signal scale-invariant across episodes. Whether the agent starts 5cm or 20cm from the goal, a 1mm step forward yields the same reward magnitude. This produces consistent gradient updates throughout training. The large success bonus (200) makes the terminal condition strongly dominant, preventing the agent from loitering near the threshold without crossing it. The efficiency bonus (remaining steps × multiplier) additionally incentivizes speed.

**Result:** ~99% success rate, ~50 steps mean episode length at convergence.

---

### 2. Exponential (tanh-based)

```
normalized_dist = distance / initial_distance
reward = -scale * (1 - tanh(alpha * (1 - normalized_dist)))
       - time_penalty
       + success_bonus  (if goal reached)
```

**Parameters:** `alpha=5.0`, `scale=100`, `time_penalty=0.05`, `success_bonus=250`

`tanh` was used instead of raw `exp` to prevent numerical explosion as distance approaches zero. Despite this, the gradient is very flat when the agent is far from the goal, producing small policy updates during early training and slowing initial convergence.

---

### 3. Staged Milestones

One-time bonuses awarded the first time the agent crosses each distance threshold:

| Threshold | Bonus |
|-----------|-------|
| 50mm | +20 |
| 20mm | +40 |
| 5mm | +80 |
| 2mm | +120 |
| 1mm | +200 |

A continuous term `(-0.01 × distance)` was added to maintain gradient signal between milestone crossings. Without this the agent has no learning signal between thresholds, causing stagnant policy updates.

**Why it underperformed:** Even with the continuous term, the reward is dominated by sparse milestone events. Between crossings the agent has weak incentive to move toward the goal, causing inconsistent training and slow convergence.

---

### 4. Dense Potential Shaping

```
reward = Phi(s') - Phi(s)  where Phi(s) = -shaping_scale * distance
       - time_penalty
       + success_bonus  (if goal reached)
```

**Parameters:** `shaping_scale=300`, `time_penalty=0.05`, `success_bonus=200`

Theoretically sound — potential-based shaping is policy-invariant. Expanding the potential difference gives `shaping_scale * (prev_dist - curr_dist)`, which is the same per-step progress signal as `normalized_progress` without the `1 / initial_distance` factor. The two differ by that scale term alone.

That term matters because `initial_distance` varies from roughly 0.05 m to 0.45 m across episodes. Under normalization, an episode that ends at the goal accumulates about `shaping_scale` in total progress reward regardless of where it started. Without it, the same trajectory is worth ~135 when the spawn is far from the goal and ~15 when it is close. The critic therefore has to fit a return whose scale swings by an order of magnitude for reasons unrelated to policy quality, which raises advantage variance, and in the near-spawn case the fixed `success_bonus=200` dominates the shaped signal entirely — leaving credit assignment closer to sparse than dense.

> **Note on the published implementation.** `ot2_env.py` was revised after these experiments were run — the reward guards and the potential-based form above reflect the current code. The ~90% figure reported below was produced by an earlier variant using a raw negative-distance penalty rather than the potential difference. The ranking of the five reward functions was not re-measured against the revised code.

---

### 5. Energy Efficient

Extends normalized progress with penalties for action magnitude and jerk:

```
reward = progress_scale * progress
       - action_penalty * ||action||²
       - smoothness_penalty * ||action - prev_action||²
       - time_penalty
       + success_bonus  (if goal reached)
```

**Parameters:** `progress_scale=300`, `action_penalty=0.02`, `smoothness_penalty=0.02`

**Why it underperformed:** Even small action penalties conflict with the progress signal during early training when the agent needs to move aggressively to explore the workspace. Penalties were reduced from 0.1 to 0.02 during development, which improved but did not fully resolve the issue.

---

### Reward Function Summary

| Reward Type | Convergence Speed | Final Success Rate | Key Weakness |
|---|---|---|---|
| Normalized Progress | Fast (~750k steps) | ~99% | None identified |
| Dense Shaping | Fast | ~90% | Weaker early signal |
| Exponential | Medium | ~90–95% | Flat gradient far from goal |
| Energy Efficient | Medium | ~88% | Action penalties slow exploration |
| Staged | Slow | ~85% | Sparse signal between milestones |


### Why Normalized Progress Won

`normalized_progress` outperformed every other reward function because it provides a dense, scale-invariant gradient signal at every step of every episode.

Compared to each alternative:

- **vs Dense Shaping:** Both reward the same per-step distance reduction; dense shaping omits the `1 / initial_distance` factor. That leaves the total shaped return proportional to how far the episode happened to start from the goal, so the critic fits a target that varies by roughly an order of magnitude across episodes for reasons unrelated to the policy. Normalizing makes an episode's total progress reward scale-invariant.
- **vs Exponential:** The tanh-based exponential produces a near-flat gradient when the agent is far from the goal, meaning the policy receives almost no useful update signal during the critical initial exploration phase. Normalized progress is linear in improvement, providing strong updates from step one.
- **vs Staged:** Staged rewards are fundamentally sparse — the agent only receives meaningful signal when crossing a threshold. Between crossings the policy trains blind. Normalized progress gives a reward at every step proportional to actual progress made.
- **vs Energy Efficient:** Energy efficient adds action penalties on top of normalized progress. These penalties directly oppose the aggressive exploration needed early in training when the agent must move quickly to discover the goal region. The penalties slow initial learning without benefit until the agent is already performing well.

In short, `normalized_progress` combines a dense per-step signal, normalization by episode-specific initial distance, a large success bonus (200), and an efficiency bonus for speed. None of the other four designs cover all of those at once.

---

## Training

### Algorithm

PPO (Proximal Policy Optimization) via Stable Baselines 3 with an `MlpPolicy`. PPO was chosen as the group's shared algorithm to enable fair comparison across hyperparameter experiments.

### Infrastructure

- GPU server: RTX A6000 / RTX 6000 Ada / L40S (48GB VRAM)
- Job submission: ClearML task queue (`default` queue)
- Experiment tracking: ClearML + TensorBoard
- Docker base image: `deanis/2023y2b-rl:latest`
- ClearML project: University experiment tracking server

### Best Individual Hyperparameters

| Hyperparameter | Value | Justification |
|---|---|---|
| Algorithm | PPO | Group standard; stable for continuous control |
| Learning Rate | 0.001 | Converges ~250k steps faster than 0.0003 at same final performance |
| Batch Size | 128 | Standard for continuous control |
| N Steps | 2048 | Sufficient trajectory length for credit assignment over 250-step episodes |
| N Epochs | 10 | Default PPO; stable updates |
| Gamma | 0.99 | Long-horizon discounting appropriate for 250-step episodes |
| GAE Lambda | 0.95 | Default PPO; good bias-variance trade-off |
| Clip Range | 0.2 | Default PPO |
| Total Timesteps | 2,000,000 | Required for convergence at 1mm threshold |
| Max Episode Steps | 250 | ~4x typical success length; sufficient headroom |
| Target Threshold | 0.001 m | 1mm precision requirement |
| Reward Type | normalized_progress | Best empirical performance across all 5 tested |

### Hyperparameter Search

Each group member tested assigned hyperparameter combinations. Individual experiments varied:
- Learning rate: `[0.0003, 0.001]`
- Target threshold: `[0.005m, 0.003m, 0.001m]`
- Total timesteps: `[500k, 750k, 2000k]`
- Reward function: all 5 types (individual contribution)

Results were tracked on the group ClearML leaderboard comparing success rate, final distance, and episode length metrics.

---

## Individual vs Team Implementation Comparison

### Key Differences

| Aspect | Team Implementation | Individual (Filipp) | Better |
|---|---|---|---|
| Z workspace lower bound | `0.1195` (wrong) | `0.1700` (correct) | Filipp |
| Observation clipping | Missing | `np.clip(-1, 1)` applied | Filipp |
| Reward functions | 1 (static bonuses) | 5 (systematic comparison) | Filipp |
| Progress tracking | No `previous_distance` | `previous_distance` + `initial_distance` | Filipp |
| Division by zero guard | Not applicable | `initial_distance < 1e-6` guard | Filipp |
| Continuous staged signal | Not applicable | `0.01 × distance` base term | Filipp |
| Evaluation script | Single episode | 10 episodes with summary stats | Filipp |
| `check_env` validation | No | Yes | Filipp |
| Code documentation | Thorough docstrings | Inline comments | Team |

### Why the Individual Implementation is Better

**1. Correct workspace bounds.** The team's `workspace_low[2] = 0.1195` was identified as wrong by the team lead on 22 December 2025. This value includes ~18% of the Z range that the simulator physically cannot reach. Goals generated in this region are impossible to achieve, corrupting the training distribution. The individual implementation used `0.1700` from the start.

**2. Observation clipping.** The team's `bug_fix_comparison.png` proves this matters — without clipping, the 5mm threshold model plateaued at ~80% success rate and was unstable. With clipping, 99%+ was achieved cleanly. The individual wrapper includes this fix from the initial version.

**3. Superior reward design.** The team's single reward function used static proximity bonuses (+2.0 at 10mm, +1.0 at 20mm), which create artificial plateaus where the agent can collect reward by hovering near a boundary rather than converging. The individual implementation uses progress normalized by `initial_distance`, providing clean gradient signal throughout each episode with no plateau risk.

**4. Systematic experimentation.** Testing 5 reward functions with identical hyperparameters provides scientific evidence for the design choice rather than a single untested assumption.

---

## Performance Metrics

### Individual Best Model

| Metric | Value |
|---|---|
| Algorithm | PPO |
| Reward function | normalized_progress |
| Training timesteps | 2,000,000 |
| Success rate | ~99% |
| Mean episode length at convergence | ~50 steps |
| Approximate time to target | ~0.21s at 240Hz simulation |
| Final positioning error | < 1mm |

### Agent Demonstration

Demo GIFs of the integrated system (including RL-informed positioning) are in `results/demos/`.

### Model Weights

| Model | File | Threshold | Success Rate |
|---|---|---|---|
| Individual Best | `best_model.zip` | 1mm | ~99% |

Available in [GitHub Releases](https://github.com/filipp-lotsmanov/autonomous-root-inoculation/releases).

---

## Comparison with PID Controller

### Methodology Note

This comparison is not perfectly apples-to-apples. The PID controller was evaluated on a theoretical `VelocityRobotSimulator` with modelled friction and velocity lag dynamics. RL was evaluated in the actual PyBullet simulator. Dynamics differ between the two and this should be considered when interpreting numerical comparisons.

### Performance Comparison

| Metric | PID (theoretical model) | RL — normalized_progress | Winner |
|---|---|---|---|
| Positioning error (X/Y) | 0.031 mm | < 1mm | PID |
| Positioning error (Z) | 0.011 mm | < 1mm | PID |
| Settling / convergence time | ~5.0–5.28 s | ~0.21 s (~50 steps @ 240Hz) | RL |
| Success rate | ~100% (deterministic) | ~99% | PID |
| Overshoot | 13.9–15.3% | None | RL |
| Requires manual tuning | Yes (per-axis grid search) | No (learned) | RL |
| Deterministic output | Yes | No | PID |
| Interpretability | High | Low (black box) | PID |

### PID Gains (theoretical model, for reference)

| Axis | Kp | Ki | Kd | Steady-State Error | Settling Time |
|---|---|---|---|---|---|
| X | 3.0 | 1.8 | 0.6 | 0.031 mm | 5.28 s |
| Y | 3.0 | 1.8 | 0.6 | 0.031 mm | 5.28 s |
| Z | 3.5 | 2.29 | 0.6 | 0.011 mm | 5.00 s |

### Why PID is the Better Controller for This Task

For this particular task — precision pipette positioning — PID makes more sense:

**1. Accuracy margin.** PID achieves 0.011–0.031mm steady-state error — 30 to 90 times more accurate than the RL agent's 1mm threshold. In laboratory automation where dispensing into specific wells requires exact placement, this margin is critical.

**2. Determinism.** PID produces identical trajectories for identical targets every time. The RL agent's stochastic policy can vary between runs, which is undesirable when reproducibility matters in scientific experiments.

**3. RL's speed advantage is less impactful here.** The OT-2 protocol involves aspiration, dispensing, and tip changes — controller settling time is a small fraction of total operation time. The 5-second PID settling vs 0.21-second RL settling rarely affects overall throughput meaningfully.

**4. RL at 1mm is not production-ready.** The current training budget achieves the minimum passing threshold. Reaching 0.1mm or better would require curriculum learning (5mm → 3mm → 1mm → 0.1mm) and significantly more compute.

**RL's future potential:** With curriculum learning, longer training, and a tighter threshold, RL could match PID precision while retaining its speed advantage and requiring no manual tuning. At the current stage, PID remains the better practical controller.

PID was selected for the integration pipeline. The 30–90x accuracy advantage is what decided it — in a lab setting where you need to hit a specific root tip, consistent sub-0.05mm error matters more than getting there faster.

---

## Known Bugs and Limitations

### Bug 1 — Z Workspace Lower Bound (Resolved)

**Issue:** A simulator initialization bug causes the pipette's initial Z position to be outside the actual working envelope, yielding two possible minimum Z values: `0.1195` and `0.1700`.

**Impact:** Using `0.1195` includes ~18% of Z positions the simulation cannot physically reach. Goals generated there are unreachable, corrupting the training distribution. Identified and announced by team lead a team member on 22 December 2025.

**Resolution:** Used `workspace_low[2] = 0.1700` from the start. Both values are officially accepted for grading per instructor announcement.

---

### Bug 2 — Observation Space Violation Without Clipping (Resolved)

**Issue:** PyBullet physics can return pipette positions slightly outside declared workspace bounds due to momentum and constraint solving. Without clipping, normalized observations fall outside `[-1, 1]`, violating the declared `observation_space`.

**Impact:** Proven significant by team's `bug_fix_comparison.png` — unclipped version plateaued at ~80% success rate; clipped version reached 99%+.

**Resolution:** Applied `np.clip(normalized, -1.0, 1.0)` in `_normalize_position`.

---

### Bug 3 — Division by Zero in Progress Rewards (Resolved)

**Issue:** `normalized_progress` and `energy_efficient` divide by `initial_distance`. If the agent spawns exactly on the goal, `initial_distance = 0` causes a `ZeroDivisionError`.

**Impact:** Practically zero — probability of spawning within 1mm of a uniformly random goal across a ~0.09m³ workspace is negligible. No crash occurred during any training run.

**Resolution:** Added guard: `if self.initial_distance < 1e-6: progress = 0.0`.

---

### Bug 4 — Staged Reward Lacks Continuous Signal (Resolved)

**Issue:** Between milestone threshold crossings, the `staged` reward provides only a constant time penalty with no gradient toward the goal.

**Impact:** Explains why `staged` converged slowest and achieved the lowest success rate (~85%). No impact on submitted results since `normalized_progress` was the best model.

**Resolution:** Added `reward -= 0.01 * distance_to_goal` as a continuous base term.

---

### Bug 5 — Single Episode Evaluation (Resolved)

**Issue:** Original `evaluate.py` ran only a single episode due to a `break` inside the step loop on the first terminal state.

**Impact:** One episode is insufficient to demonstrate consistent performance across random goal positions.

**Resolution:** Replaced with a 10-episode loop reporting per-episode status and a summary of success rate, mean final distance, and mean steps.

---

### Limitation 1 — Single Algorithm Tested

Only PPO was evaluated individually. Off-policy algorithms (SAC, TD3) typically require fewer environment steps for continuous control and may achieve better sample efficiency. This was a compute and time constraint given the 1–2 day training requirement per run.

---

### Limitation 2 — No EvalCallback

Training scripts save only the final model, not the best model during training. If PPO performance degrades late in training (possible at `lr=0.001`), the saved weights may not represent peak performance. An `EvalCallback` with `save_best_model=True` would address this in future work.

---

### Limitation 3 — PID Comparison Uses Different Simulator

PID metrics come from a theoretical `VelocityRobotSimulator` with modelled friction and velocity lag. RL metrics come from the actual PyBullet simulator. Direct numerical comparison should account for this difference.

---

### Limitation 4 — Global RNG Seed in reset()

`reset()` uses `np.random.seed(seed)` which sets the global NumPy RNG rather than the Gymnasium-local `self.np_random` from `super().reset(seed=seed)`. No practical impact with a single environment instance, but would break reproducibility in vectorized environments (`VecEnv`).

---

## Libraries Used

| Library | Version | Purpose |
|---|---|---|
| Stable Baselines 3 | 2.2.1 | PPO implementation, callbacks, logging |
| Gymnasium | 0.29.1 | Environment interface and API compliance |
| PyTorch | 2.1.2+cu121 | Neural network backend for PPO policy |
| NumPy | 1.26.2 | Array operations, normalization, distance calculations |
| ClearML | ≥2.1.2 | Experiment tracking, remote job submission, artifact storage |
| TensorBoard | ≥2.0.0 | Training curve logging |
| PyBullet | ≥3.2.7 | Physics simulation (via `sim_class.py`) |

**System:** Python 3.10.12, Ubuntu 22.04 LTS, CUDA 12.1, GPU-enabled

---

## How to Run

### Install Dependencies
```bash
uv sync
```

### Verify Gymnasium Compliance
```bash
uv run python validate_gains.py
```

### Run 1000 Random Action Steps
```bash
uv run python test_wrapper.py
```

### Train with Specific Reward Function
```bash
uv run python train_normalized_progress.py
```
Or via the general script with CLI arguments:
```bash
uv run python train.py \
  --reward_type normalized_progress \
  --learning_rate 0.001 \
  --total_timesteps 2000000 \
  --target_threshold 0.001
```

### Evaluate Trained Model (10 Episodes)
```bash
uv run python evaluate.py
```
