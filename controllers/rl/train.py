import argparse
import os
from datetime import datetime
import numpy as np
from clearml import Task

# ============================================================================
# CONFIGURATION
# ============================================================================
RUN_LABEL = os.environ.get("OT2_RUN_LABEL", "ot2")

# Generate timestamp for unique task name and model filename
timestamp = datetime.now().strftime("%y%m%d.%H%M")

# ============================================================================
# ClearML Setup
#
# Training runs are submitted to a ClearML server, which must be configured
# separately (`clearml-init`). All deployment-specific settings come from the
# environment so that no account, repository or project is hardcoded here.
#   CLEARML_PROJECT  target project name
#   CLEARML_QUEUE    execution queue
#   CLEARML_DOCKER   base image providing the OT-2 simulation
#   CLEARML_REPO     optional git repo for the agent to clone
#   CLEARML_BRANCH   optional branch within CLEARML_REPO
# ============================================================================
CLEARML_PROJECT = os.environ.get("CLEARML_PROJECT", "OT2-RL")
CLEARML_QUEUE = os.environ.get("CLEARML_QUEUE", "default")
CLEARML_DOCKER = os.environ.get("CLEARML_DOCKER", "deanis/2023y2b-rl:latest")
CLEARML_REPO = os.environ.get("CLEARML_REPO")
CLEARML_BRANCH = os.environ.get("CLEARML_BRANCH")

task_name = f'OT2_RL_{RUN_LABEL}_{timestamp}'

task = Task.init(
    project_name=CLEARML_PROJECT,
    task_name=task_name,
)

if CLEARML_REPO:
    task.set_repo(repo=CLEARML_REPO, branch=CLEARML_BRANCH)

task.set_base_docker(CLEARML_DOCKER)

# CRITICAL: Install tensorboard and clearml
task.set_packages(['tensorboard', 'clearml'])

# ============================================================================
# Command Line Arguments
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0003)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--total_timesteps", type=int, default=500000)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--max_steps_truncate", type=int, default=300)
parser.add_argument("--target_threshold", type=float, default=0.005)
parser.add_argument("--reward_type", type=str, default='normalized_progress',
                    choices=['normalized_progress', 'exponential', 'staged',
                             'dense_shaping', 'energy_efficient'],
                    help='Reward function type to use')
args = parser.parse_args()

# Execute remotely
task.execute_remotely(queue_name=CLEARML_QUEUE)

# ============================================================================
# ML IMPORTS (AFTER execute_remotely)
# ============================================================================
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Import wrapper
from ot2_env import OT2Env


# ============================================================================
# Custom Callback for OT2 Metrics
# ============================================================================
class OT2Callback(BaseCallback):
    """
    Callback for logging OT2-specific metrics during training.
    """

    def __init__(self, threshold=0.005, verbose=0):
        super().__init__(verbose)
        self.threshold = threshold
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_final_distances = []

    def _on_step(self) -> bool:
        """Called after each step in all environments"""
        dones = self.locals.get('dones', [])

        for i, done in enumerate(dones):
            if done:
                infos = self.locals.get('infos', [])
                if i < len(infos):
                    info = infos[i]

                    # Extract metrics
                    final_dist = info.get('distance_to_goal', float('inf'))

                    # Get episode info from SB3
                    ep_info = info.get('episode')
                    if ep_info is not None:
                        ep_reward = ep_info['r']
                        ep_length = ep_info['l']

                        # Store metrics
                        self.episode_rewards.append(ep_reward)
                        self.episode_lengths.append(ep_length)

                        success = float(final_dist < self.threshold)
                        self.episode_successes.append(success)
                        self.episode_final_distances.append(final_dist)

                        # Log to tensorboard
                        self.logger.record('ot2/episode_reward', ep_reward)
                        self.logger.record('ot2/episode_length', ep_length)
                        self.logger.record('ot2/final_distance_mm', final_dist * 1000)
                        self.logger.record('ot2/success', success)

                        # Rolling averages
                        if len(self.episode_successes) >= 10:
                            window = min(100, len(self.episode_successes))
                            self.logger.record('ot2/success_rate_100ep',
                                               np.mean(self.episode_successes[-window:]))
                            self.logger.record('ot2/avg_length_100ep',
                                               np.mean(self.episode_lengths[-window:]))
                            self.logger.record('ot2/avg_final_dist_mm_100ep',
                                               np.mean(self.episode_final_distances[-window:]) * 1000)

        return True

    def _on_training_end(self) -> None:
        """Print summary at end of training"""
        if len(self.episode_successes) > 0:
            print("\n" + "=" * 60)
            print("TRAINING SUMMARY")
            print("=" * 60)
            print(f"Total episodes: {len(self.episode_successes)}")
            print(f"Success rate: {100 * np.mean(self.episode_successes):.1f}%")
            print(f"Average episode length: {np.mean(self.episode_lengths):.1f} steps")
            print(f"Average final distance: {1000 * np.mean(self.episode_final_distances):.3f} mm")

            successful_lengths = [length for length, ok
                                  in zip(self.episode_lengths, self.episode_successes) if ok]
            if successful_lengths:
                print(f"Successful episodes avg length: {np.mean(successful_lengths):.1f} steps")

            print("=" * 60)


# ============================================================================
# Generate Filename
# ============================================================================
def format_lr(lr):
    """Convert learning rate to scientific notation for filename"""
    return f"{lr:.0e}".replace("+", "").replace("-0", "-")


lr_str = format_lr(args.learning_rate)
filename = f"{timestamp}_{RUN_LABEL}_lr{lr_str}_b{args.batch_size}_s{args.n_steps}_reward{args.reward_type}"

print("=" * 60)
print("Training Configuration:")
print(f"  Run label: {RUN_LABEL}")
print(f"  Learning Rate: {args.learning_rate}")
print(f"  Batch Size: {args.batch_size}")
print(f"  N Steps: {args.n_steps}")
print(f"  Total Timesteps: {args.total_timesteps:,}")
print(f"  Max Episode Steps: {args.max_steps_truncate}")
print(f"  Target Threshold: {args.target_threshold * 1000:.1f}mm")
print(f"  Model Name: {filename}")
print("=" * 60)

# ============================================================================
# Environment Setup
# ============================================================================
env = OT2Env(
    render=False,
    max_steps=args.max_steps_truncate,
    target_threshold=args.target_threshold,
    reward_type=args.reward_type  # ADD THIS LINE
)

# ============================================================================
# Model Setup
# ============================================================================
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=10,
    gamma=args.gamma,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    tensorboard_log=f"runs/{RUN_LABEL}/{args.reward_type}"
)

# ============================================================================
# Training
# ============================================================================
ot2_callback = OT2Callback(threshold=args.target_threshold, verbose=1)

model.learn(
    total_timesteps=args.total_timesteps,
    callback=ot2_callback,
    tb_log_name=f"PPO_{filename}"
)

# ============================================================================
# Save and Upload Model
# ============================================================================
model_name = f"{filename}.zip"
model.save(model_name)
print(f"\nModel saved: {model_name}")

task.upload_artifact("model", artifact_object=model_name)
print(f"Artifact uploaded: {model_name}")

print("\nTraining complete!")

# Close environment
try:
    env.close()
except Exception as error:
    print(f"Environment close failed: {error}")