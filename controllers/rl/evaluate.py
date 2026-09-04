import numpy as np
from stable_baselines3 import PPO
from ot2_env import OT2Env

NUM_EPISODES = 10
MODEL_PATH = "best_model.zip"

model = PPO.load(MODEL_PATH)
env = OT2Env(render=True, max_steps=1000)

print(f"Testing model: {MODEL_PATH}")
print(f"Episodes: {NUM_EPISODES}\n")

final_distances = []
episode_lengths = []
successes = []

for episode in range(NUM_EPISODES):
    obs, _ = env.reset()
    steps = 0

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        if terminated or truncated:
            dist_mm = info['distance_to_goal'] * 1000
            success = terminated  # terminated = goal reached, truncated = timeout
            final_distances.append(dist_mm)
            episode_lengths.append(steps)
            successes.append(success)
            status = "SUCCESS" if success else "TIMEOUT"
            print(f"Episode {episode + 1:2d} | {status} | Distance: {dist_mm:.2f} mm | Steps: {steps}")
            break

env.close()

print("\n--- Summary ---")
print(f"Success rate:       {100 * np.mean(successes):.1f}%")
print(f"Mean final distance: {np.mean(final_distances):.2f} mm")
print(f"Mean steps:         {np.mean(episode_lengths):.1f}")