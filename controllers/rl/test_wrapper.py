"""
Smoke test for the OT-2 Gymnasium wrapper.

Runs SB3's env checker, then steps the environment 1000 times with random
actions to verify the wrapper behaves correctly without a trained model.
This is a manual script, not part of the pytest suite in tests/.
"""
from ot2_env import OT2Env
from stable_baselines3.common.env_checker import check_env

def main():
    env = OT2Env(render=False, max_steps=1000)

    # Verify Gymnasium API compliance before running
    print("Running check_env...")
    check_env(env)
    print("check_env passed.\n")

    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}\n")

    total_steps = 1000
    episodes = 0
    successes = 0

    for step in range(1, total_steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            dist = info.get('distance_to_goal', float('nan'))
            print(f"Step {step:4d}/{total_steps} | "
                  f"Distance: {dist*1000:.2f} mm | "
                  f"Reward: {reward:.4f}")

        if terminated or truncated:
            episodes += 1
            if terminated:
                successes += 1
            obs, info = env.reset()

    env.close()

    print("\n--- Test Complete ---")
    print(f"Total steps:    {total_steps}")
    print(f"Episodes:       {episodes}")
    print(f"Goal reached:   {successes}")

if __name__ == "__main__":
    main()
