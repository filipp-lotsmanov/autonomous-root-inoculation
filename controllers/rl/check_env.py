from stable_baselines3.common.env_checker import check_env
from ot2_env import OT2Env

env = OT2Env()
check_env(env)  # Verifies gym compliance
print("Wrapper passed checks!")