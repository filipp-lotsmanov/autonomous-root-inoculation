"""
OT2 Gym Environment - Fixed and Optimized Reward Functions
Properly scaled for 1mm precision threshold
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation


class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=250, target_threshold=0.001, 
                 reward_type='normalized_progress', reward_params=None):
        super(OT2Env, self).__init__()
        
        self.render_mode = render
        self.max_steps = max_steps
        self.target_threshold = target_threshold
        self.reward_type = reward_type
        
        self.sim = Simulation(num_agents=1, render=render)
        
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32
        )
        
        self.workspace_low = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        self.workspace_high = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        self.reward_params = self._get_default_reward_params(reward_type)
        if reward_params is not None:
            self.reward_params.update(reward_params)
        
        self.steps = 0
        self.goal_position = None
        self.initial_distance = None
        self.previous_distance = None
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.achieved_stages = set()
        self.cumulative_reward = 0.0
    
    def reset(self, seed=None, options=None):
        # Seed through the Gymnasium base class so sampling uses this env's own
        # self.np_random rather than the process-wide NumPy RNG.
        super().reset(seed=seed)

        self.goal_position = self.np_random.uniform(
            self.workspace_low,
            self.workspace_high
        ).astype(np.float32)
        
        state_dict = self.sim.reset(num_agents=1)
        current_pos = self._extract_position(state_dict)
        
        self.initial_distance = float(np.linalg.norm(current_pos - self.goal_position))
        self.previous_distance = self.initial_distance
        
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        self.steps = 0
        self.previous_action = np.zeros(3, dtype=np.float32)
        self.achieved_stages = set()
        self.cumulative_reward = 0.0
        
        return observation, {}
    
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        
        max_velocity = 2.0
        velocity = action * max_velocity
        full_action = [float(velocity[0]), float(velocity[1]), float(velocity[2]), 0.0]
        
        state_dict = self.sim.run([full_action])
        current_pos = self._extract_position(state_dict)
        distance_to_goal = float(np.linalg.norm(current_pos - self.goal_position))
        
        reward = self._calculate_reward(distance_to_goal, action)
        self.cumulative_reward += reward
        
        terminated = bool(distance_to_goal < self.target_threshold)
        self.steps += 1
        truncated = bool(self.steps >= self.max_steps)
        
        self.previous_distance = distance_to_goal
        self.previous_action = action.copy()
        
        observation = np.concatenate([
            self._normalize_position(current_pos),
            self._normalize_position(self.goal_position)
        ], dtype=np.float32)
        
        info = {
            'distance_to_goal': distance_to_goal,
            'current_position': current_pos.tolist(),
            'goal_position': self.goal_position.tolist(),
            'cumulative_reward': self.cumulative_reward,
            'reward_type': self.reward_type
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self, distance_to_goal, action):
        if self.reward_type == 'normalized_progress':
            return self._reward_normalized_progress(distance_to_goal, action)
        elif self.reward_type == 'exponential':
            return self._reward_exponential(distance_to_goal)
        elif self.reward_type == 'staged':
            return self._reward_staged(distance_to_goal)
        elif self.reward_type == 'dense_shaping':
            return self._reward_dense_shaping(distance_to_goal)
        elif self.reward_type == 'energy_efficient':
            return self._reward_energy_efficient(distance_to_goal, action)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _reward_normalized_progress(self, distance_to_goal, action):
        """
        Progress reward normalized by initial distance.
        Guard against division by zero if spawn is on goal.
        """
        params = self.reward_params

        # FIX: guard division by zero
        if self.initial_distance < 1e-6:
            progress = 0.0
        else:
            progress = (self.previous_distance - distance_to_goal) / self.initial_distance
        progress_reward = params['progress_scale'] * progress

        time_penalty = -params['time_penalty']

        if distance_to_goal < self.target_threshold:
            remaining_steps = self.max_steps - self.steps
            efficiency_bonus = remaining_steps * params['efficiency_multiplier']
            success_bonus = params['success_bonus'] + efficiency_bonus
        else:
            success_bonus = 0.0

        reward = progress_reward + time_penalty + success_bonus
        return float(reward)

    def _reward_exponential(self, distance_to_goal):
        """
        Uses tanh instead of exp to prevent numerical explosion.
        Guard against division by zero if spawn is on goal.
        """
        params = self.reward_params

        # FIX: guard division by zero
        if self.initial_distance < 1e-6:
            normalized_distance = 0.0
        else:
            normalized_distance = distance_to_goal / self.initial_distance

        distance_reward = -params['scale'] * (1.0 - np.tanh(params['alpha'] * (1.0 - normalized_distance)))

        time_penalty = -params['time_penalty']

        success_bonus = params['success_bonus'] if distance_to_goal < self.target_threshold else 0.0

        reward = distance_reward + time_penalty + success_bonus
        return float(reward)

    def _reward_staged(self, distance_to_goal):
        """
        One-time bonuses at progressive distance thresholds.
        Small continuous distance term keeps gradient between milestones.
        """
        params = self.reward_params
        reward = 0.0

        for i, threshold in enumerate(params['thresholds']):
            if distance_to_goal < threshold and i not in self.achieved_stages:
                self.achieved_stages.add(i)
                reward += params['bonuses'][i]

        reward -= params['time_penalty']
        # FIX: continuous signal between milestones so agent isn't gradient-blind
        reward -= 0.01 * distance_to_goal

        return float(reward)

    def _reward_dense_shaping(self, distance_to_goal):
        """
        Potential-based reward shaping: F = gamma * Phi(s') - Phi(s).
        """
        params = self.reward_params

        current_potential = -params['shaping_scale'] * distance_to_goal
        previous_potential = -params['shaping_scale'] * self.previous_distance
        shaping_reward = current_potential - previous_potential

        time_penalty = -params['time_penalty']

        success_bonus = params['success_bonus'] if distance_to_goal < self.target_threshold else 0.0

        reward = shaping_reward + time_penalty + success_bonus
        return float(reward)

    def _reward_energy_efficient(self, distance_to_goal, action):
        """
        Progress-primary with small penalties for action magnitude and jerk.
        Guard against division by zero if spawn is on goal.
        """
        params = self.reward_params

        # FIX: guard division by zero
        if self.initial_distance < 1e-6:
            progress = 0.0
        else:
            progress = (self.previous_distance - distance_to_goal) / self.initial_distance
        progress_reward = params['progress_scale'] * progress

        action_magnitude = np.linalg.norm(action)
        action_penalty = -params['action_penalty'] * (action_magnitude ** 2)

        action_change = np.linalg.norm(action - self.previous_action)
        smoothness_penalty = -params['smoothness_penalty'] * (action_change ** 2)

        time_penalty = -params['time_penalty']

        success_bonus = params['success_bonus'] if distance_to_goal < self.target_threshold else 0.0

        reward = progress_reward + action_penalty + smoothness_penalty + time_penalty + success_bonus
        return float(reward)

    def _get_default_reward_params(self, reward_type):
        defaults = {
            'normalized_progress': {
                'progress_scale': 300,
                'time_penalty': 0.05,
                'efficiency_multiplier': 1.0,
                'success_bonus': 200
            },
            'exponential': {
                'alpha': 5.0,
                'scale': 100,
                'time_penalty': 0.05,
                'success_bonus': 250
            },
            'staged': {
                'thresholds': [0.05, 0.02, 0.005, 0.002, 0.001],
                'bonuses': [20, 40, 80, 120, 200],
                'time_penalty': 0.05
            },
            'dense_shaping': {
                'shaping_scale': 300,
                'time_penalty': 0.05,
                'success_bonus': 200
            },
            'energy_efficient': {
                'progress_scale': 300,
                'action_penalty': 0.02,
                'smoothness_penalty': 0.02,
                'time_penalty': 0.05,
                'success_bonus': 200
            }
        }
        return defaults.get(reward_type, {})

    def render(self, mode='human'):
        pass

    def close(self):
        self.sim.close()

    def _extract_position(self, state_dict):
        robotId = list(sorted(state_dict.keys()))[0]
        robot_state = state_dict.get(robotId, {})
        position = np.array(
            robot_state.get('pipette_position', [0.0, 0.0, 0.0]),
            dtype=np.float32
        )
        return position

    def _normalize_position(self, position):
        normalized = 2.0 * (position - self.workspace_low) / (self.workspace_high - self.workspace_low) - 1.0
        # FIX: clip to declared observation space bounds
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)