import gymnasium as gym
import numpy as np
from river_follow_env import WamvGazeboEnv  # Import your wrapper

# Register the environment (optional)
gym.register(
    id='WamvGazeboEnv-v0',
    entry_point='river_follow_env:WamvGazeboEnv',
)

# Create and use the environment
env = WamvGazeboEnv(render_mode='rgb_array')  # or gym.make('WamvGazeboEnv-v0', render_mode='human')

for episode in range(10):
    observation, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Example: simple forward motion with slight turn
        action = np.array([800, 700])  # left, right thrust
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward

    print(f"Episode {episode} reward: {episode_reward}")

env.close()