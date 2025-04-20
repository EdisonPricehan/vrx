import gymnasium as gym
import numpy as np
from river_follow_env import WamvGazeboEnv

# Register the environment (optional)
gym.register(
    id='WamvGazeboEnv-v0',
    entry_point='river_follow_env:WamvGazeboEnv',
)


if __name__ == '__main__':
    # Create and use the environment. render_mode can be either "rgb_array" or "human".
    env = WamvGazeboEnv(render_mode='human')  # or gym.make('WamvGazeboEnv-v0', render_mode='human')
    observation, info = env.reset()

    for episode in range(10):
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

