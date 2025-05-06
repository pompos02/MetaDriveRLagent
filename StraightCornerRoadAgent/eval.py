import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from environmentStraightCorner import StraightCornerEnv

model = PPO.load("StraightCornerRoadAgent/ppo_straight_corner.zip")

config = {"use_render": True}
env = StraightCornerEnv(config)

obs, info = env.reset()

num_episodes = 3
for ep in range(num_episodes):
    print(f"Episode {ep+1} ==================================")
    obs, info = env.reset()
    terminated, truncated = False, False
    episode_reward = 0.0

    while not (terminated or truncated):
        # Predict action
        action, _ = model.predict(obs, deterministic=True)

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward


    print(f"Episode {ep+1} reward: {episode_reward}")

env.close()
