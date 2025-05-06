import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from environmentStraight import StraightEnv


model = PPO.load("StraightRoadAgent/ppo_straight.zip")

config = {"use_render": True}
env = StraightEnv(config)

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

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward


    print(f"Episode {ep+1} reward: {episode_reward}")

env.close()
