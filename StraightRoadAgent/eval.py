import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from StraightRoadAgent.environmentStraight import StraightEnv

# 1) Load the trained model
model = PPO.load("ppo_straight.zip")

config = {"use_render": True}
env = StraightEnv(config)

# If your environment has a new-style reset, do:
obs, info = env.reset()

# 3) Run a few episodes
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
