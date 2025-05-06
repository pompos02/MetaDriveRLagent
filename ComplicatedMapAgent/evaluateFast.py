import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from env import MyEnvNoTraffic

# 1) Load the trained model
model = PPO.load("ComplicatedMapAgent/ppoPeriploko3NoTrafALLAGES.zip")

config = {"use_render": True,
        "on_continuous_line_done": False,
        "out_of_road_done": True,
        }
env = MyEnvNoTraffic(config)

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
        env.render()

    print(f"Episode {ep+1} reward: {episode_reward}")

env.close()
