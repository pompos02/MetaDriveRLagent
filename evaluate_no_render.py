import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from env import MyEnvNoTraffic

# 1) Load the trained model
model = PPO.load("ppo.zip")

env = MyEnvNoTraffic()

# If your environment has a new-style reset, do:
obs, info = env.reset()

# 4) Evaluation without rendering
num_eval_episodes = 100
successful_episodes = 0
episode_steps_list = []

for ep in range(num_eval_episodes):
    obs, info = env.reset()
    terminated, truncated = False, False
    steps = 0
    route_completion = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        route_completion = info.get("route_completion", 0)

    episode_steps_list.append(steps)
    if route_completion >= 0.99:
        successful_episodes += 1

success_rate = (successful_episodes / num_eval_episodes) * 100
avg_steps = np.mean(episode_steps_list)
print("\nEvaluation Summary (No Rendering):")
print(f"Success rate: {success_rate:.2f}%")
print(f"Average episode steps: {avg_steps:.2f}")