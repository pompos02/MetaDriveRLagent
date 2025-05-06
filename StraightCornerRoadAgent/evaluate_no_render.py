import torch
import numpy as np

from stable_baselines3 import PPO
from environmentStraightCorner import StraightCornerEnv



num_episodes = 1000
env = StraightCornerEnv()
model = PPO.load("StraightCornerRoadAgent/ppo_straight_corner.zip")

success_count = 0
successful_steps = []

for episode_idx in range(1, num_episodes + 1):
    obs, info = env.reset()
    
    done = False
    ep_reward = 0.0
    ep_steps = 0
    route_completion = 0.0
    
    while not done:

        action, _,  = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        ep_reward += reward
        ep_steps += 1
        route_completion = info.get("route_completion", 0.0)
        has_failed = info.get("has_failed", False)

        if route_completion > 0.98:
            success_count += 1
            successful_steps.append(ep_steps)
            break
        if has_failed:
            break

    
    print(f"Episode {episode_idx} finished in {ep_steps} steps with total reward {ep_reward:.2f} "
            f"and route_completion {route_completion:.2f}")

success_rate = success_count / num_episodes
avg_steps = np.mean(successful_steps) if successful_steps else 0

print(f"\nSuccess Rate: {success_rate*100:.2f}% ({success_count}/{num_episodes})")
print(f"Average Steps for Successful Episodes: {avg_steps:.2f}")

env.close()


