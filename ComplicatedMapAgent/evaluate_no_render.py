import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from env import MyEnvNoTraffic

# 1) Load the trained model
model = PPO.load("ComplicatedMapAgent/ppoPeriplokoNoTraf.zip")

env = MyEnvNoTraffic()

obs, info = env.reset()

# 4) Evaluation without rendering
num_eval_episodes = 100
successful_episodes = 0
episode_steps_list = []
successful_episodes_count = 0
successful_episode_steps = []  

for ep in range(num_eval_episodes):
    obs, info = env.reset()
    terminated, truncated = False, False
    steps = 0
    route_completion = 0
    is_successful = False # Flag for success in this episode

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        route_completion = info.get("route_completion")
        if route_completion >= 0.99:
                is_successful = True
                terminated = True
    # Episode finished, record results
    if is_successful: # Check if the episode was flagged as successful
        successful_episodes_count += 1
        successful_episode_steps.append(steps) #

    #Progress indicator
    print(f"  ... Completed episode {ep + 1}/{num_eval_episodes}, route completion: {route_completion:.2f}")


# 5) Calculate and print results
print("\n--- Evaluation Summary ---")

# Calculate success rate
if num_eval_episodes > 0:
    success_rate = (successful_episodes_count / num_eval_episodes) * 100
    print(f"Total episodes evaluated: {num_eval_episodes}")
    print(f"Successful episodes: {successful_episodes_count}")
    print(f"Success rate: {success_rate:.2f}%")
else:
    print("No episodes were evaluated.")
    success_rate = 0

# Calculate average steps for successful episodes
if successful_episodes_count > 0:
    avg_steps_successful = np.mean(successful_episode_steps)
    print(f"Average steps for successful episodes: {avg_steps_successful:.2f}")
else:
    # Handle the case where there were no successful episodes
    print("Average steps for successful episodes: N/A (no successful episodes)")
    avg_steps_successful = 0 # Or None, or np.nan