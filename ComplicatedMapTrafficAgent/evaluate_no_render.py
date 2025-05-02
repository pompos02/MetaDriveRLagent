import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import MyEnvTraffic

# 1) Load your trained model
model = PPO.load("ComplicatedMapTrafficAgent/ppo_continued.zip")

# 2) Wrap your env in DummyVecEnv
def make_env():
    env = MyEnvTraffic(config=dict(
            num_scenarios=1, # Set num_scenarios for this specific env instance
            start_seed=0,
        ))
    
    return env

eval_env = DummyVecEnv([make_env])

# 3) Re‐attach the env so predict() works correctly
model.set_env(eval_env)

# 4) Evaluation loop
num_eval_episodes = 100
successful_episodes_count = 0
episode_steps = []
successful_episode_steps = []
crashes = []
route_progress = []

for ep in range(num_eval_episodes):
    # reset() now returns only obs_batch, so we index [0]
    obs_batch = eval_env.reset()
    obs = obs_batch[0]
    done = False
    steps = 0
    crash_counter = 0
    route_completion = 0.0
    is_successful = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        # step() still takes a batch, so we wrap action in a list
        next_obs_batch, reward_batch, done_batch, info_batch = eval_env.step([action])

        # unpack the batch of size 1
        obs = next_obs_batch[0]
        reward = reward_batch[0]
        done = done_batch[0]
        info = info_batch[0]

        steps += 1
        route_completion = info.get("route_completion", route_completion)
        crash_counter += info.get("crash_vehicle", 0)

        if route_completion >= 0.99:
            is_successful = True
            done = True

    # record stats
    episode_steps.append(steps)
    crashes.append(crash_counter)
    route_progress.append(route_completion)
    if is_successful:
        successful_episodes_count += 1
        successful_episode_steps.append(steps)

    print(f"  ... Completed episode {ep + 1}/{num_eval_episodes}, "
          f"route completion: {route_completion:.2f}, crashes: {crash_counter}")

# 5) Print summary
print("\n--- Evaluation Summary ---")
print(f"Total episodes evaluated: {num_eval_episodes}")
print(f"Successful episodes: {successful_episodes_count}")
print(f"Success rate: {successful_episodes_count / num_eval_episodes * 100:.2f}%")
print(f"Average route completion: {np.mean(route_progress):.2f}")
print(f"Average crashes per episode: {np.mean(crashes):.2f}")
print(f"Average steps per episode: {np.mean(episode_steps):.2f}")
if successful_episode_steps:
    print(f"Average steps for successful episodes: {np.mean(successful_episode_steps):.2f}")
else:
    print("Average steps for successful episodes: N/A (no successful episodes)")
