import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList
)
from env import MyEnvNoTraffic
# --- CHANGE HERE: Import DummyVecEnv instead of SubprocVecEnv ---
from stable_baselines3.common.vec_env import DummyVecEnv
from functools import partial
import logging
logging.getLogger().setLevel(logging.ERROR)

class CustomMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successful_episodes = 0
        self.episode_rewards = []
        self.episode_durations = []
        self.current_episode_rewards = None
        self.current_episode_steps = None
        self.crash = 0

    def _on_training_start(self) -> None:
        # This will still work correctly, even if n_envs is 1
        n_envs = self.training_env.num_envs
        self.current_episode_rewards = [0.0] * n_envs
        self.current_episode_steps = [0] * n_envs

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i in range(self.training_env.num_envs):
            self.current_episode_rewards[i] += rewards[i]
            self.current_episode_steps[i] += 1
            self.crash += infos[i].get("crash", 0) # Added crash tracking
            if dones[i]:
                # Log episode metrics
                self.episode_rewards.append(self.current_episode_rewards[i])
                self.episode_durations.append(self.current_episode_steps[i])
                self.logger.record("custom/crash", self.crash) # Log crash count

                # Check for success
                # Ensure 'route_completion' is always in infos, handle potential KeyError if not
                if infos[i] is not None and infos[i].get("route_completion", 0) >= 0.99:
                    self.successful_episodes += 1

                # Reset counters
                self.current_episode_rewards[i] = 0.0
                self.current_episode_steps[i] = 0
                self.crash = 0

        return True

    def _on_rollout_end(self):
        # Calculate metrics based on episodes finished within the rollout
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards)
            avg_duration = np.mean(self.episode_durations)
            avg_crash = np.mean(self.crash) # Average crash count per episode
            # Ensure division by zero doesn't happen if len is 0 (though checked by if)
            success_rate = self.successful_episodes / len(self.episode_rewards) if len(self.episode_rewards) > 0 else 0
        else:
            avg_reward = 0
            avg_duration = 0
            success_rate = 0
            avg_crash = 0

        self.logger.record("custom/avg_reward", avg_reward)
        self.logger.record("custom/avg_episode_duration", avg_duration)
        self.logger.record("custom/success_rate", success_rate)
        self.logger.record("custom/successful_episodes_in_rollout", self.successful_episodes) # Renamed for clarity
        self.logger.record("custom/avg_crash", avg_crash) # Log average crash count
        # Reset stats for the next rollout
        self.successful_episodes = 0
        self.episode_rewards = []
        self.episode_durations = []

    def _on_training_end(self) -> None:
        print("Training ended.")

# make_env function remains the same
def make_env(seed: int):
    def _init():
        print(f"Initializing environment with seed {seed}...") # Added print for debugging
        # Consider setting traffic_density=0.0 if MyEnvNoTraffic truly means no traffic
        env = MyEnvNoTraffic(config=dict(
            num_scenarios=1, # Set num_scenarios for this specific env instance
            start_seed=seed,
        ))
        print(f"Environment with seed {seed} initialized.") # Added print for debugging
        return env
    return _init

if __name__ == "__main__":
    total_steps = 5_000_000 # Reduce this significantly for initial testing/debugging

    # --- CHANGE HERE: Use DummyVecEnv ---
    # Create a single environment running in the main process
    num_envs = 1
    print("Creating DummyVecEnv...")
    env = DummyVecEnv([
        make_env(seed=i) for i in range(num_envs)
    ])
    print("DummyVecEnv created.")
    print("Loading pretrained model...")
    model = PPO.load(
        "ppoPeriploko3NoTrafALLAGES",
        env=env,                
    )
    print("Model loaded. Resuming training...")

    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU,
        #log_std_init = -2.0 
    )

    # Checkpoint callback remains the same, but saving vec_normalize might not be needed
    # if you don't wrap the DummyVecEnv in VecNormalize
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // num_envs, 1), 
        save_path="./checkpoints",
        name_prefix="ppo_dummy", 
        save_replay_buffer=False, # PPO doesn't use a replay buffer by default
        save_vecnormalize=False # Set to False unless you explicitly use VecNormalize
    )

    # Combined callback list remains the same
    combined_callback = CallbackList([
        checkpoint_callback,
        CustomMetricsCallback()
    ])

    print(f"Starting training for {total_steps} timesteps...")
    try:
        model.learn(
            total_timesteps=total_steps,
            callback=combined_callback,
            reset_num_timesteps=False, # Keep the number of timesteps from the loaded model
            progress_bar=True
        )
        print("Training finished successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback on error


    try:
        model.save("ppo_dummy_continued") # Changed save name slightly
        print("Model saved as ppo_dummy_continued.zip")
    except Exception as e:
        print(f"An error occurred during model saving: {e}")

    # Close the environment(s)
    env.close()
    print("Environment closed.")