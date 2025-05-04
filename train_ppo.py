import numpy as np
import torch as th
import torch.nn as nn
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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
logging.getLogger().setLevel(logging.ERROR)

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # First, initialize the parent class with the observation space and features dim
        super().__init__(observation_space, features_dim=features_dim)
        
        # Extract image space and state space
        image_shape = observation_space["image"].shape
        state_shape = observation_space["state"].shape[0]

        # Initialize CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=image_shape[2], out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute output size of CNN
        with th.no_grad():
            dummy_input = th.as_tensor(observation_space["image"].sample()[None]).float()
            cnn_out_size = self.cnn(dummy_input.permute(0, 3, 1, 2)).shape[1]

        # Initialize state network
        self.state_net = nn.Sequential(
            nn.Linear(state_shape, 64),
            nn.ReLU()
        )

        # Calculate combined size
        combined_size = cnn_out_size + 64
        
        # Initialize final linear layer to output the desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        image = observations["image"].permute(0, 3, 1, 2).float()  # NHWC -> NCHW
        state = observations["state"].float()
        image_out = self.cnn(image)
        state_out = self.state_net(state)
        return self.linear(th.cat([image_out, state_out], dim=1))
    
class CustomMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successful_episodes = 0
        self.episode_rewards = []
        self.episode_durations = []
        self.current_episode_rewards = None
        self.current_episode_steps = None

    def _on_training_start(self) -> None:
        # This will still work correctly, even if n_envs is 1
        n_envs = self.training_env.num_envs
        self.current_episode_rewards = [0.0] * n_envs
        self.current_episode_steps = [0] * n_envs

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", [])
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # This loop will just run once if num_envs is 1
        for i in range(self.training_env.num_envs):
            self.current_episode_rewards[i] += rewards[i]
            self.current_episode_steps[i] += 1

            if dones[i]:
                # Log episode metrics
                self.episode_rewards.append(self.current_episode_rewards[i])
                self.episode_durations.append(self.current_episode_steps[i])

                # Check for success
                # Ensure 'route_completion' is always in infos, handle potential KeyError if not
                if infos[i] is not None and infos[i].get("route_completion", 0) >= 0.99:
                    self.successful_episodes += 1

                # Reset counters
                self.current_episode_rewards[i] = 0.0
                self.current_episode_steps[i] = 0

        return True

    def _on_rollout_end(self):
        # Calculate metrics based on episodes finished within the rollout
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards)
            avg_duration = np.mean(self.episode_durations)
            # Ensure division by zero doesn't happen if len is 0 (though checked by if)
            success_rate = self.successful_episodes / len(self.episode_rewards) if len(self.episode_rewards) > 0 else 0
        else:
            avg_reward = 0
            avg_duration = 0
            success_rate = 0

        self.logger.record("custom/avg_reward", avg_reward)
        self.logger.record("custom/avg_episode_duration", avg_duration)
        self.logger.record("custom/success_rate", success_rate)
        self.logger.record("custom/successful_episodes_in_rollout", self.successful_episodes) # Renamed for clarity

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
            traffic_density=0.0  # Explicitly setting to 0 based on class name assumption
        ))
        print(f"Environment with seed {seed} initialized.") # Added print for debugging
        return env
    return _init

if __name__ == "__main__":
    total_steps = 30_000_000 # Reduce this significantly for initial testing/debugging

    # --- CHANGE HERE: Use DummyVecEnv ---
    # Create a single environment running in the main process
    num_envs = 1
    print("Creating DummyVecEnv...")
    env = MyEnvNoTraffic()  # Use DummyVecEnv to wrap the environment
    obs, info= env.reset()  # VecEnv reset() returns just observations, not a tuple
    print("Initial observation keys:", obs.keys())
    print("Type of obs:", type(obs))
    print("Image shape:", obs["image"].shape)
    print("State shape:", obs["state"].shape)
    print("Image dtype:", obs["image"].dtype)
    print("State dtype:", obs["state"].dtype)


    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    model_verbose_level = 1 # Set to 1 or 2 for more SB3 output

    print("Creating PPO model...")
    model = PPO(
        policy="MultiInputPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        verbose=model_verbose_level, # Increased verbosity
        n_steps=4096,           # Runs 4096 steps in the single env before updating
        batch_size=256,         # Samples batches of 256 from the 4096 steps
        learning_rate=5e-5,
        gamma=0.99,
        gae_lambda=0.97,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.25,
        tensorboard_log="./logs/tb_logs/"
    )
    print("PPO model created.")

    # Checkpoint callback remains the same, but saving vec_normalize might not be needed
    # if you don't wrap the DummyVecEnv in VecNormalize
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // num_envs, 1), 
        save_path="./checkpoints",
        name_prefix="ppo_dummy", # Changed prefix slightly
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
            progress_bar=True
        )
        print("Training finished successfully.")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback on error


    try:
        model.save("ppo_dummy") # Changed save name slightly
        print("Model saved as ppo_dummy.zip")
    except Exception as e:
        print(f"An error occurred during model saving: {e}")

    # Close the environment(s)
    env.close()
    print("Environment closed.")