import numpy as np
import torch 
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv
from environmentStraightCorner import StraightCornerEnv

class CustomMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.successful_episodes = 0
        self.episode_durations = []
        self.current_episode_steps = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            self.current_episode_steps += 1
            if done:
                if info.get("route_completion", 0) >= 0.99:
                    self.successful_episodes += 1
                    self.episode_durations.append(self.current_episode_steps)
                self.current_episode_steps = 0
        return True

    def _on_rollout_end(self):
        if self.episode_durations:
            avg_duration = np.mean(self.episode_durations)
        else:
            avg_duration = 0
        self.logger.record("custom/successful_episodes", self.successful_episodes)
        self.logger.record("custom/avg_episode_duration", avg_duration)
        # Reset stats after logging
        self.successful_episodes = 0
        self.episode_durations = []

    def _on_training_end(self) -> None:
        print(f"Training ended.")

def make_env():
    return StraightCornerEnv()


if __name__ == "__main__":
    total_steps = 3_000_000

    policy_kwargs = dict(
        net_arch=[256, 256],  # two layers with 256 units each
        activation_fn=torch.nn.ReLU, 
        log_std_init=-3.0,  
    )

    env = make_env()

    # Build the PPO model
    model = PPO(
        policy="MlpPolicy",
        policy_kwargs=policy_kwargs,
        env=env,
        verbose=0,
        n_steps=4096,
        batch_size=256,
        learning_rate=5e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef = 0.001,      # encourages more exploration initially
        vf_coef=0.5,       # balanced value function
        max_grad_norm=0.25,
        tensorboard_log="./logs/tb_logs/"
    )

    # Callback to save checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,          # Save every 100k timesteps
        save_path="./checkpoints",  # Directory to save models
        name_prefix="ppo",          # Prefix for saved models
        save_replay_buffer=True,
        save_vecnormalize=True
    )


    # Combine both callbacks
    combined_callback = CallbackList([checkpoint_callback, CustomMetricsCallback()])

    # Train with combined callbacks
    model.learn(
        total_timesteps=total_steps,
        callback=combined_callback,
        progress_bar=True
    )
    model.save("ppo")
    print("Model saved as ppo.zip")
