import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os
import datetime # Added
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter # Added

from environmentStraightCorner import StraightCornerEnv


# Checkpoint Saving Function  
def save_checkpoint(step, policy, optimizer, episode_count, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "step": step,
        "episode": episode_count,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    ckpt_path = os.path.join(save_dir, f"ppo_checkpoint_step_{step}.pt")
    torch.save(checkpoint, ckpt_path)
    print(f"\n💾 Checkpoint saved at step {step} to {ckpt_path}")

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256), log_std_init=-0.5):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        # Action mean head
        self.mu_head = nn.Linear(hidden_sizes[1], action_dim)
        # Log standard deviation
        self.log_std = nn.Parameter(torch.ones(action_dim, dtype=torch.float32) * log_std_init)
        # Value head
        self.v_head = nn.Linear(hidden_sizes[1], 1)

    def forward(self, x):
        features = self.shared(x)
        mu = self.mu_head(features)
        std = torch.exp(self.log_std)
        value = self.v_head(features)
        return mu, std, value

    def act(self, obs):
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1) # Sum across action dimensions
        return action, log_prob, value # Return value tensor

class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim, gamma=0.99, lam=0.95):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.start_ptr = 0
        # Initialize buffers on CPU
        self.obs_buf = torch.zeros((size, obs_dim), dtype=torch.float32)
        self.act_buf = torch.zeros((size, act_dim), dtype=torch.float32)
        self.logp_buf = torch.zeros(size, dtype=torch.float32)
        self.rew_buf = torch.zeros(size, dtype=torch.float32)
        self.val_buf = torch.zeros(size, dtype=torch.float32)
        self.adv_buf = torch.zeros(size, dtype=torch.float32)
        self.ret_buf = torch.zeros(size, dtype=torch.float32)

    def store(self, obs, act, logp, rew, val):
        if self.ptr >= self.size:
            print("Warning: RolloutBuffer overflow!")
            return
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.logp_buf[self.ptr] = logp
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val # Store the scalar value
        self.ptr += 1

    def _discount_cumsum_torch(self, x, discount):
        vals = torch.zeros_like(x)
        running_add = 0.0
        for t in reversed(range(x.shape[0])):
            running_add = x[t] + discount * running_add
            vals[t] = running_add
        return vals

    def finish_path(self, last_val=0.0):
        if self.ptr == self.start_ptr:  
            return
        path_slice = slice(self.start_ptr, self.ptr)
        # Ensure last_val is a tensor for concatenation
        last_val_tensor = torch.tensor([last_val], dtype=torch.float32)

        rews = torch.cat([self.rew_buf[path_slice], last_val_tensor])
        vals = torch.cat([self.val_buf[path_slice], last_val_tensor])

        # GAE Calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self._discount_cumsum_torch(deltas, self.gamma * self.lam)

        # Returns calculation (Reward-to-go)
        self.ret_buf[path_slice] = self._discount_cumsum_torch(rews[:-1], self.gamma)

        # Mark this segment as finished
        self.start_ptr = self.ptr

    def get(self):
        if self.ptr == 0:
            print("Warning: RolloutBuffer.get() called with ptr=0")
            # Return empty tensors with correct dimensions
            return (torch.zeros((0, self.obs_buf.shape[1])),
                    torch.zeros((0, self.act_buf.shape[1])),
                    torch.zeros(0),
                    torch.zeros(0),
                    torch.zeros(0))

        # Slice the buffers to get only the collected data up to self.ptr
        obs_batch = self.obs_buf[:self.ptr]
        act_batch = self.act_buf[:self.ptr]
        logp_batch = self.logp_buf[:self.ptr]
        adv_batch = self.adv_buf[:self.ptr] # Advantages calculated in finish_path
        ret_batch = self.ret_buf[:self.ptr] # Returns calculated in finish_path

        # Normalize advantage
        adv_mean = adv_batch.mean()
        adv_std = adv_batch.std()
        normalized_adv_batch = (adv_batch - adv_mean) / (adv_std + 1e-8)

        # Reset pointer *after* slicing and processing
        self.ptr = 0
        return obs_batch, act_batch, logp_batch, normalized_adv_batch, ret_batch

def ppo_update(policy, optimizer, buffer, writer, current_step, clip_ratio=0.2, vf_coef=0.5, ent_coef=0.001, epochs=10, batch_size=256, grad_clip_norm=0.5):
    obs, act, logp_old, adv, ret = buffer.get()

    ret_mean = ret.mean()
    ret_std = ret.std()
    normalized_ret = (ret - ret_mean) / (ret_std + 1e-8)

    data_size = obs.shape[0]
    # Adjust batch size if collected data is less
    effective_batch_size = min(batch_size, data_size)

    if data_size == 0:
        print("Warning: ppo_update called with empty buffer. Skipping update.")
        return # Skip update if buffer was empty

    # Store losses per epoch for logging average
    all_policy_loss = []
    all_value_loss = []
    all_entropy = []
    all_approx_kl = []

    for i in range(epochs):
        indices = torch.randperm(data_size) # Shuffle indices each epoch
        for start in range(0, data_size, effective_batch_size):
            end = start + effective_batch_size
            batch_indices = indices[start:end]

            # Get batch data
            batch_obs = obs[batch_indices]
            batch_act = act[batch_indices]
            batch_logp_old = logp_old[batch_indices]
            batch_adv = adv[batch_indices] # Already normalized
            batch_normalized_ret = normalized_ret[batch_indices]

            # Forward pass with current policy
            mu, std, val = policy(batch_obs)
            dist = Normal(mu, std)
            logp = dist.log_prob(batch_act).sum(-1) # Sum log_prob across action dimensions
            ratio = torch.exp(logp - batch_logp_old)

            # Calculate approximate KL divergence (for monitoring)
            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                all_approx_kl.append(approx_kl)

            # Policy loss (Clipped Surrogate Objective)
            clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * batch_adv
            policy_loss = -(torch.min(ratio * batch_adv, clip_adv)).mean()

            # Value loss (MSE)
            value_loss = F.mse_loss(val.squeeze(-1), batch_normalized_ret) # Squeeze value output

            # Entropy bonus (maximize entropy)
            entropy = dist.entropy().mean() # Mean entropy over batch

            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            # Gradient Clipping
            nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()

            # Store batch metrics
            all_policy_loss.append(policy_loss.item())
            all_value_loss.append(value_loss.item())
            all_entropy.append(entropy.item())

    # Log average metrics for the entire update phase to TensorBoard
    if all_policy_loss: # Check if list is not empty (if epochs > 0 and data_size > 0)
        avg_policy_loss = sum(all_policy_loss) / len(all_policy_loss)
        avg_value_loss = sum(all_value_loss) / len(all_value_loss)
        avg_entropy = sum(all_entropy) / len(all_entropy)
        avg_approx_kl = sum(all_approx_kl) / len(all_approx_kl)

        writer.add_scalar("Update/PolicyLoss", avg_policy_loss, current_step)
        writer.add_scalar("Update/ValueLoss", avg_value_loss, current_step)
        writer.add_scalar("Update/Entropy", avg_entropy, current_step)
        writer.add_scalar("Update/ApproxKL", avg_approx_kl, current_step)
        writer.add_scalar("Update/PolicyStd", std.mean().item(), current_step) # Log avg std dev
        writer.add_scalar("Update/ReturnMean_Raw", ret_mean.item(), current_step)
        writer.add_scalar("Update/ReturnStd_Raw", ret_std.item(), current_step)

def train_ppo_with_tensorboard():
    """Main PPO training loop with TensorBoard logging."""

        # --- Configuration ---
    total_steps = 3_000_000 # Target total steps for training
    buffer_size = 4096    # Size of the rollout buffer
    learning_rate = 5e-5  # Adam learning rate
    gamma = 0.99          # Discount factor
    lam = 0.95            # GAE lambda parameter
    ppo_epochs = 10       # Number of PPO update epochs per rollout
    ppo_batch_size = 256  # Minibatch size for PPO updates
    ppo_clip_ratio = 0.3  # PPO clipping parameter
    vf_coef = 0.3       # Value function loss coefficient
    ent_coef = 0.0001      # Entropy bonus coefficient
    grad_clip_norm = 0.5  # Gradient clipping norm
    checkpoint_every = 100_000 # Save checkpoint every N steps
    log_std_init = -1.0   # Initial log standard deviation for policy

    try:
        env = StraightCornerEnv()
        # Reset environment to get initial observation and info
        initial_obs = env.reset()
        if isinstance(initial_obs, tuple) and len(initial_obs) == 2:
            obs, reset_info = initial_obs
        else:
            # Fallback if reset returns only observation
            obs = initial_obs
            reset_info = {}
            print("Warning: env.reset() did not return (obs, info) tuple.")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = ActorCritic(obs_dim, act_dim, log_std_init=log_std_init)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    buffer = RolloutBuffer(buffer_size, obs_dim, act_dim, gamma=gamma, lam=lam)

    base_log_dir = "./logs/ppo_metadrive_runs" # Base directory for logs
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    run_name = f"PPO_{timestamp}"
    run_log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(run_log_dir, exist_ok=True)
    print(f"TensorBoard logs and checkpoints will be saved to: {run_log_dir}")
    writer = SummaryWriter(log_dir=run_log_dir)
    # Save checkpoints inside the run's log directory
    checkpoint_save_dir = os.path.join(run_log_dir, "checkpoints")
    os.makedirs(checkpoint_save_dir, exist_ok=True)
    # -------------------------------------------

    steps_collected = 0 # Total steps collected across all rollouts
    episode_count = 0   # Total completed episodes

    while steps_collected < total_steps:

        rollout_steps = 0
        rollout_done = False 

        while buffer.ptr < buffer.size:
            # Convert observation to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32)

            # Get action, log_prob, and value from policy
            with torch.no_grad():
                action, logp, val_tensor = policy.act(obs_tensor)
                val_scalar = val_tensor.item() # Convert value to scalar

            # Step the environment
            try:
                # Ensure action is in numpy format for env.step
                action_np = action.cpu().numpy() # Move to CPU if on GPU
                step_output = env.step(action_np)
                next_obs_array, reward, terminated, truncated, info = step_output
                done = terminated or truncated
            except Exception as e:
                print(f"\nError during env.step: {e}")
                done = True # Treat error as end of episode segment
                try:
                    obs, reset_info = env.reset()
                    print("Environment reset after step error.")
                except Exception as reset_e:
                    print(f"Error resetting environment after step error: {reset_e}")
                    writer.close()
                    env.close()
                    return # Critical error, exit training

            # Store experience in buffer
            buffer.store(obs_tensor, action.detach(), logp.detach(), reward, val_scalar)

            # Update current observation
            obs = next_obs_array
            rollout_steps += 1
            rollout_done = done # Store if this step ended an episode

            if done:
                episode_count += 1
                ep_ret = info.get("episode_reward", "N/A") 
                ep_len = info.get("episode_length", "N/A")
                ep_route_comp = info.get("route_completion", 0.0) # Get route completion specifically
                success = 1 if terminated and ep_route_comp >= 0.99 else 0

                # Log true end-of-episode stats using estimated total steps
                current_total_step_est = steps_collected + buffer.ptr
                writer.add_scalar("Charts/EpisodeReward", ep_ret if ep_ret != "N/A" else 0, current_total_step_est)
                writer.add_scalar("Charts/EpisodeLength", ep_len if ep_len != "N/A" else 0, current_total_step_est)
                writer.add_scalar("Charts/RouteCompletion", ep_route_comp, current_total_step_est)
                writer.add_scalar("Charts/SuccessRate", success, current_total_step_est) # Log 0 or 1 for success
                writer.flush()

                print(f"\n[Ep {episode_count}]"
                        f"Len: {ep_len} | Rew: {ep_ret} | Route: {ep_route_comp:.2f} | Success: {success}"
                        f"progress= {steps_collected}/{total_steps}")

                # Finish the current trajectory segment since the episode ended
                buffer.finish_path(0.0)
                
                # Reset environment for the next episode
                obs, reset_info = env.reset()

                # If buffer becomes full exactly when episode ends, break inner loop
                if buffer.ptr == buffer.size:
                    break

            # Break inner loop if buffer is full
            if buffer.ptr == buffer.size:
                break

        # Get actual number of steps collected in this rollout
        num_collected_steps = rollout_steps # buffer.ptr should equal rollout_steps here

        # Calculate last value for GAE (bootstrap if buffer filled mid-episode)
        with torch.no_grad():
            last_val = 0.0
            if not rollout_done: # If the last step collected didn't end an episode
                obs_tensor_end = torch.tensor(obs, dtype=torch.float32)
                _, _, last_val_tensor = policy.forward(obs_tensor_end)
                last_val = last_val_tensor.item()

        buffer.finish_path(last_val) # Calculate advantages and returns

        # PPO Update Phase 
        if num_collected_steps > 0:
            # Store steps before update for accurate logging/checkpointing
            steps_before_update = steps_collected
            # Perform PPO updates and log metrics
            ppo_update(policy, optimizer, buffer, writer, steps_before_update,
                       clip_ratio=ppo_clip_ratio, vf_coef=vf_coef, ent_coef=ent_coef,
                       epochs=ppo_epochs, batch_size=ppo_batch_size, grad_clip_norm=grad_clip_norm)

            steps_collected += num_collected_steps
        else:
            print("Warning: Skipping PPO update as no steps were collected in this rollout.")

        # Log total steps collected vs wall time (default x-axis)
        writer.add_scalar("Training/TotalStepsCollected", steps_collected, steps_collected)

        current_checkpoint_interval = steps_collected // checkpoint_every
        previous_checkpoint_interval = steps_before_update // checkpoint_every
        if current_checkpoint_interval > previous_checkpoint_interval:
            save_checkpoint(steps_collected, policy, optimizer, episode_count, checkpoint_save_dir)


    final_model_path = os.path.join(run_log_dir, "ppo_final_model.pth")
    try:
        torch.save(policy.state_dict(), final_model_path)
        print(f"\nFinal model saved to {final_model_path}")
    except Exception as e:
        print(f"\nError saving final model: {e}")

    writer.close() # Close TensorBoard writer
    env.close() # Close the environment cleanly
    print("\nTraining finished.")

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)

    print("Starting PPO training...")
    train_ppo_with_tensorboard()
    print("Script finished.")