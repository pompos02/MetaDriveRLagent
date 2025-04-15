# %%


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import typing
import gymnasium as gym
from dataclasses import dataclass
from metadrive.envs import MetaDriveEnv

from torch.utils.tensorboard import SummaryWriter


# %%


def collect_trajectory(env:gym.Env, policy:typing.Callable[[npt.NDArray], int]) -> tuple[list[npt.NDArray], list[int], list[float]]:
    """
    Collect a trajectory from the environment using the given policy
    """
    observations = []
    actions = []
    rewards = []
    episode_length = 0 # +++ Track episode length +++
    obs, info = env.reset()
    # print("Observation shape:", obs.shape)


    while True:
        observations.append(obs)
        action = policy(obs)
        # print("Action:", action)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        episode_length += 1 # +++ Increment length +++
        if terminated or truncated:
            break
    # +++ Return episode length as well +++
    return observations, actions, rewards, episode_length



# %%


def deviceof(m: nn.Module) -> torch.device:
    """
    Get the device of the given module
    """
    return next(m.parameters()).device

# %%


def obs_batch_to_tensor(obs: list[npt.NDArray[np.float32]], device: torch.device) -> torch.Tensor:
    """
    Convert observation batch to tensor
    """
    return torch.tensor(np.stack(obs), dtype=torch.float32, device=device)



# %%


# %%
class Actor(nn.Module):
    def __init__(self, initial_sigma=0.5, final_sigma=0.05, exploration_decay_epochs=5000):
        super().__init__()
        self.fc1 = nn.Linear(259, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)    # raw output

        # --- Exploration Parameters ---
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.exploration_decay_epochs = exploration_decay_epochs
        self.current_sigma = self.initial_sigma # Initialize sigma
        # -----------------------------

    def forward(self, x: torch.Tensor) -> torch.distributions.MultivariateNormal:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        raw = self.fc3(x)   # shape [batch_size, 2]

        # If your action space is Box(-1,1,2):
        #   raw[:,0] is steering in [-1,1]
        #   raw[:,1] is throttle in [-1,1]
        steering = torch.tanh(raw[:, 0])        # [-1,1]
        throttle = torch.tanh(raw[:, 1])        # [-1,1]
        mu = torch.stack([steering, throttle], dim=1)

        # --- Use current_sigma for exploration ---
        # Ensure sigma is on the same device as mu
        sigma_val = self.current_sigma
        # Create a diagonal covariance matrix
        # Assuming independent noise for steering and throttle for simplicity
        # You could also have separate decaying sigmas if needed
        cov_diag = torch.full_like(mu, sigma_val**2) # Use variance (sigma squared)
        # Clamp variance to avoid numerical issues if sigma becomes very small
        cov_diag = torch.clamp(cov_diag, min=1e-6)
        # -----------------------------------------

        return torch.distributions.MultivariateNormal(mu, torch.diag_embed(cov_diag))

    def update_exploration(self, current_epoch: int):
        """
        Updates the exploration sigma based on the current training epoch using linear decay.
        """
        decay_fraction = min(float(current_epoch) / self.exploration_decay_epochs, 1.0)
        self.current_sigma = self.initial_sigma + decay_fraction * (self.final_sigma - self.initial_sigma)
        # Ensure sigma doesn't go below the final value
        self.current_sigma = max(self.current_sigma, self.final_sigma)


class NNPolicy:
    def __init__(self, net: Actor):
        self.net = net

    def __call__(self, obs:npt.NDArray) -> tuple[float, float]:
        # convert observation to a tensor
        obs_tensor = obs_batch_to_tensor([obs], deviceof(self.net))
        # sample an action from the policy network
        with torch.no_grad():
            action_sample = self.net(obs_tensor).sample()[0]
            steering = action_sample[0].item()
            throttle = action_sample[1].item()
            steering = np.clip(steering, -1.0, 1.0)
            throttle = np.clip(throttle, -1.0, 1.0)
        return steering, throttle

# critic network
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(259, 512)  
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.squeeze(x, dim=1)


def rewards_to_go(trajectory_rewards: list[float], gamma) -> list[float]:
    """
    Computes the gamma discounted reward-to-go for each state in the trajectory.
    """

    trajectory_len = len(trajectory_rewards)

    v_batch = np.zeros(trajectory_len)

    v_batch[-1] = trajectory_rewards[-1]

    # Use gamma to decay the advantage
    for t in reversed(range(trajectory_len - 1)):
        v_batch[t] = trajectory_rewards[t] + gamma * v_batch[t + 1]

    return list(v_batch)

def compute_advantage(
    critic: Critic,
    trajectory_observations: list[npt.NDArray[np.float32]],
    trajectory_rewards: list[float],
    gamma: float
) -> list[float]:
    """
    Computes advantage using GAE (or simple TD residual in this case).
    Note: This implementation calculates V(s_t+1) - V(s_t) style advantage, not full GAE.
    For GAE you would need lambda parameter and a different calculation.
    Let's assume this implementation is intended.
    """

    trajectory_len = len(trajectory_rewards)

    assert len(trajectory_observations) == trajectory_len
    assert len(trajectory_rewards) == trajectory_len

    # calculate the value of each state
    with torch.no_grad():
        obs_tensor = obs_batch_to_tensor(trajectory_observations, deviceof(critic))
        obs_values = critic.forward(obs_tensor).detach().cpu().numpy()

    # calculate rewards-to-go which act as the target value Q(s,a) estimate here
    trajectory_rtg = np.array(rewards_to_go(trajectory_rewards, gamma))

    # Advantage A(s,a) = Q(s,a) - V(s) = RTG(s) - V(s)
    trajectory_advantages = trajectory_rtg - obs_values

    return list(trajectory_advantages)


@dataclass
class PPOConfig:
    ppo_eps: float
    ppo_grad_descent_steps: int

def compute_ppo_loss(
    # Old policy network's distribution of actions given a state
    # inner shape = (Batch, 2)
    pi_thetak_given_st: torch.distributions.MultivariateNormal,
    # Current policy network's distribution of actions given a state
    # in (Batch, Action)
    pi_theta_given_st: torch.distributions.MultivariateNormal,
    # The action chosen by the old policy network
    # in (Batch, 2)
    a_t: torch.Tensor,
    # Advantage of the chosen action
    # in (Batch,)
    A_pi_thetak_given_st_at: torch.Tensor,
    # configuration options
    config: PPOConfig
) -> torch.Tensor:
    # the likelihood ratio (used to penalize divergence from the old policy)
    # in (Batch,)
    # Use clamp to prevent division by zero or infinite ratios if probabilities are tiny
    log_prob_new = pi_theta_given_st.log_prob(a_t)
    log_prob_old = pi_thetak_given_st.log_prob(a_t)
    likelihood_ratio = torch.exp(log_prob_new - log_prob_old)


    # in (Batch,)
    # Unclipped objective
    unclipped_obj = likelihood_ratio * A_pi_thetak_given_st_at
    # Clipped objective
    clipped_obj = torch.clip(likelihood_ratio, 1 - config.ppo_eps, 1 + config.ppo_eps) * A_pi_thetak_given_st_at

    # PPO's pessimistic objective (take the minimum)
    ppo_loss_per_example = -torch.minimum(unclipped_obj, clipped_obj) # Negative sign because optimizers minimize

    # we take the average loss over all examples
    return ppo_loss_per_example.mean()



# %%
def train_ppo(
    actor: Actor,
    critic: Critic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    observation_batch: list[npt.NDArray[np.float32]],
    action_batch: list[tuple[float, float]],
    advantage_batch: list[float],
    reward_to_go_batch: list[float],
    config: PPOConfig
) -> tuple[list[float], list[float]]:
    # assert that the models are on the same device
    assert deviceof(critic) == deviceof(actor)
    # assert that the batch_lengths are the same
    batch_size = len(observation_batch)
    assert batch_size == len(action_batch)
    assert batch_size == len(advantage_batch)
    assert batch_size == len(reward_to_go_batch)

    # get device
    device = deviceof(critic)

    # convert data to tensors on correct device

    # in (Batch, ObsDim)
    observation_batch_tensor = obs_batch_to_tensor(observation_batch, device)

    # the target value V_target(s) is reward to go
    # in (Batch,)
    target_value_batch_tensor = torch.tensor(
        reward_to_go_batch, dtype=torch.float32, device=device
    )

    # in (Batch, 2) - Ensure the order matches the policy output
    chosen_action_tensor = torch.tensor(action_batch, dtype=torch.float32, device=device)

    # in (Batch,) - Normalize advantages? (Optional but often helpful)
    advantage_batch_np = np.array(advantage_batch)
    # advantage_batch_np = (advantage_batch_np - advantage_batch_np.mean()) / (advantage_batch_np.std() + 1e-8) # Optional normalization
    advantage_batch_tensor = torch.tensor(advantage_batch_np, dtype=torch.float32, device=device)

    # --- Train Critic ---
    # Multiple critic updates per PPO batch can sometimes stabilize training
    critic_losses_epoch = []
    for _ in range(config.ppo_grad_descent_steps): # Match grad steps or set independently
        critic_optimizer.zero_grad()
        # Get current value predictions
        pred_value_batch_tensor = critic.forward(observation_batch_tensor)
        # Calculate MSE loss against the calculated rewards-to-go
        critic_loss = F.mse_loss(pred_value_batch_tensor, target_value_batch_tensor)
        critic_loss.backward()
        critic_optimizer.step()
        critic_losses_epoch.append(critic_loss.item())
    avg_critic_loss = np.mean(critic_losses_epoch)


    # --- Train Actor ---

    # Get the action log-probabilities from the policy *before* the PPO updates (theta_k)
    # The actor's current_sigma is implicitly used here when forward is called.
    with torch.no_grad():
        old_policy_dist = actor.forward(observation_batch_tensor)
        # Detach is crucial - we don't propagate gradients back through the old policy calculation
        old_policy_dist_detached = torch.distributions.MultivariateNormal(
            loc=old_policy_dist.loc.detach(),
            covariance_matrix=old_policy_dist.covariance_matrix.detach()
            )


    actor_losses_epoch = []
    # Perform multiple gradient steps on the PPO surrogate objective
    for i in range(config.ppo_grad_descent_steps):
        actor_optimizer.zero_grad()
        # Get action probabilities from the *current* policy (theta)
        current_policy_dist = actor.forward(observation_batch_tensor)

        # Calculate the PPO clipped surrogate objective loss
        actor_loss = compute_ppo_loss(
            old_policy_dist_detached, # Pass the detached old distribution
            current_policy_dist,
            chosen_action_tensor,
            advantage_batch_tensor, # Use potentially normalized advantages
            config
        )
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
        actor_optimizer.step()
        actor_losses_epoch.append(actor_loss.item())

    avg_actor_loss = np.mean(actor_losses_epoch)

    return [avg_actor_loss], [avg_critic_loss] # Return lists of length 1 containing averages


import os # +++ Need os for paths +++


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# --- Define Exploration Hyperparameters ---
INITIAL_SIGMA = 0.5  # Start with more exploration (Increased from 0.5)
FINAL_SIGMA = 0.05   # Reduce exploration significantly (Increased from 0.05)
EXPLORATION_DECAY_EPOCHS = 3000 # Decay over most of the training

# --- TensorBoard Setup ---
log_dir = "runs/PPO_MetaDrive_DecayExplore_v1" # Choose a version/name
os.makedirs(log_dir, exist_ok=True) # Create dir if needed
writer = SummaryWriter(log_dir=log_dir)
print(f"--- TensorBoard logging initialized. Log directory: {log_dir} ---")
# -------------------------


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} ---")

# --- Instantiate Actor with exploration parameters ---
actor = Actor(
    initial_sigma=INITIAL_SIGMA,
    final_sigma=FINAL_SIGMA,
    exploration_decay_epochs=EXPLORATION_DECAY_EPOCHS
).to(device)
# -------------------------------------------------
critic = Critic().to(device)

# Learning Rates - Adjusted based on typical PPO findings
ACTOR_LR = 5e-5 # Often smaller than critic LR
CRITIC_LR = 1e-4 # Often larger to learn value function faster

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)

policy = NNPolicy(actor)

step = 0 # Represents the training epoch/batch number
total_timesteps = 0 # Track total environment steps
returns_window = [] # For smoothed return logging
MAX_RETURNS_WINDOW = 100 # Size of the smoothing window

# Clear previous lists if rerunning cell
actor_losses = []
critic_losses = []

from gymnasium import Env



# Ensure you have latest metadrive: pip install -U metadrive-simulator

config = {
            "start_seed": 0,
            "num_scenarios":1, 
            "traffic_density":0.0, 
            "use_render": False, 
            "map": "SSSCSSSSC", 
            "horizon": 1500,  
        }

try:
    env = MetaDriveEnv(config)
    print("--- MetaDrive Environment Created Successfully ---")
    print("Observation Space:", env.observation_space)
    print("Action Space:", env.action_space)
    assert isinstance(env.observation_space, gym.spaces.Box) and env.observation_space.shape == (259,), \
        f"Observation space shape mismatch! Expected (259,), Got {env.observation_space.shape}"
    assert isinstance(env.action_space, gym.spaces.Box) and env.action_space.shape == (2,), \
        f"Action space shape mismatch! Expected (2,), Got {env.action_space.shape}"

except Exception as e:
    print(f"--- Error creating MetaDrive Environment: {e} ---")
    print("--- Check MetaDrive installation and configuration. ---")
    raise e # Stop execution if env fails


TRAIN_EPOCHS = 100_000 # Total number of PPO update batches
EPISODES_PER_BATCH = 3 # Collect more data per update step (Increased from 1)
GAMMA = 0.99 # Discount factor, often higher for continuous tasks
CONFIG = PPOConfig(
    ppo_eps=0.3, # PPO clip range, common value is 0.2
    ppo_grad_descent_steps=10, # Number of optimization steps per batch
)

print(f"--- Starting Training ---")
print(f"Device: {device}, Train Epochs: {TRAIN_EPOCHS}, Episodes/Batch: {EPISODES_PER_BATCH}")
print(f"Gamma: {GAMMA}, PPO Clip Epsilon: {CONFIG.ppo_eps}, Grad Steps: {CONFIG.ppo_grad_descent_steps}")
print(f"Actor LR: {ACTOR_LR}, Critic LR: {CRITIC_LR}")
print(f"Initial Sigma: {INITIAL_SIGMA}, Final Sigma: {FINAL_SIGMA}, Decay Epochs: {EXPLORATION_DECAY_EPOCHS}")
print(f"TensorBoard Log Dir: {log_dir}")
print("-" * 30)


# Train
try: 
    while step < TRAIN_EPOCHS:
        # --- Update exploration sigma before collecting trajectories ---
        actor.update_exploration(step)
        current_sigma = actor.current_sigma # Get current sigma for logging

        obs_batch:list[npt.NDArray[np.float32]] = []
        act_batch:list[tuple[float, float]] = []
        rtg_batch:list[float] = []
        adv_batch:list[float] = []

        batch_ep_returns = [] # Store returns for this batch
        batch_ep_lengths = [] # Store lengths for this batch
        batch_total_timesteps = 0 # Timesteps collected in this batch

        # --- Data Collection Phase ---
        for ep_num in range(EPISODES_PER_BATCH):
            # Collect trajectory
            # Ensure collect_trajectory returns length now
            obs_traj, act_traj, rew_traj, ep_len = collect_trajectory(env, policy)
            rtg_traj = rewards_to_go(rew_traj, GAMMA)
            # Advantage calculation uses the critic, not directly affected by actor's sigma
            adv_traj = compute_advantage(critic, obs_traj, rew_traj, GAMMA)

            # Update batch lists
            obs_batch.extend(obs_traj)
            act_batch.extend(act_traj) # act_traj contains tuples
            rtg_batch.extend(rtg_traj)
            adv_batch.extend(adv_traj)

            # Store episode statistics
            ep_return = sum(rew_traj)
            batch_ep_returns.append(ep_return)
            batch_ep_lengths.append(ep_len)
            batch_total_timesteps += ep_len

        total_timesteps += batch_total_timesteps # Update global timestep counter

        # --- Learning Phase ---
        # train_ppo now returns average losses for the batch
        batch_avg_actor_loss, batch_avg_critic_loss = train_ppo(
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            obs_batch,
            act_batch,
            adv_batch,
            rtg_batch,
            CONFIG,
        )
        actor_loss_val = batch_avg_actor_loss[0] # Extract float from list
        critic_loss_val = batch_avg_critic_loss[0]

        # --- Logging Phase ---
        # Calculate return statistics for the batch
        avg_batch_return = np.mean(batch_ep_returns)
        std_batch_return = np.std(batch_ep_returns)
        median_batch_return = np.median(batch_ep_returns)
        avg_batch_length = np.mean(batch_ep_lengths)

        # Update sliding window of returns
        returns_window.extend(batch_ep_returns)
        if len(returns_window) > MAX_RETURNS_WINDOW:
            returns_window = returns_window[-MAX_RETURNS_WINDOW:]
        avg_window_return = np.mean(returns_window) if returns_window else 0.0


        # +++ Log metrics to TensorBoard +++
        writer.add_scalar('Reward/BatchAverageReturn', avg_batch_return, step)
        writer.add_scalar('Reward/BatchStdReturn', std_batch_return, step)
        writer.add_scalar('Reward/BatchMedianReturn', median_batch_return, step)
        writer.add_scalar('Reward/WindowAverageReturn', avg_window_return, step) # Smoothed return
        writer.add_scalar('Loss/Actor', actor_loss_val, step)
        writer.add_scalar('Loss/Critic', critic_loss_val, step)
        writer.add_scalar('Episode/AverageLength', avg_batch_length, step)
        writer.add_scalar('Params/CurrentSigma', current_sigma, step)
        writer.add_scalar('Params/LearningRate_Actor', actor_optimizer.param_groups[0]['lr'], step) # Log LR
        writer.add_scalar('Params/LearningRate_Critic', critic_optimizer.param_groups[0]['lr'], step)
        writer.add_scalar('Info/TotalTimesteps', total_timesteps, step) # Track total env interactions
        writer.add_scalar('Info/BatchSize_Timesteps', batch_total_timesteps, step) # Timesteps in this batch
        writer.add_scalar('Info/BatchSize_Episodes', EPISODES_PER_BATCH, step)

        avg_advantage = np.mean(adv_batch) if adv_batch else 0.0
        avg_rtg = np.mean(rtg_batch) if rtg_batch else 0.0
        writer.add_scalar('Stats/AverageAdvantage', avg_advantage, step)
        writer.add_scalar('Stats/AverageRTG', avg_rtg, step)


        # Print statistics 
        if step % 10 == 0 or step == TRAIN_EPOCHS - 1: # Print every 10 steps
            print(
                f"Epoch {step}/{TRAIN_EPOCHS} | "
                f"Timesteps: {total_timesteps} | "
                f"Avg Return (Win {MAX_RETURNS_WINDOW}): {avg_window_return:.2f} | "
                f"Avg Return (Batch): {avg_batch_return:.2f} +/- {std_batch_return:.2f} | "
                f"Avg Ep Len: {avg_batch_length:.1f} | "
                f"Actor Loss: {actor_loss_val:.4f} | "
                f"Critic Loss: {critic_loss_val:.4f} | "
                f"Sigma: {current_sigma:.4f}"
            )

        step += 1 # Increment epoch counter

finally: 
    if 'writer' in locals() and writer:
        writer.close()
        print("--- TensorBoard Writer Closed ---")
    # -------------------------------
    # --- Close Environment ---
    if 'env' in locals() and env:
        env.close()
        print("--- MetaDrive Environment Closed ---")
    # -------------------------


print("--- Training Finished ---")
print("Saving final models...")
save_path_actor = os.path.join(log_dir, "ppo_actor_final.pt") # Save inside log_dir
save_path_critic = os.path.join(log_dir, "ppo_critic_final.pt")
torch.save(actor.state_dict(), save_path_actor)
torch.save(critic.state_dict(), save_path_critic)
print(f"Models saved successfully to {log_dir}")


print("\n--- Starting Evaluation with Final Model ---")

# Load models from the run directory
actor_load_path = save_path_actor
critic_load_path = save_path_critic

# Re-initialize models and load state dicts
eval_actor = Actor(initial_sigma=FINAL_SIGMA, final_sigma=FINAL_SIGMA, exploration_decay_epochs=1).to(device) # Use final sigma for eval
eval_critic = Critic().to(device)

if os.path.exists(actor_load_path):
    eval_actor.load_state_dict(torch.load(actor_load_path, map_location=device))
    print(f"Loaded Actor model from {actor_load_path}")
else:
    print(f"Warning: Actor model file not found at {actor_load_path}. Using untrained model.")

# Critic isn't strictly needed for evaluation rollout, but good practice
if os.path.exists(critic_load_path):
    eval_critic.load_state_dict(torch.load(critic_load_path, map_location=device))
    print(f"Loaded Critic model from {critic_load_path}")
else:
    print(f"Warning: Critic model file not found at {critic_load_path}. Using untrained model.")


eval_policy = NNPolicy(eval_actor)
# Set actor to evaluation mode (disables dropout, batchnorm etc. if used)
eval_actor.eval()

# Setup evaluation environment (with rendering)
eval_config = config.copy() # Start with training config
eval_config["use_render"] = True


eval_env = None
try:
    eval_env = MetaDriveEnv(eval_config)
    print("--- Evaluation Environment Created ---")

    total_eval_reward = 0
    num_eval_episodes = 3 # Run a few evaluation episodes

    for i in range(num_eval_episodes):
        print(f"\n--- Running Evaluation Episode {i+1}/{num_eval_episodes} ---")
        # Use the modified collect_trajectory that returns length
        obs_traj, act_traj, rew_traj, ep_len = collect_trajectory(eval_env, eval_policy)
        ep_reward = sum(rew_traj)
        total_eval_reward += ep_reward
        print(f"Episode Reward: {ep_reward:.2f}, Episode Length: {ep_len}")

    avg_eval_reward = total_eval_reward / num_eval_episodes
    print(f"\n--- Average Evaluation Reward ({num_eval_episodes} episodes): {avg_eval_reward:.2f} ---")

except Exception as e:
    print(f"--- Error during evaluation: {e} ---")
finally:
    if eval_env:
        eval_env.close()
        print("--- Evaluation Environment Closed ---")