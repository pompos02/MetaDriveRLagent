import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import typing
import time
from metadrive.envs import MetaDriveEnv

# Define the Actor network
class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(259, 512)  # input size matches observation dimension
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)    # output size 2 for throttle and steering

    def forward(self, x: torch.Tensor) -> torch.distributions.MultivariateNormal:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        raw = self.fc3(x)
        steering = torch.tanh(raw[:, 0])
        throttle = torch.tanh(raw[:, 1])
        mu = torch.stack([steering, throttle], dim=1)

        # sigma is fixed at 0.05
        sigma = 0.05 * torch.ones_like(mu)
        return torch.distributions.MultivariateNormal(mu, torch.diag_embed(sigma))

# Policy wrapper for the Actor network
class NNPolicy:
    def __init__(self, net: Actor):
        self.net = net

    def __call__(self, obs: npt.NDArray) -> tuple[float, float]:
        # convert observation to a tensor
        device = next(self.net.parameters()).device
        obs_tensor = torch.tensor(np.stack([obs]), dtype=torch.float32, device=device)
        # sample an action from the policy network
        with torch.no_grad():
            throttle, steering = self.net(obs_tensor).sample()[0]
        return throttle.item(), steering.item()

def evaluate_policy(env_config, policy, num_episodes=5, render=True, verbose=True):
    """
    Evaluate a policy in the MetaDrive environment
    """
    env = MetaDriveEnv(env_config)
    total_rewards = []
    
    try:
        for episode in range(num_episodes):
            if verbose:
                print(f"\nEpisode {episode+1}/{num_episodes} ==============================")
            
            obs, info = env.reset()
            done = False
            episode_reward = 0.0
            step = 0
            
            while not done:
                # Get action from policy
                action = policy(obs)
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                
                # Print info every 100 steps
                if verbose and step % 100 == 0:
                    print(f"Step {step}, Current reward: {episode_reward:.2f}")
                
                # Check if episode is done
                done = terminated or truncated
                
                # For visualization
                if render:
                    env.render()
                    # Add small delay to make visualization smoother
                    time.sleep(0.01)
            
            total_rewards.append(episode_reward)
            if verbose:
                print(f"Episode {episode+1} finished after {step} steps with reward: {episode_reward:.2f}")
    
    finally:
        # Always close the environment properly
        env.close()
    
    return total_rewards

if __name__ == "__main__":
    print("Loading trained PPO models...")
    
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    log_dir = "runs/PPO_MetaDrive_DecayExplore_v1" # Choose a version/name
    
    # Create the actor network and load weights
    actor = Actor().to(device)
    actor.load_state_dict(torch.load("StraightCornerRoadPytorchAgent2/ppo_actor_model.pt", map_location=device))
    actor.eval()  # Set to evaluation mode
    
    # Create policy wrapper
    policy = NNPolicy(actor)
    
    # Environment configuration for evaluation
    config = {
            "start_seed": 0, 
            "num_scenarios":1,
        #   "accident_prob":1.0,
            "traffic_density":0.0,
            "use_render": True,
            "map": "SSSCSSSSC",
            "horizon": 3000,  # Maximum steps per episode
          }
    
    # Run evaluation
    print("Starting evaluation...")
    rewards = evaluate_policy(
        env_config=config,
        policy=policy,
        num_episodes=3,  # Change this to run more or fewer episodes
        render=True,
        verbose=True
    )
    
    # Print summary statistics
    print("\nEvaluation Results:")
    print(f"Number of episodes: {len(rewards)}")
    print(f"Average reward: {np.mean(rewards):.2f}")
    print(f"Standard deviation: {np.std(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    print("\nIndividual episode rewards:")
    for i, reward in enumerate(rewards):
        print(f"Episode {i+1}: {reward:.2f}")
