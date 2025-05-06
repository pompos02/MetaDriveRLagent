import torch
import numpy as np

from train import ActorCritic
from environmentStraightCorner import StraightCornerEnv

def evaluate_model(model_path, num_episodes=100):
    """
    Evaluate the trained PPO model for 100 episodes without rendering,
    and output the success rate where route_completion > 0.98,
    as well as the average steps for successful episodes.
    """

    env = StraightCornerEnv()
    
    # Load the trained ActorCritic weights
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = ActorCritic(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()  # Set to evaluation mode 
    
    success_count = 0
    successful_steps = []

    for episode_idx in range(1, num_episodes + 1):
        obs_tuple = env.reset()
        obs, _reset_info = obs_tuple if isinstance(obs_tuple, tuple) else (obs_tuple, {})
        
        done = False
        ep_reward = 0.0
        ep_steps = 0
        route_completion = 0.0
        
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action, log_prob, val = policy.act(obs_tensor)
            
            next_obs, reward, terminated, truncated, info = env.step(action.numpy())
            
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
            
            obs = next_obs
        
        print(f"Episode {episode_idx} finished in {ep_steps} steps with total reward {ep_reward:.2f} "
              f"and route_completion {route_completion:.2f}")
    
    success_rate = success_count / num_episodes
    avg_steps = np.mean(successful_steps) if successful_steps else 0

    print(f"\nSuccess Rate: {success_rate*100:.2f}% ({success_count}/{num_episodes})")
    print(f"Average Steps for Successful Episodes: {avg_steps:.2f}")
    
    env.close()


if __name__ == "__main__":
    model_path = "StraightCornerRoadPytorchAgent/ppo_final_model.pth"
    evaluate_model(model_path=model_path, num_episodes=1000)