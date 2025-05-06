import torch
import numpy as np
import time

from train import ActorCritic
from environmentStraightCorner import StraightCornerEnv

def evaluate_model(model_path, num_episodes=3):
    """
    Evaluate the trained PPO model for `num_episodes` with rendering enabled.
    """
    eval_config = {
        "use_render": True, 

    }
    env = StraightCornerEnv(config=eval_config)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    policy = ActorCritic(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()  

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
            if info.get("route_completion", 0.0) > 0.98:
                print("Manually ending episode due to route completion.")
                break   
            
            if has_failed:
                print("Agent failed (crash or out of road). Ending episode.")
                break
            obs = next_obs
            

        print(f"Episode {episode_idx} finished in {ep_steps} steps "
                f"with total reward {ep_reward:.2f} and route completion {route_completion:.2f}")

    env.close()


if __name__ == "__main__":
    model_path = "StraightCornerRoadPytorchAgent/ppo_final_model.pth"
    evaluate_model(model_path=model_path, num_episodes=3)