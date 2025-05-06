import torch
import numpy as np
import time

from train import ActorCritic
from env import MyEnvNoTraffic
import cv2
def evaluate_model(model_path, num_episodes=3):

    eval_config = {
        "use_render": True, 
        "interface_panel": ["rgb_camera", "dashboard"],

    }
    env = MyEnvNoTraffic(config=eval_config)
    
    # Get dimensions from the dictionary observation space
    img_h, img_w, C, T = env.observation_space['image'].shape
    img_ch = C * T  # Combine channels and frames/color dimensions
    state_dim = env.observation_space['state'].shape[0]
    act_dim = env.action_space.shape[0]
    
    # Initialize the model with the correct parameters
    policy = ActorCritic(img_ch, state_dim, act_dim)
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
            # Process image observation - reshape and permute like in training
            img = torch.tensor(obs['image'], dtype=torch.float32)
            img = img.permute(3, 2, 0, 1).reshape(img_ch, img_h, img_w)
            state = torch.tensor(obs['state'], dtype=torch.float32)
            
            with torch.no_grad():
                action, log_prob, val = policy.act(img.unsqueeze(0), state.unsqueeze(0))
            
            # Before step, convert action to numpy and clip
            action_np = action.squeeze().numpy()  
            action_np = np.clip(action_np, -1.0, 1.0)  
            next_obs, reward, terminated, truncated, info = env.step(action_np)

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
            # Update obs
            obs = next_obs
            cv2.imshow('image', obs['image'][...,-1])
            cv2.waitKey(1)

        print(f"Episode {episode_idx} finished in {ep_steps} steps "
                f"with total reward {ep_reward:.2f} and route completion {route_completion:.2f}")

    env.close()


if __name__ == "__main__":
    model_path = "ppo_final_model.pth"
    evaluate_model(model_path=model_path, num_episodes=3)