import torch
import numpy as np
import gymnasium as gym
import os 

from train import ActorCritic
from env import MyEnvNoTraffic

def evaluate_model(model_path, num_episodes=100):
    print(f"Evaluating model: {model_path}")

    try:
        env = MyEnvNoTraffic()
        print("Environment initialized successfully.")
    except Exception as e:
        print(f"Error initializing environment: {e}")
        return

    try:
        obs_space = env.observation_space
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError(f"Expected observation space to be a Dict, but got {type(obs_space)}")

        img_shape = obs_space['image'].shape # Should be (H, W, C, T)
        state_dim = obs_space['state'].shape[0]
        act_dim = env.action_space.shape[0]

        img_h, img_w, C, T = img_shape
        img_channels = C * T # Total channels = Channels per frame * Num frames

        print(f"Image Shape (H, W, C, T): {img_shape}")
        print(f"Calculated Image Channels: {img_channels}")
        print(f"State Dimension: {state_dim}")
        print(f"Action Dimension: {act_dim}")

    except Exception as e:
        print(f"Error accessing environment spaces: {e}")
        env.close()
        return

    try:
        policy = ActorCritic(image_channels=img_channels, state_dim=state_dim, action_dim=act_dim)

        policy.load_state_dict(torch.load(model_path))
        policy.eval() 
        print("Policy loaded and set to evaluation mode.")

    except Exception as e:
        print(f"Error loading policy model: {e}")
        env.close()
        return

    # --- Evaluation Loop ---
    success_count = 0
    successful_steps = []
    all_rewards = []
    all_route_completions = []

    for episode_idx in range(1, num_episodes + 1):
        try:
            # Reset environment and get initial dictionary observation
            reset_output = env.reset()
            obs = reset_output[0] if isinstance(reset_output, tuple) else reset_output

            terminated = False
            truncated = False
            ep_reward = 0.0
            ep_steps = 0
            route_completion = 0.0

            while not (terminated or truncated):
                # Process image observation - reshape and permute like in training
                img = torch.tensor(obs['image'], dtype=torch.float32)
                img = img.permute(3, 2, 0, 1).reshape(img_channels, img_h, img_w)
                state = torch.tensor(obs['state'], dtype=torch.float32)
                
                with torch.no_grad():
                    action, log_prob, val = policy.act(img.unsqueeze(0), state.unsqueeze(0))
                
                # Before step, convert action to numpy and clip 
                action_np = action.squeeze().numpy()  # Remove any extra dimensions
                action_np = np.clip(action_np, -1.0, 1.0)  
                next_obs, reward, terminated, truncated, info = env.step(action_np)

                obs = next_obs # Update observation for next step
                ep_reward += reward
                ep_steps += 1
                route_completion = info.get("route_completion", route_completion) # Keep last known value if not present
                if route_completion >= 0.98:
                    success_count += 1
                    successful_steps.append(ep_steps)
                    print(f"Episode {episode_idx}: Success! Steps: {ep_steps}, Reward: {ep_reward:.2f}, Route: {route_completion:.3f}")
                    terminated = True # End episode on success criteria met

            if not (terminated or truncated): # Should not happen if loop condition is correct, but for safety
                    print(f"Warning: Episode {episode_idx} ended unexpectedly.")

            all_rewards.append(ep_reward)
            all_route_completions.append(route_completion)

            if not (route_completion >= 0.98): 
                    print(f"Episode {episode_idx}: Finished. Steps: {ep_steps}, Reward: {ep_reward:.2f}, Route: {route_completion:.3f}")


        except Exception as e:
            print(f"Error during episode {episode_idx}: {e}")

    print("\n--- Evaluation Summary ---")
    if num_episodes > 0:
        success_rate = success_count / num_episodes
        avg_steps_success = np.mean(successful_steps) if successful_steps else 0
        avg_reward = np.mean(all_rewards) if all_rewards else 0
        avg_route_completion = np.mean(all_route_completions) if all_route_completions else 0

        print(f"Total Episodes: {num_episodes}")
        print(f"Successful Episodes (Route >= 0.98): {success_count}")
        print(f"Success Rate: {success_rate * 100:.2f}%")
        print(f"Average Steps for Successful Episodes: {avg_steps_success:.2f}")
        print(f"Average Reward (All Episodes): {avg_reward:.2f}")
        print(f"Average Route Completion (All Episodes): {avg_route_completion:.3f}")
    else:
        print("No episodes were run.")

    # --- Cleanup ---
    env.close()
    print("Environment closed.")


if __name__ == "__main__":

    MODEL_PATH = "CameraAgent/ppo_final_model.pth"  

    NUM_EVAL_EPISODES = 100

    evaluate_model(model_path=MODEL_PATH, num_episodes=NUM_EVAL_EPISODES)

    print("\nEvaluation script finished.")