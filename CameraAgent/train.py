import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os
import datetime
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from env import MyEnvNoTraffic

# --- Checkpoint Saving Function --
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


# --- CNN-based Actor-Critic ----
class ActorCritic(nn.Module):
    def __init__(self, image_channels, state_dim, action_dim,
                    hidden_sizes=(256, 256), log_std_init=-0.5):
        super().__init__()
        # CNN for image processing (9 channels: 3 frames × 3 RGB)
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Hardcoded conv output size for input (9, 60, 80) → 64 × 4 × 6 = 1536
        conv_output_size = 64 * 4 * 6

        # Shared MLP after CNN + state
        self.fc_shared = nn.Sequential(
            nn.Linear(conv_output_size + state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_sizes[1], action_dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        self.v_head = nn.Linear(hidden_sizes[1], 1)

    def forward(self, img, state):
        feat = self.cnn(img)
        x = torch.cat([feat, state], dim=-1)
        hidden = self.fc_shared(x)
        mu = self.mu_head(hidden)
        std = torch.exp(self.log_std)
        value = self.v_head(hidden)
        return mu, std, value

    def act(self, img, state):
        mu, std, value = self.forward(img, state)
        dist = Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value


# --- Rollout Buffer for dict obs ----
class RolloutBuffer:
    def __init__(self, size, image_channels, img_h, img_w, state_dim,
                    action_dim, gamma=0.99, lam=0.95):
        self.size = size
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.start_ptr = 0
        self.img_buf = torch.zeros((size, image_channels, img_h, img_w), dtype=torch.float32)
        self.state_buf = torch.zeros((size, state_dim), dtype=torch.float32)
        self.act_buf = torch.zeros((size, action_dim), dtype=torch.float32)
        self.logp_buf = torch.zeros(size, dtype=torch.float32)
        self.rew_buf = torch.zeros(size, dtype=torch.float32)
        self.val_buf = torch.zeros(size, dtype=torch.float32)
        self.adv_buf = torch.zeros(size, dtype=torch.float32)
        self.ret_buf = torch.zeros(size, dtype=torch.float32)

    def store(self, obs_img, obs_state, act, logp, rew, val):
        if self.ptr >= self.size:
            print("Warning: RolloutBuffer overflow!")
            return
        self.img_buf[self.ptr] = obs_img
        self.state_buf[self.ptr] = obs_state
        self.act_buf[self.ptr] = act
        self.logp_buf[self.ptr] = logp
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.ptr += 1

    def _discount_cumsum(self, x, discount):
        result = torch.zeros_like(x)
        running = 0.0
        for t in reversed(range(x.shape[0])):
            running = x[t] + discount * running
            result[t] = running
        return result

    def finish_path(self, last_val=0.0):
        if self.ptr == self.start_ptr:
            return
        slice_ = slice(self.start_ptr, self.ptr)
        rews = torch.cat([self.rew_buf[slice_], torch.tensor([last_val])])
        vals = torch.cat([self.val_buf[slice_], torch.tensor([last_val])])
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[slice_] = self._discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[slice_] = self._discount_cumsum(rews[:-1], self.gamma)
        self.start_ptr = self.ptr

    def get(self):
        if self.ptr == 0:
            return (torch.zeros((0, *self.img_buf.shape[1:])),
                    torch.zeros((0, self.state_buf.shape[1])),
                    torch.zeros((0, self.act_buf.shape[1])),
                    torch.zeros(0),
                    torch.zeros(0),
                    torch.zeros(0))
        img_batch = self.img_buf[:self.ptr]
        state_batch = self.state_buf[:self.ptr]
        act_batch = self.act_buf[:self.ptr]
        logp_batch = self.logp_buf[:self.ptr]
        adv_batch = self.adv_buf[:self.ptr]
        ret_batch = self.ret_buf[:self.ptr]
        adv_mean, adv_std = adv_batch.mean(), adv_batch.std()
        adv_batch = (adv_batch - adv_mean) / (adv_std + 1e-8)
        self.ptr = 0
        return img_batch, state_batch, act_batch, logp_batch, adv_batch, ret_batch

# --- PPO Update ---
def ppo_update(policy, optimizer, buffer, writer, current_step,
               clip_ratio=0.2, vf_coef=0.5, ent_coef=0.001,
               epochs=10, batch_size=256, grad_clip_norm=0.5):
    img_b, state_b, act_b, logp_old_b, adv_b, ret_b = buffer.get()
    if img_b.shape[0] == 0:
        print("Warning: Empty buffer, skipping update.")
        return
    ret_mean, ret_std = ret_b.mean(), ret_b.std()
    ret_norm = (ret_b - ret_mean) / (ret_std + 1e-8)

    data_size = img_b.shape[0]
    bs = min(batch_size, data_size)

    pl_list, vl_list, ent_list, kl_list = [], [], [], []
    for _ in range(epochs):
        idxs = torch.randperm(data_size)
        for start in range(0, data_size, bs):
            mb = idxs[start:start+bs]
            mb_img, mb_state = img_b[mb], state_b[mb]
            mb_act, mb_logp_old = act_b[mb], logp_old_b[mb]
            mb_adv, mb_ret = adv_b[mb], ret_norm[mb]

            mu, std, val = policy(mb_img, mb_state)
            dist = Normal(mu, std)
            logp = dist.log_prob(mb_act).sum(-1)
            ratio = torch.exp(logp - mb_logp_old)
            with torch.no_grad():
                kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            kl_list.append(kl)

            clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * mb_adv
            pl = -(torch.min(ratio * mb_adv, clip_adv)).mean()
            vl = F.mse_loss(val.squeeze(-1), mb_ret)
            ent = dist.entropy().mean()
            loss = pl + vf_coef * vl - ent_coef * ent

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            optimizer.step()

            pl_list.append(pl.item())
            vl_list.append(vl.item())
            ent_list.append(ent.item())

    writer.add_scalar("Update/PolicyLoss", sum(pl_list)/len(pl_list), current_step)
    writer.add_scalar("Update/ValueLoss", sum(vl_list)/len(vl_list), current_step)
    writer.add_scalar("Update/Entropy", sum(ent_list)/len(ent_list), current_step)
    writer.add_scalar("Update/ApproxKL", sum(kl_list)/len(kl_list), current_step)
    writer.add_scalar("Update/PolicyStd", policy.log_std.exp().mean().item(), current_step)
    writer.add_scalar("Update/ReturnMean", ret_mean.item(), current_step)
    writer.add_scalar("Update/ReturnStd", ret_std.item(), current_step)


# === Main Training Function ===
def train_ppo_with_tensorboard():
    total_steps = 3_000_000
    buffer_size = 4096
    lr = 5e-5; gamma, lam = 0.99, 0.95
    ppo_epochs, ppo_bs, ppo_clip = 10, 256, 0.3
    vf_coef, ent_coef = 0.3, 1e-4; grad_clip = 0.5
    ckpt_int = 100_000; log_std_init = -1.0

    env = MyEnvNoTraffic()
    obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()

    img_h, img_w, C, T = env.observation_space['image'].shape
    img_ch = C * T
    state_dim = env.observation_space['state'].shape[0]
    act_dim = env.action_space.shape[0]

    policy = ActorCritic(img_ch, state_dim, act_dim, log_std_init=log_std_init)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    buffer = RolloutBuffer(buffer_size, img_ch, img_h, img_w, state_dim,
                            act_dim, gamma=gamma, lam=lam)

    base_dir = "./logs/ppo_metadrive_cnn"
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"PPO_CNN_{now}")
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    steps, ep_count = 0, 0
    while steps < total_steps:
        rollout_steps = 0; rollout_done = False
        while buffer.ptr < buffer.size:
            img = torch.tensor(obs['image'], dtype=torch.float32)
            img = img.permute(3,2,0,1).reshape(img_ch, img_h, img_w)
            state = torch.tensor(obs['state'], dtype=torch.float32)

            with torch.no_grad():
                act, logp, val = policy.act(img.unsqueeze(0), state.unsqueeze(0))
                val_s = val.item()
            act_np = act.squeeze(0).cpu().numpy()

            next_obs, rew, term, trunc, info = env.step(act_np)
            done = term or trunc
            buffer.store(img, state, act.squeeze(0), logp.squeeze(0), rew, val_s)

            obs = next_obs; rollout_steps += 1; rollout_done = done
            if done:
                ep_count += 1
                ep_rew = info.get("episode_reward", 0)
                ep_len = info.get("episode_length", 0)
                ep_route = info.get("route_completion", 0.0)
                success = int(term and ep_route >= 0.99)
                curr_step = steps + buffer.ptr
                writer.add_scalar("Charts/EpisodeReward", ep_rew, curr_step)
                writer.add_scalar("Charts/EpisodeLength", ep_len, curr_step)
                writer.add_scalar("Charts/RouteCompletion", ep_route, curr_step)
                writer.add_scalar("Charts/SuccessRate", success, curr_step)
                writer.flush()

                print(f"\n[Ep {ep_count}] Len: {ep_len} | Rew: {ep_rew} | Route: {ep_route:.2f} | Success: {success} | progress= {steps}/{total_steps}")
                buffer.finish_path(0.0)
                obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
                if buffer.ptr == buffer.size:
                    break
            if buffer.ptr == buffer.size:
                break

        last_val = 0.0
        if not rollout_done:
            img = torch.tensor(obs['image'], dtype=torch.float32)
            img = img.permute(3,2,0,1).reshape(img_ch, img_h, img_w)
            state = torch.tensor(obs['state'], dtype=torch.float32)
            with torch.no_grad():
                _, _, lv = policy.forward(img.unsqueeze(0), state.unsqueeze(0))
                last_val = lv.item()
        buffer.finish_path(last_val)

        if rollout_steps > 0:
            before = steps
            ppo_update(policy, optimizer, buffer, writer, before,
                        clip_ratio=ppo_clip, vf_coef=vf_coef,
                        ent_coef=ent_coef, epochs=ppo_epochs,
                        batch_size=ppo_bs, grad_clip_norm=grad_clip)
            steps += rollout_steps
        else:
            print("Warning: no steps collected this rollout.")

        writer.add_scalar("Training/TotalSteps", steps, steps)
        if steps // ckpt_int > (steps - rollout_steps) // ckpt_int:
            save_checkpoint(steps, policy, optimizer, ep_count, ckpt_dir)

    final_path = os.path.join(run_dir, "ppo_final_model.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"\n✅ Final model saved to {final_path}")

    writer.close()
    env.close()
    print("\nTraining finished.")

if __name__ == "__main__":
    torch.manual_seed(42)
    print("Starting PPO training with CNN...")
    train_ppo_with_tensorboard()
    print("Script finished.")