import os
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from flatland.envs.step_utils.states import TrainState
from reinforcement_learning.create_curriculum_envs import create_curriculum_env
from reinforcement_learning.my_observation_builder import MyObservationBuilder
from reinforcement_learning.my_policy import AttentionActorCritic, sanitize_action_masks

warnings.filterwarnings("ignore", message="Could not find path connecting all")


@dataclass
class CurriculumStage:
    name: str
    scene: str
    n_agents_range: Tuple[int, int]
    line_length: int
    promote_arrival_rate: float


CURRICULUM = [
    CurriculumStage("Stage0_Solo", "scene_1", (1, 2), 2, 0.85),
    CurriculumStage("Stage1_Easy", "scene_1", (4, 8), 2, 0.68),
    CurriculumStage("Stage2_Medium", "scene_1", (10, 18), 3, 0.70),
    CurriculumStage("Stage3_Hard", "scene_4", (18, 30), 3, 0.65),
    CurriculumStage("Stage4_FullMap", "scene_5", (30, 50), 4, 0.99),
]


class DecoupledCritic(nn.Module):
    def __init__(self, obs_dim: int = 96, global_dim: int = 6, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + global_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, global_state], dim=-1)
        return self.net(x).squeeze(-1)


class MAPPOAgent(nn.Module):
    def __init__(self, policy: AttentionActorCritic, global_dim: int = 6):
        super().__init__()
        self.policy = policy
        self.critic = DecoupledCritic(
            obs_dim=policy.obs_size, global_dim=global_dim, hidden_size=128
        )

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        mask: torch.Tensor,
        global_state: torch.Tensor,
        action: torch.Tensor | None = None,
    ):
        logits, _ = self.policy(obs)
        logits = logits.masked_fill(mask < 0.5, float("-inf"))
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(obs, global_state)
        return action, log_prob, entropy, value


def compute_global_state(env, obs_builder: MyObservationBuilder) -> np.ndarray:
    num_agents = max(1, env.get_num_agents())
    active = 0
    done = 0
    malf = 0
    stopped = 0
    total_wp_ratio = 0.0

    for h in env.get_agent_handles():
        agent = env.agents[h]
        if agent.state == TrainState.DONE:
            done += 1
        elif agent.state == TrainState.MALFUNCTION:
            active += 1
            malf += 1
        elif agent.state == TrainState.STOPPED:
            active += 1
            stopped += 1
        elif agent.state == TrainState.MOVING:
            active += 1

        n_wp = max(1, len(agent.waypoints) - 1)
        served = obs_builder.waypoint_index.get(h, 0)
        total_wp_ratio += min(1.0, served / n_wp)

    elapsed_ratio = env._elapsed_steps / max(1, env._max_episode_steps)
    return np.array(
        [
            active / num_agents,
            done / num_agents,
            malf / num_agents,
            stopped / num_agents,
            total_wp_ratio / num_agents,
            elapsed_ratio,
        ],
        dtype=np.float32,
    )


def get_agent_distance(env, obs_builder: MyObservationBuilder, handle: int) -> float:
    agent = env.agents[handle]
    if agent.state == TrainState.DONE:
        return 0.0
    pos = agent.position if agent.position is not None else agent.initial_position
    direction = agent.direction if agent.direction is not None else agent.initial_direction
    if pos is None or direction is None:
        return float(env.width + env.height)
    dmap = obs_builder._get_distance_map(handle)
    d = dmap[pos[0], pos[1], direction]
    return float(d) if np.isfinite(d) else float(env.width + env.height)


def train(
    total_updates: int = 800,
    steps_per_update: int = 2048,
    lr: float = 1.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    update_epochs: int = 4,
    minibatch_size: int = 512,
    team_reward_weight: float = 0.05,
    resume: bool = True,
    save_path: str = "reinforcement_learning/checkpoint_v5.pt",
    submission_save_path: str = "submission/checkpoint_v5.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Corridor-Yield MAPPO v5 on device: {device}")

    obs_builder = MyObservationBuilder(max_depth=3, with_action_mask=True)
    policy = AttentionActorCritic(checkpoint_path=None)
    mappo = MAPPOAgent(policy=policy, global_dim=6).to(device)
    optimizer = optim.AdamW(mappo.parameters(), lr=lr, eps=1e-5)

    stage_idx = 0
    if resume and os.path.exists(save_path):
        ckpt = torch.load(save_path, weights_only=False, map_location=device)
        mappo.policy.load_state_dict(ckpt["model"])
        if "critic" in ckpt:
            mappo.critic.load_state_dict(ckpt["critic"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "stage_idx" in ckpt:
            stage_idx = ckpt["stage_idx"]
        print(f"Resumed full MAPPO state from {save_path} at stage_idx={stage_idx}")

    stage = CURRICULUM[stage_idx]
    env = create_curriculum_env(
        scene=stage.scene,
        n_agents_range=stage.n_agents_range,
        line_length=stage.line_length,
    )
    env.obs_builder = obs_builder

    recent_arrivals = deque(maxlen=25)
    best_score = -1.0

    def reset_env_trackers():
        obs_d, _ = env.reset()
        handles = env.get_agent_handles()
        return (
            obs_d,
            {h: get_agent_distance(env, obs_builder, h) for h in handles},
            {h: obs_builder.waypoint_index.get(h, 0) for h in handles},
            {h: 0 for h in handles},
            {h: env.agents[h].position for h in handles},
            {h: False for h in handles},
            {h: [] for h in handles},
        )

    (
        obs_dict,
        prev_dists,
        prev_wps,
        stall_counters,
        prev_positions,
        deadlocked,
        trajectories,
    ) = reset_env_trackers()

    for update in range(1, total_updates + 1):
        t0 = time.time()
        completed_trajs: List[Tuple[List[tuple], float]] = []

        collected_transitions = 0
        while collected_transitions < steps_per_update:
            active_handles = [
                h
                for h in env.get_agent_handles()
                if env.agents[h].state != TrainState.DONE
            ]

            non_deadlocked = [h for h in active_handles if not deadlocked.get(h, False)]
            if not active_handles or not non_deadlocked:
                if active_handles:
                    n_total = max(1, env.get_num_agents())
                    n_arrived = sum(
                        1
                        for h in env.get_agent_handles()
                        if env.agents[h].state == TrainState.DONE
                    )
                    recent_arrivals.append(n_arrived / n_total)
                for h, traj in trajectories.items():
                    if traj:
                        completed_trajs.append((traj, 0.0))
                (
                    obs_dict,
                    prev_dists,
                    prev_wps,
                    stall_counters,
                    prev_positions,
                    deadlocked,
                    trajectories,
                ) = reset_env_trackers()
                continue

            obs_arr = np.asarray([obs_dict[h][: policy.obs_size] for h in active_handles], dtype=np.float32)
            raw_mask_arr = np.asarray(
                [obs_dict[h][policy.obs_size : policy.obs_size + policy.n_actions] for h in active_handles],
                dtype=np.float32,
            )

            g_state = compute_global_state(env, obs_builder)
            g_arr = np.repeat(g_state[None, :], len(active_handles), axis=0)

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_arr).to(device)
                mask_t = sanitize_action_masks(obs_t, torch.from_numpy(raw_mask_arr).to(device))
                g_t = torch.from_numpy(g_arr).to(device)
                actions_t, logprobs_t, _, values_t = mappo.get_action_and_value(obs_t, mask_t, g_t)

            mask_arr = mask_t.cpu().numpy()
            actions_np = actions_t.cpu().numpy()
            logprobs_np = logprobs_t.cpu().numpy()
            values_np = values_t.cpu().numpy()

            action_dict = {h: int(a) for h, a in zip(active_handles, actions_np)}
            next_obs_dict, _, dones, _ = env.step(action_dict)

            indiv_rewards = np.zeros(len(active_handles), dtype=np.float32)
            is_done_flags = []
            just_deadlocked = [False] * len(active_handles)

            for idx, h in enumerate(active_handles):
                agent = env.agents[h]
                act = int(actions_np[idx])
                must_yield = obs_arr[idx, 3] > 0.5

                # 1. Waypoint completion bonus
                curr_wp = obs_builder.waypoint_index.get(h, 0)
                wp_bonus = 1.0 * max(0, curr_wp - prev_wps[h])
                prev_wps[h] = curr_wp

                # 2. Shortest-path distance reduction
                curr_dist = get_agent_distance(env, obs_builder, h)
                dist_delta = np.clip(prev_dists[h] - curr_dist, -2.0, 2.0)
                progress_r = 0.1 * dist_delta if wp_bonus == 0 else 0.1
                prev_dists[h] = curr_dist

                # 3. Priority-aware yield & routing alignment shaping
                yield_r = 0.0
                if must_yield:
                    if act in (0, 4) and stall_counters[h] <= 8:
                        yield_r = 0.025
                    elif act not in (0, 4):
                        yield_r = -0.05
                elif act in (1, 2, 3) and obs_arr[idx, act - 1] > 0.5:
                    yield_r = 0.03 if act in (1, 3) else 0.01

                # 4. Step time cost
                time_cost = -0.015

                # 5. Stall & Deadlock detection
                stall_pen = 0.0
                if agent.state in (TrainState.MOVING, TrainState.STOPPED, TrainState.READY_TO_DEPART):
                    if agent.position == prev_positions[h]:
                        stall_counters[h] += 1
                        opp_ahead = obs_arr[idx, 14:17].max()
                        malf_ahead = obs_arr[idx, 36:39].max()
                        same_ahead = obs_arr[idx, 18:21].max()
                        ed_wait = obs_arr[idx, 43]
                        if opp_ahead == 0 and malf_ahead == 0 and same_ahead == 0 and ed_wait == 0:
                            stall_pen = -0.05
                        if stall_counters[h] >= 22 and not deadlocked[h]:
                            deadlocked[h] = True
                            just_deadlocked[idx] = True
                            stall_pen = -1.5
                    else:
                        stall_counters[h] = 0
                prev_positions[h] = agent.position

                # 6. Arrival bonus
                is_done = bool(dones.get(h, False) or agent.state == TrainState.DONE)
                is_done_flags.append(is_done)
                arrival_bonus = 5.0 if agent.state == TrainState.DONE else 0.0

                indiv_rewards[idx] = wp_bonus + progress_r + yield_r + time_cost + stall_pen + arrival_bonus

            team_mean_r = float(np.mean(indiv_rewards)) if len(indiv_rewards) > 0 else 0.0
            mixed_rewards = (1.0 - team_reward_weight) * indiv_rewards + team_reward_weight * team_mean_r

            for idx, h in enumerate(active_handles):
                if deadlocked[h] and not just_deadlocked[idx]:
                    continue

                is_decision = float(mask_arr[idx].sum() > 1.5)
                term_flag = float(is_done_flags[idx] or just_deadlocked[idx])
                trajectories[h].append(
                    (
                        obs_arr[idx],
                        mask_arr[idx],
                        g_arr[idx],
                        actions_np[idx],
                        logprobs_np[idx],
                        float(mixed_rewards[idx]),
                        term_flag,
                        values_np[idx],
                        is_decision,
                    )
                )
                collected_transitions += 1

                if term_flag > 0.5 and trajectories[h]:
                    completed_trajs.append((trajectories[h], 0.0))
                    trajectories[h] = []

            obs_dict = next_obs_dict

            if dones.get("__all__", False):
                n_total = max(1, env.get_num_agents())
                n_arrived = sum(
                    1
                    for h in env.get_agent_handles()
                    if env.agents[h].state == TrainState.DONE
                )
                recent_arrivals.append(n_arrived / n_total)

                for h, traj in trajectories.items():
                    if traj:
                        completed_trajs.append((traj, 0.0))

                (
                    obs_dict,
                    prev_dists,
                    prev_wps,
                    stall_counters,
                    prev_positions,
                    deadlocked,
                    trajectories,
                ) = reset_env_trackers()

        unfinished_handles = [h for h, traj in trajectories.items() if traj]
        if unfinished_handles:
            next_g = compute_global_state(env, obs_builder)
            next_obs_arr = np.asarray(
                [obs_dict[h][: policy.obs_size] for h in unfinished_handles], dtype=np.float32
            )
            next_g_arr = np.repeat(next_g[None, :], len(unfinished_handles), axis=0)
            with torch.no_grad():
                boot_vals = mappo.critic(
                    torch.from_numpy(next_obs_arr).to(device),
                    torch.from_numpy(next_g_arr).to(device),
                ).cpu().numpy()
            for h, b_val in zip(unfinished_handles, boot_vals):
                is_term = env.agents[h].state == TrainState.DONE or deadlocked.get(h, False)
                completed_trajs.append((trajectories[h], 0.0 if is_term else float(b_val)))

        b_obs, b_mask, b_g, b_act, b_logp, b_adv, b_ret, b_dec = [], [], [], [], [], [], [], []
        for traj, boot_val in completed_trajs:
            T = len(traj)
            advs = np.zeros(T, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(T)):
                _, _, _, _, _, r, done_flag, val, _ = traj[t]
                next_val = boot_val if t == T - 1 else traj[t + 1][7]
                next_non_terminal = 1.0 - done_flag
                delta = r + gamma * next_val * next_non_terminal - val
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                advs[t] = last_gae

            for t in range(T):
                o, m, g, a, lp, _, _, val, dec = traj[t]
                b_obs.append(o)
                b_mask.append(m)
                b_g.append(g)
                b_act.append(a)
                b_logp.append(lp)
                b_adv.append(advs[t])
                b_ret.append(advs[t] + val)
                b_dec.append(dec)

        b_obs_t = torch.from_numpy(np.asarray(b_obs, dtype=np.float32)).to(device)
        b_mask_t = torch.from_numpy(np.asarray(b_mask, dtype=np.float32)).to(device)
        b_g_t = torch.from_numpy(np.asarray(b_g, dtype=np.float32)).to(device)
        b_act_t = torch.from_numpy(np.asarray(b_act, dtype=np.int64)).to(device)
        b_logp_t = torch.from_numpy(np.asarray(b_logp, dtype=np.float32)).to(device)
        b_adv_t = torch.from_numpy(np.asarray(b_adv, dtype=np.float32)).to(device)
        b_ret_t = torch.from_numpy(np.asarray(b_ret, dtype=np.float32)).to(device)
        b_dec_t = torch.from_numpy(np.asarray(b_dec, dtype=np.float32)).to(device)

        dec_mask_bool = b_dec_t > 0.5
        if dec_mask_bool.sum() > 1:
            adv_mean = b_adv_t[dec_mask_bool].mean()
            adv_std = b_adv_t[dec_mask_bool].std() + 1e-8
            b_adv_t = (b_adv_t - adv_mean) / adv_std
        else:
            b_adv_t = (b_adv_t - b_adv_t.mean()) / (b_adv_t.std() + 1e-8)

        N = b_obs_t.shape[0]
        indices = np.arange(N)
        for _ in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                _, new_logp, entropy, new_val = mappo.get_action_and_value(
                    b_obs_t[mb_idx], b_mask_t[mb_idx], b_g_t[mb_idx], b_act_t[mb_idx]
                )
                logratio = new_logp - b_logp_t[mb_idx]
                ratio = logratio.exp()

                mb_dec = b_dec_t[mb_idx]
                dec_denom = mb_dec.sum().clamp(min=1.0)

                pg_loss1 = -b_adv_t[mb_idx] * ratio
                pg_loss2 = -b_adv_t[mb_idx] * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                pg_loss = (torch.max(pg_loss1, pg_loss2) * mb_dec).sum() / dec_denom

                v_loss = 0.5 * ((new_val - b_ret_t[mb_idx]) ** 2).mean()
                ent_loss = (entropy * mb_dec).sum() / dec_denom

                loss = pg_loss - ent_coef * ent_loss + vf_coef * v_loss
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean().item()
                if approx_kl > 0.03:
                    break
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(mappo.parameters(), 0.5)
                optimizer.step()

        mean_arrival = float(np.mean(recent_arrivals)) if recent_arrivals else 0.0
        dt = time.time() - t0
        print(
            f"[Update {update:03d}/{total_updates}] {stage.name} | "
            f"ArrivalRate(25ep): {mean_arrival:.3f} | Transitions: {N} | {dt:.2f}s"
        )

        composite = stage_idx + mean_arrival
        if len(recent_arrivals) >= 10 and composite > best_score:
            best_score = composite
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            os.makedirs(os.path.dirname(submission_save_path), exist_ok=True)
            ckpt = {
                "model": mappo.policy.state_dict(),
                "critic": mappo.critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stage_idx": stage_idx,
                "stage": stage.name,
                "arrival_rate": mean_arrival,
            }
            torch.save(ckpt, save_path)
            torch.save(ckpt, submission_save_path)
            print(f"  --> Saved checkpoint ({stage.name}, arrival={mean_arrival:.3f})")

        if (
            len(recent_arrivals) >= 15
            and mean_arrival >= stage.promote_arrival_rate
            and stage_idx < len(CURRICULUM) - 1
        ):
            stage_idx += 1
            stage = CURRICULUM[stage_idx]
            recent_arrivals.clear()
            print(f"\n=== Promoting to {stage.name} (scene={stage.scene}, agents={stage.n_agents_range}) ===\n")
            env = create_curriculum_env(
                scene=stage.scene,
                n_agents_range=stage.n_agents_range,
                line_length=stage.line_length,
            )
            env.obs_builder = obs_builder
            (
                obs_dict,
                prev_dists,
                prev_wps,
                stall_counters,
                prev_positions,
                deadlocked,
                trajectories,
            ) = reset_env_trackers()


if __name__ == "__main__":
    train()