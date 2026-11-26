import copy
import os
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from flatland.envs.step_utils.states import TrainState
from reinforcement_learning.create_curriculum_envs import create_curriculum_env
from reinforcement_learning.my_observation_builder import MyObservationBuilder
from reinforcement_learning.my_policy import AttentionDuelingDQN, sanitize_action_masks

warnings.filterwarnings("ignore", message="Could not find path connecting all")


@dataclass
class CurriculumStage:
    name: str
    scene: str
    n_agents_range: Tuple[int, int]
    line_length: int
    promote_arrival_rate: float
    lr: float


CURRICULUM = [
    CurriculumStage("Stage0_Solo", "scene_1", (1, 2), 2, 0.82, 2.5e-4),
    CurriculumStage("Stage1_Easy", "scene_1", (4, 8), 2, 0.70, 1.5e-4),
    CurriculumStage("Stage2_Medium", "scene_1", (10, 18), 3, 0.55, 5.0e-5),
    CurriculumStage("Stage3_Hard", "scene_4", (18, 30), 3, 0.60, 7.0e-5),
    CurriculumStage("Stage4_FullMap", "scene_5", (30, 50), 4, 0.99, 5.0e-5),
]


class PrioritizedMacroReplayBuffer:
    """Semi-MDP Prioritized Experience Replay (PER) storing macro-transitions with gamma^k."""

    def __init__(
        self,
        capacity: int = 60_000,
        obs_dim: int = 136,
        n_actions: int = 5,
        alpha: float = 0.6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.mask = np.zeros((capacity, n_actions), dtype=np.float32)
        self.act = np.zeros(capacity, dtype=np.int64)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.gamma_k = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_mask = np.zeros((capacity, n_actions), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, mask, act, rew, gamma_k, next_obs, next_mask, done):
        idx = self.ptr
        self.obs[idx] = obs
        self.mask[idx] = mask
        self.act[idx] = act
        self.rew[idx] = rew
        self.gamma_k[idx] = gamma_k
        self.next_obs[idx] = next_obs
        self.next_mask[idx] = next_mask
        self.done[idx] = done
        self.priorities[idx] = self.max_priority
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int, beta: float, device: torch.device):
        prios = self.priorities[: self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        idx = np.random.choice(self.size, size=batch_size, p=probs)
        weights = (self.size * probs[idx]) ** (-beta)
        weights /= weights.max()

        return (
            idx,
            torch.from_numpy(weights.astype(np.float32)).to(device),
            torch.from_numpy(self.obs[idx]).to(device),
            torch.from_numpy(self.mask[idx]).to(device),
            torch.from_numpy(self.act[idx]).to(device),
            torch.from_numpy(self.rew[idx]).to(device),
            torch.from_numpy(self.gamma_k[idx]).to(device),
            torch.from_numpy(self.next_obs[idx]).to(device),
            torch.from_numpy(self.next_mask[idx]).to(device),
            torch.from_numpy(self.done[idx]).to(device),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        new_prios = np.clip(np.abs(td_errors), 0.0, 5.0) + 1e-4
        self.priorities[indices] = new_prios
        self.max_priority = max(self.max_priority, float(new_prios.max()))


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
    total_episodes: int = 2500,
    gamma: float = 0.95,
    tau: float = 0.002,
    batch_size: int = 256,
    learning_starts: int = 512,
    train_freq: int = 4,
    grad_steps: int = 1,
    coop_weight: float = 0.20,
    eps_start: float = 0.08,
    eps_end: float = 0.02,
    eps_decay_episodes: int = 100,
    resume: bool = True,
    save_path: str = "reinforcement_learning/checkpoint_dqn.pt",
    submission_save_path: str = "submission/checkpoint_dqn.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Cooperative PER Semi-MDP Dueling DQN on device: {device}")

    obs_builder = MyObservationBuilder(max_depth=3, with_action_mask=True)
    q_net = AttentionDuelingDQN(checkpoint_path=None).to(device)
    target_net = copy.deepcopy(q_net).to(device)
    target_net.eval()

    stage_idx = 0
    stage = CURRICULUM[stage_idx]
    optimizer = optim.AdamW(q_net.parameters(), lr=stage.lr, weight_decay=1e-5)
    buffer = PrioritizedMacroReplayBuffer(
        capacity=60_000, obs_dim=q_net.obs_size, n_actions=q_net.n_actions, alpha=0.3
    )

    if resume and os.path.exists(save_path):
        ckpt = torch.load(save_path, weights_only=False, map_location=device)
        q_net.load_state_dict(ckpt["model"])
        target_net.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "stage_idx" in ckpt:
            stage_idx = ckpt["stage_idx"]
            stage = CURRICULUM[stage_idx]
        for pg in optimizer.param_groups:
            pg["lr"] = stage.lr
        print(f"Resumed Cooperative PER DQN from {save_path} at stage_idx={stage_idx}")

    env = create_curriculum_env(
        scene=stage.scene,
        n_agents_range=stage.n_agents_range,
        line_length=stage.line_length,
    )
    env.obs_builder = obs_builder

    recent_arrivals = deque(maxlen=30)
    best_score = -1.0
    total_steps = 0
    stage_ep_count = 0

    for ep in range(1, total_episodes + 1):
        t0 = time.time()
        stage_ep_count += 1
        eps = max(
            eps_end,
            eps_start - (eps_start - eps_end) * (stage_ep_count / max(1, eps_decay_episodes)),
        )
        beta = min(1.0, 0.4 + 0.6 * (ep / max(1, total_episodes // 2)))

        obs_dict, _ = env.reset()
        handles = env.get_agent_handles()
        prev_dists = {h: get_agent_distance(env, obs_builder, h) for h in handles}
        prev_wps = {h: obs_builder.waypoint_index.get(h, 0) for h in handles}
        stall_counters = {h: 0 for h in handles}
        prev_positions = {h: env.agents[h].position for h in handles}
        deadlocked = {h: False for h in handles}
        pending = {h: None for h in handles}

        while True:
            active_handles = [
                h for h in handles
                if env.agents[h].state != TrainState.DONE and not deadlocked[h]
            ]
            if not active_handles:
                break

            obs_arr = np.asarray(
                [obs_dict[h][: q_net.obs_size] for h in active_handles], dtype=np.float32
            )
            raw_mask_arr = np.asarray(
                [obs_dict[h][q_net.obs_size : q_net.obs_size + q_net.n_actions] for h in active_handles],
                dtype=np.float32,
            )

            with torch.no_grad():
                obs_t = torch.from_numpy(obs_arr).to(device)
                mask_t = sanitize_action_masks(obs_t, torch.from_numpy(raw_mask_arr).to(device))
                q_vals = q_net(obs_t, mask_t)
                greedy_actions = q_vals.argmax(dim=-1).cpu().numpy()

            mask_arr = mask_t.cpu().numpy()
            actions_np = greedy_actions.copy()

            for idx in range(len(active_handles)):
                valid_acts = np.where(mask_arr[idx] > 0.5)[0]
                if len(valid_acts) > 1 and np.random.rand() < eps:
                    actions_np[idx] = np.random.choice(valid_acts)

            for idx, h in enumerate(active_handles):
                ag = env.agents[h]
                can_act = (ag.state != TrainState.MALFUNCTION) and (
                    ag.state != TrainState.MOVING or ag.speed_counter.is_cell_entry
                )
                is_decision_now = can_act and (mask_arr[idx].sum() > 1.5)
                if pending[h] is not None and is_decision_now:
                    p_obs, p_mask, p_act, p_rew, p_gk = pending[h]
                    buffer.add(
                        p_obs, p_mask, p_act, p_rew, p_gk,
                        obs_arr[idx], mask_arr[idx], 0.0
                    )
                    pending[h] = None

                if pending[h] is None and is_decision_now:
                    pending[h] = [obs_arr[idx].copy(), mask_arr[idx].copy(), int(actions_np[idx]), 0.0, 1.0]

            action_dict = {h: int(a) for h, a in zip(active_handles, actions_np)}
            next_obs_dict, _, dones, _ = env.step(action_dict)

            # Pass 1: Compute individual step rewards & terminal flags
            indiv_rewards = np.zeros(len(active_handles), dtype=np.float32)
            done_flags = [False] * len(active_handles)
            deadlock_flags = [False] * len(active_handles)

            for idx, h in enumerate(active_handles):
                agent = env.agents[h]

                curr_wp = obs_builder.waypoint_index.get(h, 0)
                wp_bonus = 0.5 * max(0, curr_wp - prev_wps[h])
                prev_wps[h] = curr_wp

                curr_dist = get_agent_distance(env, obs_builder, h)
                dist_delta = np.clip(prev_dists[h] - curr_dist, -2.0, 2.0)
                progress_r = 0.15 * dist_delta if wp_bonus == 0 else 0.15
                prev_dists[h] = curr_dist

                act = int(actions_np[idx])
                align_r = 0.0
                if obs_arr[idx, 7] > 0.5 and act in (1, 2, 3):
                    align_r = 0.15 if obs_arr[idx, act - 1] > 0.5 else -0.15
                time_cost = -0.015

                stall_pen = 0.0
                just_deadlocked = False
                ed_wait = obs_arr[idx, 43]
                malf_ahead = obs_arr[idx, 36:39].max()
                opp_ahead = obs_arr[idx, 14:17].max()
                same_ahead = obs_arr[idx, 18:21].max()

                if agent.state in (TrainState.MOVING, TrainState.STOPPED, TrainState.READY_TO_DEPART):
                    actually_stuck = (agent.position == prev_positions[h]) and (
                        agent.state != TrainState.MOVING or agent.speed_counter.is_cell_entry
                    )
                    if actually_stuck:
                        if ed_wait == 0.0 and malf_ahead == 0.0:
                            stall_counters[h] += 1
                            must_yield = obs_arr[idx, 3] > 0.5
                            if not must_yield:
                                stall_pen = -0.04
                            if stall_counters[h] >= 25 and not deadlocked[h]:
                                deadlocked[h] = True
                                just_deadlocked = True
                                stall_pen = -2.0
                    else:
                        stall_counters[h] = 0
                prev_positions[h] = agent.position

                is_done = bool(dones.get(h, False) or agent.state == TrainState.DONE)
                arrival_bonus = 2.0 if agent.state == TrainState.DONE else 0.0

                indiv_rewards[idx] = wp_bonus + progress_r + align_r + time_cost + stall_pen + arrival_bonus
                done_flags[idx] = is_done
                deadlock_flags[idx] = just_deadlocked

            # Pass 2: Cooperative Reward Mixing (allows 3 trains to yield for 5 trains' arrival)
            team_mean_r = float(np.mean(indiv_rewards)) if len(indiv_rewards) > 1 else 0.0
            eff_coop = coop_weight if len(active_handles) > 1 else 0.0

            for idx, h in enumerate(active_handles):
                reward = (1.0 - eff_coop) * float(indiv_rewards[idx]) + eff_coop * team_mean_r
                if pending[h] is not None:
                    pending[h][3] += pending[h][4] * reward
                    pending[h][4] *= gamma

                    if done_flags[idx] or deadlock_flags[idx]:
                        p_obs, p_mask, p_act, p_rew, p_gk = pending[h]
                        next_full = next_obs_dict[h]
                        n_obs = next_full[: q_net.obs_size]
                        n_mask = np.zeros(q_net.n_actions, dtype=np.float32)
                        buffer.add(
                            p_obs, p_mask, p_act, p_rew, p_gk,
                            n_obs, n_mask, 1.0
                        )
                        pending[h] = None

            obs_dict = next_obs_dict
            total_steps += 1

            if buffer.size >= learning_starts and total_steps % train_freq == 0:
                for _ in range(grad_steps):
                    (
                        b_idx, is_weights, s_obs, s_mask, s_act, s_rew,
                        s_gk, s_nobs, s_nmask, s_done
                    ) = buffer.sample(batch_size, beta, device)

                    q_pred = q_net(s_obs, s_mask).gather(1, s_act.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_online_q = q_net(s_nobs, s_nmask)
                        next_best_act = next_online_q.argmax(dim=-1, keepdim=True)
                        next_target_q = target_net(s_nobs, s_nmask).gather(1, next_best_act).squeeze(1)
                        has_valid_next = (s_nmask.sum(dim=-1) > 0.5).float()
                        y = s_rew + s_gk * next_target_q * has_valid_next * (1.0 - s_done)

                    td_err = y - q_pred
                    elementwise_loss = nn.functional.smooth_l1_loss(q_pred, y, reduction="none")
                    loss = (is_weights * elementwise_loss).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                    optimizer.step()

                    buffer.update_priorities(b_idx, td_err.detach().cpu().numpy())

                    with torch.no_grad():
                        for p, tp in zip(q_net.parameters(), target_net.parameters()):
                            tp.data.mul_(1.0 - tau).add_(tau * p.data)

            if dones.get("__all__", False):
                for h in handles:
                    if pending[h] is not None:
                        p_obs, p_mask, p_act, p_rew, p_gk = pending[h]
                        next_full = obs_dict[h]
                        n_obs = next_full[: q_net.obs_size]
                        n_mask = np.zeros(q_net.n_actions, dtype=np.float32)
                        buffer.add(
                            p_obs, p_mask, p_act, p_rew, p_gk,
                            n_obs, n_mask, 1.0
                        )
                        pending[h] = None
                break

        n_total = max(1, env.get_num_agents())
        n_arrived = sum(
            1 for h in handles if env.agents[h].state == TrainState.DONE
        )
        arrival_rate = n_arrived / n_total
        recent_arrivals.append(arrival_rate)
        mean_arrival = float(np.mean(recent_arrivals))

        if ep % 10 == 0:
            dt = time.time() - t0
            print(
                f"[Ep {ep:04d}/{total_episodes}] {stage.name} | "
                f"Arrival(30ep): {mean_arrival:.3f} | Eps: {eps:.3f} | PERBuf: {buffer.size} | {dt:.2f}s"
            )

        composite = stage_idx + mean_arrival
        if len(recent_arrivals) >= 15 and composite > best_score:
            best_score = composite
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            os.makedirs(os.path.dirname(submission_save_path), exist_ok=True)
            ckpt = {
                "model": q_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "stage_idx": stage_idx,
                "stage": stage.name,
                "arrival_rate": mean_arrival,
            }
            torch.save(ckpt, save_path)
            torch.save(ckpt, submission_save_path)
            print(f"  --> Saved DQN checkpoint ({stage.name}, arrival={mean_arrival:.3f})")

        if (
            len(recent_arrivals) >= 20
            and mean_arrival >= stage.promote_arrival_rate
            and stage_idx < len(CURRICULUM) - 1
        ):
            stage_idx += 1
            stage = CURRICULUM[stage_idx]
            recent_arrivals.clear()
            stage_ep_count = 0
            eps_start = 0.20
            for pg in optimizer.param_groups:
                pg["lr"] = stage.lr
            print(
                f"\n=== Promoting to {stage.name} "
                f"(scene={stage.scene}, agents={stage.n_agents_range}, lr={stage.lr}) ===\n"
            )
            env = create_curriculum_env(
                scene=stage.scene,
                n_agents_range=stage.n_agents_range,
                line_length=stage.line_length,
            )
            env.obs_builder = obs_builder


if __name__ == "__main__":
    train()