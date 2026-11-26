import os
import torch
import torch.nn as nn
import numpy as np
from typing import Any, List, Dict

from flatland.envs.rail_env_action import RailEnvActions


def sanitize_action_masks(features: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    1. On tracks (self_obs[5] == 1), disallow DO_NOTHING(0).
    2. For READY_TO_DEPART (self_obs[4] == 1), keep DO_NOTHING(0) open to hold inside station.
    3. On risk-free tracks with no oncoming/malfunction/same-ahead trains and no ed_wait,
       disallow unnecessary STOP(4) so trains cruise smoothly between decision points.
    """
    masks = masks.clone()
    on_track = features[:, 5] > 0.5
    has_move = masks[:, 1:].sum(dim=-1) > 0.5
    masks[on_track & has_move, 0] = 0.0

    opp_max = features[:, 14:17].max(dim=-1).values
    malf_max = features[:, 36:39].max(dim=-1).values
    stopped_same_max = features[:, [13, 17, 21]].max(dim=-1).values

    # 1. Safe station departure shield: hold inside station if exit corridor is contested/jammed
    ready_to_depart = features[:, 4] > 0.5
    unsafe_spawn = ready_to_depart & ((opp_max > 0.25) | (stopped_same_max > 0.5))
    masks[unsafe_spawn, 2] = 0.0

    # 2. Allow STOP(4) at safe switch entry when oncoming, malfunction, or stopped queue is ahead
    at_safe_switch_entry = (features[:, 9] > 0.5) & (features[:, 7] < 0.5)
    no_risk = (opp_max == 0.0) & (malf_max == 0.0) & (stopped_same_max <= 0.4)
    no_hold = features[:, 43] == 0.0
    has_dir = masks[:, 1:4].sum(dim=-1) > 0.5
    block_stop = on_track & no_hold & has_dir & ((~at_safe_switch_entry) | no_risk)
    masks[block_stop, 4] = 0.0
    return masks


class AttentionDuelingDQN(nn.Module):
    def __init__(
        self,
        self_dim: int = 52,
        neighbor_k: int = 12,
        neighbor_dim: int = 7,
        n_actions: int = 5,
        hidden_size: int = 128,
        checkpoint_path: str | None = "./submission/checkpoint_dqn.pt",
    ):
        super().__init__()
        self.self_dim = self_dim
        self.neighbor_k = neighbor_k
        self.neighbor_dim = neighbor_dim
        self.obs_size = self_dim + neighbor_k * neighbor_dim  # 136
        self.n_actions = n_actions

        self.self_encoder = nn.Sequential(
            nn.Linear(self_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        self.trunk = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, n_actions),
        )

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
            self.load_state_dict(ckpt["model"])
            self.eval()
        else:
            self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, obs: torch.Tensor, masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
            if masks is not None:
                masks = masks.unsqueeze(0)

        B = obs.shape[0]
        self_feat = obs[:, : self.self_dim]
        neigh_raw = obs[:, self.self_dim : self.obs_size].view(
            B, self.neighbor_k, self.neighbor_dim
        )

        neigh_pad_mask = neigh_raw.abs().sum(dim=-1) == 0
        all_empty = neigh_pad_mask.all(dim=-1, keepdim=True)
        safe_pad_mask = neigh_pad_mask & (~all_empty)

        h_self = self.self_encoder(self_feat)
        h_neigh = self.neighbor_encoder(neigh_raw)

        attn_out, _ = self.cross_attn(
            query=h_self.unsqueeze(1),
            key=h_neigh,
            value=h_neigh,
            key_padding_mask=safe_pad_mask,
        )
        attn_out = attn_out.squeeze(1)
        attn_out = torch.where(all_empty, torch.zeros_like(attn_out), attn_out)

        h_context = self.attn_norm(h_self + attn_out)
        fused = torch.cat([h_self, h_context], dim=-1)
        features = self.trunk(fused)

        val = self.value_stream(features)
        adv = self.advantage_stream(features)

        if masks is not None:
            valid = (masks > 0.5).float()
            valid_count = valid.sum(dim=-1, keepdim=True).clamp(min=1.0)
            adv_mean = (adv * valid).sum(dim=-1, keepdim=True) / valid_count
            q_values = val + (adv - adv_mean)
            q_values = q_values.masked_fill(masks < 0.5, -1e9)
        else:
            q_values = val + (adv - adv.mean(dim=-1, keepdim=True))

        if single:
            return q_values.squeeze(0)
        return q_values

    @torch.no_grad()
    def act_many(
        self, handles: List[int], observations: List[Any], **kwargs
    ) -> Dict[int, RailEnvActions]:
        if not handles:
            return {}
        obs_batch = torch.from_numpy(np.asarray(observations, dtype=np.float32))
        features = obs_batch[:, : self.obs_size]
        raw_masks = obs_batch[:, self.obs_size : self.obs_size + self.n_actions]
        masks = sanitize_action_masks(features, raw_masks)

        q_values = self(features, masks)
        actions = q_values.argmax(dim=-1).tolist()
        return {handle: action for handle, action in zip(handles, actions)}

    @torch.no_grad()
    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        obs_t = torch.from_numpy(np.asarray(observation, dtype=np.float32)).unsqueeze(0)
        features = obs_t[:, : self.obs_size]
        raw_masks = obs_t[:, self.obs_size : self.obs_size + self.n_actions]
        masks = sanitize_action_masks(features, raw_masks)

        q_values = self(features, masks)
        return int(q_values.argmax(dim=-1).item())


MyPolicy = AttentionDuelingDQN