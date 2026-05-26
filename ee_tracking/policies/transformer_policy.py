"""Transformer-based residual policy for delay-compensating control.

Structural motivation
---------------------
The 95-dim observation has natural sequence structure that a flat MLP must
discover implicitly:

    robot_state (30 dims) — q, qdot, ee_pos, ee_err, target_vel, ik_qdot
    fine[0..4]  ( 3 dims) — target position at t + (i+1) × 20 ms  (delay window)
    coarse[0..3]( 3 dims) — target position at t + fine_end + (i+1) × 100 ms (trend)
    cmd[0..4]   ( 7 dims) — joint corrections queued in the FIFO (oldest→newest)
    traj_onehot ( 3 dims) — trajectory type one-hot (at END of obs vector)

Key structural relationship: cmd[i] (the i-th oldest pending command) will
execute when the target is at fine[i].  The policy must answer: "Given that
cmd[i] is already queued for execution at time i, what additional correction
is needed NOW?"

This is exactly an attention problem:
    Query  = current robot state + trajectory trend
    Key/V  = (fine[i], cmd[i]) paired slot tokens for i = 0..4

The MLP must discover this pairing from a 95-D flat vector; the Transformer
has it wired in by construction via paired slot tokens.

Architecture: DelayTransformerPolicy
--------------------------------------
    slot_token[i] = Linear(concat(fine[i], cmd[i]))   # 10 → d_model
         ↓
    + learned positional embedding (slot 0 = soonest, slot 4 = latest)
         ↓
    TransformerEncoder(n_enc_layers, d_model, nhead)   # Pre-LN self-attn over 5 slots
         ↓
    state_token = Linear(concat(robot_state, coarse_flat, traj_onehot))
         ↓
    Pre-LN Cross-attention: state_token queries over slot_tokens
         ↓
    concat(state_token, mean(slot_tokens))             # 2 × d_model
         ↓
    Actor MLP  → 7-D action
    Critic MLP → scalar value  (deeper)

Obs layout (must match env._compute_observation order):
    [robot_state(N_ROBOT) | fine(n_fine×3) | coarse(n_coarse×3) | cmd(n_cmd×7) | traj_onehot(n_pool)]
    N_ROBOT = 7+7+3+3+3+7 = 30  (fixed; independent of config)
    n_pool  = len(trajectory_pool) = inferred as feature_dim - N_ROBOT - sequences

Config YAML example
-------------------
    train:
      policy: "DelayTransformerPolicy"
      policy_kwargs:
        d_model: 64       # token embedding dim
        nhead: 4          # attention heads (must divide d_model)
        n_enc_layers: 2   # self-attention layers over slot tokens
        n_xattn_layers: 1 # cross-attention layers (state ← slots)
        actor_arch: [128, 128]
        critic_arch: [128, 128, 128]
        # Must match env config:
        n_fine: 5         # lookahead_horizon
        n_coarse: 4       # lookahead_coarse_horizon
        n_cmd: 5          # cmd_delay (FIFO depth)
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


# ── constants ──────────────────────────────────────────────────────────────────

# Fixed robot-state dims in obs: q(7)+qdot(7)+ee_pos(3)+ee_err(3)+target_vel(3)+ik_qdot(7)
N_ROBOT_STATE = 30


# ── building blocks ────────────────────────────────────────────────────────────

def make_proj(in_dim: int, out_dim: int, use_mlp: bool) -> nn.Module:
    """Linear or 2-layer MLP projection with LayerNorm+GELU.

    use_mlp=False → single Linear (v1 behaviour)
    use_mlp=True  → Linear → LN → GELU → Linear (non-linear mixing before attention)
    """
    if use_mlp:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
    return nn.Linear(in_dim, out_dim)


class SkipBlock(nn.Module):
    """Linear → LayerNorm → GELU with residual when dims match."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm   = nn.LayerNorm(out_dim)
        self.act    = nn.GELU()
        self.skip   = (in_dim == out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm(self.linear(x)))
        return h + x if self.skip else h


def build_mlp(in_dim: int, arch: List[int]) -> tuple[nn.Sequential, int]:
    blocks: list[nn.Module] = []
    for out_dim in arch:
        blocks.append(SkipBlock(in_dim, out_dim))
        in_dim = out_dim
    return nn.Sequential(*blocks), in_dim


class PreLNCrossAttentionBlock(nn.Module):
    """Pre-LN cross-attention: query attends over context, then Pre-LN FFN.

    Pre-LN (norm before attention/FFN) is more training-stable than Post-LN,
    especially with higher learning rates.  Matches norm_first=True used in
    the slot TransformerEncoderLayer.
    """

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.xattn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.ffn    = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model)
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """query: (B, Lq, d)   context: (B, Lc, d)  →  (B, Lq, d)"""
        # Pre-LN: normalise BEFORE attention, add residual after
        attended, _ = self.xattn(self.norm1(query), context, context)
        query = query + attended
        # Pre-LN FFN
        query = query + self.ffn(self.norm2(query))
        return query


# ── main feature extractor ─────────────────────────────────────────────────────

class DelayTransformerExtractor(nn.Module):
    """Shared feature extractor used by both actor and critic.

    Obs layout assumed (must match env._compute_observation):
        [robot_state(30) | fine(n_fine×3) | coarse(n_coarse×3) | cmd(n_cmd×7) | traj_onehot(n_pool)]

    n_pool is inferred as: feature_dim - N_ROBOT_STATE - n_fine*3 - n_coarse*3 - n_cmd*7
    """

    FINE_DIM   = 3
    COARSE_DIM = 3
    CMD_DIM    = 7

    def __init__(
        self,
        feature_dim: int,
        *,
        n_fine:         int = 5,
        n_coarse:       int = 4,
        n_cmd:          int = 5,
        d_model:        int = 64,
        nhead:          int = 4,
        n_enc_layers:   int = 2,
        n_xattn_layers: int = 1,
        ffn_mult:       int = 2,    # FFN width = ffn_mult × d_model (standard = 4)
        # ── ablation flags ──
        use_pos_embed:    bool = True,   # ablation A: learned PE vs no PE
        use_cross_attn:   bool = True,   # ablation B: cross-attn vs plain concat
        pair_tokens:      bool = True,   # ablation C: paired (fine[i],cmd[i]) vs unpaired
        # ── v2 architecture flags ──
        mlp_proj:         bool = False,  # v2 D: MLP (LN+GELU) projections vs single Linear
        use_reactive:     bool = False,  # v2 E: bypass robot_state → latent (fast reactive path)
        attn_pool:        bool = False,  # v2 F: attention-weighted slot pooling vs mean
    ) -> None:
        super().__init__()

        self.n_fine   = n_fine
        self.n_coarse = n_coarse
        self.n_cmd    = n_cmd
        self.d_model  = d_model
        self.use_pos_embed  = use_pos_embed
        self.use_cross_attn = use_cross_attn
        self.pair_tokens    = pair_tokens
        self.mlp_proj       = mlp_proj
        self.use_reactive   = use_reactive
        self.attn_pool      = attn_pool

        # Infer traj_onehot size from feature_dim (pool size = len(trajectory_pool))
        self.n_pool = feature_dim - N_ROBOT_STATE - n_fine * 3 - n_coarse * 3 - n_cmd * 7

        if self.n_pool < 0:
            raise ValueError(
                f"feature_dim={feature_dim} too small for n_fine={n_fine}, "
                f"n_coarse={n_coarse}, n_cmd={n_cmd}: computed n_pool={self.n_pool}"
            )

        state_in = N_ROBOT_STATE + n_coarse * 3 + self.n_pool  # 30+12+3 = 45
        self.state_proj = make_proj(state_in, d_model, mlp_proj)

        if pair_tokens:
            # ── Ablation C=ON (default): concat fine[i]+cmd[i] → 1 token per slot ──
            slot_in = self.FINE_DIM + self.CMD_DIM       # 3+7 = 10
            self.slot_proj = make_proj(slot_in, d_model, mlp_proj)
            n_tokens = min(n_fine, n_cmd)
        else:
            # ── Ablation C=OFF: fine and cmd encoded independently → 2 tokens/slot ──
            self.fine_proj = make_proj(self.FINE_DIM, d_model, mlp_proj)
            self.cmd_proj  = make_proj(self.CMD_DIM,  d_model, mlp_proj)
            n_tokens = n_fine + n_cmd   # 10 tokens total

        # ── Ablation A: learned positional embedding ──
        if use_pos_embed:
            self.pos_embed = nn.Embedding(n_tokens, d_model)

        # ── slot encoder: Pre-LN self-attention ──
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * ffn_mult,
            dropout=0.0, batch_first=True,
            norm_first=True,
        )
        self.slot_encoder = nn.TransformerEncoder(enc_layer, num_layers=n_enc_layers)

        # ── Ablation B: Pre-LN cross-attention or plain concat ──
        if use_cross_attn:
            self.cross_attn_blocks = nn.ModuleList([
                PreLNCrossAttentionBlock(d_model, nhead, ffn_dim=d_model * ffn_mult)
                for _ in range(n_xattn_layers)
            ])

        # ── v2 E: direct reactive path from robot state ──
        # Gives actor a fast bypass channel that doesn't route through attention.
        # Helps moving_target where immediate reflex matters more than queue planning.
        if use_reactive:
            self.reactive_proj = nn.Sequential(
                nn.Linear(N_ROBOT_STATE, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        # ── v2 F: attention-weighted slot pooling ──
        # Learns a scalar score per slot and pools with softmax weights.
        # Preserves per-slot salience information lost by mean pooling.
        if attn_pool:
            self.slot_pool_attn = nn.Linear(d_model, 1)

        n_paths = 2 + int(use_reactive)   # state_out + slots_out [+ reactive]
        self.latent_dim = d_model * n_paths

    def _split_obs(self, obs: torch.Tensor):
        """Correctly split flat obs into segments using known obs layout.

        Layout: [robot_state(30) | fine(n_fine×3) | coarse(n_coarse×3) | cmd(n_cmd×7) | traj_onehot(n_pool)]
        """
        B = obs.shape[0]
        idx = 0

        robot  = obs[:, idx : idx + N_ROBOT_STATE];                      idx += N_ROBOT_STATE
        fine   = obs[:, idx : idx + self.n_fine   * 3].view(B, self.n_fine,   3); idx += self.n_fine   * 3
        coarse = obs[:, idx : idx + self.n_coarse * 3].view(B, self.n_coarse, 3); idx += self.n_coarse * 3
        cmd    = obs[:, idx : idx + self.n_cmd    * 7].view(B, self.n_cmd,    7); idx += self.n_cmd    * 7
        traj   = obs[:, idx : idx + self.n_pool]

        return robot, fine, coarse, cmd, traj

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        robot, fine, coarse, cmd, traj = self._split_obs(obs)
        B = obs.shape[0]

        # ── build slot tokens ──────────────────────────────────────────────────
        if self.pair_tokens:
            n_slots = min(self.n_fine, self.n_cmd)
            slot_in = torch.cat([fine[:, :n_slots], cmd[:, :n_slots]], dim=-1)  # (B, n_slots, 10)
            slots   = self.slot_proj(slot_in)                                   # (B, n_slots, d)
        else:
            fine_tok = self.fine_proj(fine)
            cmd_tok  = self.cmd_proj(cmd)
            slots    = torch.cat([fine_tok, cmd_tok], dim=1)
            n_slots  = slots.shape[1]

        # ── optional positional embedding ──────────────────────────────────────
        if self.use_pos_embed:
            pos_idx = torch.arange(n_slots, device=obs.device)
            slots   = slots + self.pos_embed(pos_idx).unsqueeze(0)

        # ── Pre-LN self-attention over slot tokens ─────────────────────────────
        slots = self.slot_encoder(slots)                                     # (B, n_slots, d)

        # ── state token ────────────────────────────────────────────────────────
        coarse_flat = coarse.reshape(B, -1)
        state_in    = torch.cat([robot, coarse_flat, traj], dim=-1)
        state_tok   = self.state_proj(state_in).unsqueeze(1)                 # (B, 1, d)

        # ── cross-attention or plain passthrough ───────────────────────────────
        if self.use_cross_attn:
            for blk in self.cross_attn_blocks:
                state_tok = blk(state_tok, slots)

        state_out = state_tok.squeeze(1)                                     # (B, d)

        # ── slot aggregation: mean or attention-weighted pool ──────────────────
        if self.attn_pool:
            # Learned scalar score per slot → softmax weights → weighted sum
            scores    = self.slot_pool_attn(slots).squeeze(-1)               # (B, n_slots)
            weights   = torch.softmax(scores, dim=-1).unsqueeze(-1)          # (B, n_slots, 1)
            slots_out = (slots * weights).sum(dim=1)                         # (B, d)
        else:
            slots_out = slots.mean(dim=1)                                    # (B, d)

        # ── optional reactive bypass: direct robot_state → latent ──────────────
        # Fast path that doesn't route through attention; helps moving_target.
        if self.use_reactive:
            reactive  = self.reactive_proj(robot)                            # (B, d)
            return torch.cat([state_out, slots_out, reactive], dim=-1)       # (B, 3d)

        return torch.cat([state_out, slots_out], dim=-1)                     # (B, 2d)


# ── SB3 policy ─────────────────────────────────────────────────────────────────

class _SplitExtractor(nn.Module):
    """Wraps DelayTransformerExtractor: gives actor/critic separate MLP heads."""

    def __init__(
        self,
        feature_dim:    int,
        actor_arch:     List[int],
        critic_arch:    List[int],
        n_fine:         int = 5,
        n_coarse:       int = 4,
        n_cmd:          int = 5,
        d_model:        int = 64,
        nhead:          int = 4,
        n_enc_layers:   int = 2,
        n_xattn_layers: int = 1,
        # ablation flags (forwarded to DelayTransformerExtractor)
        use_pos_embed:  bool = True,
        use_cross_attn: bool = True,
        pair_tokens:    bool = True,
        # v2 architecture flags
        ffn_mult:       int  = 2,
        mlp_proj:       bool = False,
        use_reactive:   bool = False,
        attn_pool:      bool = False,
    ) -> None:
        super().__init__()

        self.shared = DelayTransformerExtractor(
            feature_dim,
            n_fine=n_fine, n_coarse=n_coarse, n_cmd=n_cmd,
            d_model=d_model, nhead=nhead,
            n_enc_layers=n_enc_layers,
            n_xattn_layers=n_xattn_layers,
            ffn_mult=ffn_mult,
            use_pos_embed=use_pos_embed,
            use_cross_attn=use_cross_attn,
            pair_tokens=pair_tokens,
            mlp_proj=mlp_proj,
            use_reactive=use_reactive,
            attn_pool=attn_pool,
        )
        shared_out = self.shared.latent_dim

        self.actor_net,  self.latent_dim_pi = build_mlp(shared_out, actor_arch)
        self.critic_net, self.latent_dim_vf = build_mlp(shared_out, critic_arch)

    def forward(self, obs: torch.Tensor):
        z = self.shared(obs)
        return self.actor_net(z), self.critic_net(z)

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_net(self.shared(obs))

    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic_net(self.shared(obs))


class DelayTransformerPolicy(ActorCriticPolicy):
    """PPO policy with paired-slot Transformer for delay compensation.

    policy_kwargs (all optional, show defaults):
        d_model:        64    token embedding dimension
        nhead:          4     attention heads (must divide d_model)
        n_enc_layers:   2     Pre-LN self-attention layers over slot tokens
        n_xattn_layers: 1     Pre-LN cross-attention layers (state ← slots)
        actor_arch:     [128, 128]
        critic_arch:    [128, 128, 128]
        # Must match env config:
        n_fine:         5     lookahead_horizon
        n_coarse:       4     lookahead_coarse_horizon
        n_cmd:          5     cmd_delay  (FIFO depth)
    """

    def _build_mlp_extractor(self) -> None:
        kw = self.net_arch if isinstance(self.net_arch, dict) else {}

        self.mlp_extractor = _SplitExtractor(
            self.features_dim,
            actor_arch    = list(kw.get("actor_arch",    [128, 128])),
            critic_arch   = list(kw.get("critic_arch",   [128, 128, 128])),
            n_fine        = int(kw.get("n_fine",           5)),
            n_coarse      = int(kw.get("n_coarse",          4)),
            n_cmd         = int(kw.get("n_cmd",             5)),
            d_model       = int(kw.get("d_model",          64)),
            nhead         = int(kw.get("nhead",             4)),
            n_enc_layers  = int(kw.get("n_enc_layers",      2)),
            n_xattn_layers= int(kw.get("n_xattn_layers",   1)),
            use_pos_embed = bool(kw.get("use_pos_embed",  True)),
            use_cross_attn= bool(kw.get("use_cross_attn", True)),
            pair_tokens   = bool(kw.get("pair_tokens",    True)),
            ffn_mult      = int(kw.get("ffn_mult",         2)),
            mlp_proj      = bool(kw.get("mlp_proj",      False)),
            use_reactive  = bool(kw.get("use_reactive",  False)),
            attn_pool     = bool(kw.get("attn_pool",     False)),
        )


# ── registry export ────────────────────────────────────────────────────────────

TRANSFORMER_POLICY_REGISTRY: dict[str, type] = {
    "DelayTransformerPolicy": DelayTransformerPolicy,
}
